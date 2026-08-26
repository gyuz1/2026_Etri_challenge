"""A finite, distributed throughput runner for the cached ETRI training path.

The normal VAD training entry point uses an epoch-based runner.  This runner
keeps all of its training hooks (including FP16 and the optimizer), but stops
cleanly after a fixed warmup and measurement window.  It is deliberately
registered only from the benchmark config.
"""

import json
import math
import os
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import EpochBasedRunner, RUNNERS, get_dist_info


HARD_SAMPLES_PER_SECOND = 3.329
SAFE_SAMPLES_PER_SECOND = 3.70


def _atomic_json_dump(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix='.throughput-', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _timing_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return dict(count=0)
    return dict(
        count=int(values.size),
        mean_ms=float(values.mean() * 1000.0),
        p50_ms=float(np.quantile(values, 0.50) * 1000.0),
        p95_ms=float(np.quantile(values, 0.95) * 1000.0),
        p99_ms=float(np.quantile(values, 0.99) * 1000.0),
        max_ms=float(values.max() * 1000.0))


def _distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@RUNNERS.register_module()
class VADThroughputBenchmarkRunner(EpochBasedRunner):
    """Run exactly ``warmup_iters + measure_iters`` optimizer iterations.

    The wall-clock window starts after the last warmup optimizer step and
    before the next DataLoader fetch.  Thus aggregate samples/s includes data
    loading, forward, backward, all-reduce, and optimizer work.  CUDA is only
    explicitly synchronized at the two window boundaries, avoiding a forced
    synchronization on every measured iteration.
    """

    def __init__(self, *args, benchmark_warmup_iters=50,
                 benchmark_measure_iters=300, expected_world_size=2,
                 expected_start_epoch=4, expected_samples_per_gpu=2,
                 expected_workers_per_gpu=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_warmup_iters = int(benchmark_warmup_iters)
        self.benchmark_measure_iters = int(benchmark_measure_iters)
        self.expected_world_size = int(expected_world_size)
        self.expected_start_epoch = int(expected_start_epoch)
        self.expected_samples_per_gpu = int(expected_samples_per_gpu)
        self.expected_workers_per_gpu = (
            None if expected_workers_per_gpu is None
            else int(expected_workers_per_gpu))
        if self.benchmark_warmup_iters < 1:
            raise ValueError('benchmark_warmup_iters must be at least 1')
        if self.benchmark_measure_iters < 1:
            raise ValueError('benchmark_measure_iters must be at least 1')
        self._benchmark_finished = False

    @staticmethod
    def _loss_anomalies(outputs, local_iteration, phase):
        context = dict(
            local_iteration=int(local_iteration), phase=phase)
        if not isinstance(outputs, dict):
            return [dict(
                context, name='outputs', value='missing_or_not_dict')]
        # MMDetection's train_step has already converted log_vars to Python
        # scalars. Do not inspect/copy outputs['loss'] here: calling .item() or
        # .cpu() on its CUDA tensor would force an extra synchronization and
        # contaminate the throughput measurement.
        log_vars = outputs.get('log_vars')
        if not isinstance(log_vars, dict):
            return [dict(
                context, name='log_vars', value='missing_or_not_dict')]
        anomalies = []
        if 'loss' not in log_vars:
            anomalies.append(dict(
                context, name='loss', value='missing_from_log_vars'))
        for name, value in log_vars.items():
            # VAD/MMDetection already returns Python/NumPy scalars here. Do
            # not coerce an unexpected CUDA tensor because that would insert
            # an extra synchronization into every benchmark iteration.
            if not isinstance(value, (int, float, np.number)):
                anomalies.append(dict(
                    context, name=str(name),
                    value=f'non_scalar:{type(value).__name__}'))
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                anomalies.append(dict(
                    context,
                    name=str(name),
                    value=repr(numeric)))
        return anomalies

    def _validate_run(self, data_loader):
        rank, world_size = get_dist_info()
        if world_size != self.expected_world_size:
            raise RuntimeError(
                f'Benchmark requires world_size={self.expected_world_size}, '
                f'got {world_size}')
        if self.epoch != self.expected_start_epoch:
            raise RuntimeError(
                f'Benchmark requires a resumed checkpoint at epoch '
                f'{self.expected_start_epoch}, got runner.epoch={self.epoch}')
        if data_loader.batch_size != self.expected_samples_per_gpu:
            raise RuntimeError(
                f'Expected batch/GPU={self.expected_samples_per_gpu}, got '
                f'DataLoader batch_size={data_loader.batch_size}')
        if self.expected_workers_per_gpu is not None:
            if data_loader.num_workers != self.expected_workers_per_gpu:
                raise RuntimeError(
                    f'Expected workers/GPU={self.expected_workers_per_gpu}, '
                    f'got {data_loader.num_workers}')
        if not data_loader.persistent_workers:
            raise RuntimeError(
                'Benchmark requires persistent_workers=True')
        required = (
            self.benchmark_warmup_iters + self.benchmark_measure_iters)
        if len(data_loader) < required:
            raise RuntimeError(
                f'DataLoader has only {len(data_loader)} iterations; '
                f'{required} are required')
        logical_iter = int(self.epoch) * len(data_loader)
        resumed_iter = int(self.iter)
        # Never conceal a checkpoint prepared for another batch size. The
        # preparation utility must have written exactly epoch * loader length.
        if resumed_iter != logical_iter:
            raise RuntimeError(
                f'Prepared checkpoint iter mismatch: meta.iter={resumed_iter}, '
                f'but epoch={self.epoch} and loader_len={len(data_loader)} '
                f'require {logical_iter}. Refusing to rewrite runner._iter.')
        return rank, world_size, resumed_iter, logical_iter

    def _optimizer_lrs(self):
        optimizers = self._optimizers()
        return {
            str(name): [float(group['lr']) for group in optimizer.param_groups]
            for name, optimizer in optimizers
        }

    def _optimizers(self):
        if isinstance(self.optimizer, dict):
            return list(self.optimizer.items())
        return [('optimizer', self.optimizer)]

    @staticmethod
    def _scalar_optimizer_step(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(
                    f'Optimizer step must be scalar, got shape '
                    f'{tuple(value.shape)}')
            value = value.item()
        if not isinstance(value, (int, float, np.number)):
            raise RuntimeError(
                f'Unexpected optimizer step type: {type(value).__name__}')
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise RuntimeError(f'Invalid optimizer step value: {value!r}')
        return int(numeric)

    def _capture_optimizer_step_state(self):
        """Track parameters that already have Adam state at resume.

        The 82 tensors restored by the autocast-cache fix create state lazily,
        so end-of-window state as a whole has mixed ages. Existing state is
        uniform in epoch_4 and provides an exact global GradScaler skip count.
        """
        tracked = {}
        summary = {}
        for name, optimizer in self._optimizers():
            parameters = []
            steps = []
            for parameter, state in optimizer.state.items():
                if not isinstance(state, dict) or 'step' not in state:
                    continue
                parameters.append(parameter)
                steps.append(self._scalar_optimizer_step(state['step']))
            if not steps:
                raise RuntimeError(
                    f'Optimizer {name!r} has no resumable step state')
            unique_steps = sorted(set(steps))
            if len(unique_steps) != 1:
                raise RuntimeError(
                    f'Optimizer {name!r} has non-uniform resume steps: '
                    f'{unique_steps}')
            tracked[str(name)] = (optimizer, parameters)
            summary[str(name)] = dict(
                tracked_state_entries=len(parameters),
                min_step=min(steps), max_step=max(steps))
        return tracked, summary

    def _finish_optimizer_step_audit(self, tracked, start, expected_updates):
        optimizers = {}
        valid = True
        for name, (optimizer, parameters) in tracked.items():
            end_steps = []
            for parameter in parameters:
                state = optimizer.state.get(parameter)
                if not isinstance(state, dict) or 'step' not in state:
                    raise RuntimeError(
                        f'Optimizer {name!r} lost tracked step state')
                end_steps.append(self._scalar_optimizer_step(state['step']))
            start_step = int(start[name]['min_step'])
            min_end = min(end_steps)
            max_end = max(end_steps)
            min_delta = min_end - start_step
            max_delta = max_end - start_step
            optimizer_valid = (
                min_end == max_end
                and min_delta == int(expected_updates))
            valid = valid and optimizer_valid
            optimizers[name] = dict(
                tracked_state_entries=len(parameters),
                start_step=start_step,
                min_end_step=min_end,
                max_end_step=max_end,
                min_update_delta=min_delta,
                max_update_delta=max_delta,
                expected_updates=int(expected_updates),
                skipped_updates=int(expected_updates) - min_delta,
                valid=bool(optimizer_valid))
        return dict(valid=bool(valid), optimizers=optimizers)

    @staticmethod
    def _validate_no_lr_jump(before, after):
        if before.keys() != after.keys():
            raise RuntimeError(
                f'Optimizer names changed across before_train_epoch: '
                f'{before.keys()} -> {after.keys()}')
        mismatches = []
        for name in before:
            if len(before[name]) != len(after[name]):
                mismatches.append(dict(
                    optimizer=name, before=before[name], after=after[name]))
                continue
            for index, (old, new) in enumerate(
                    zip(before[name], after[name])):
                if not math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-12):
                    mismatches.append(dict(
                        optimizer=name, group=index, before=old, after=new))
        if mismatches:
            raise RuntimeError(
                'LR changed at resume before the first benchmark optimizer '
                f'step. Prepare a 48-epoch checkpoint with '
                f'prepare_resume_checkpoint.py: {mismatches}')

    def _start_measurement(self):
        _cuda_synchronize()
        _distributed_barrier()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        started_unix = time.time()
        return started, started_unix

    def _local_payload(self, rank, world_size, data_loader, started,
                       finished, started_unix, finished_unix,
                       iteration_times, data_times, step_times,
                       anomalies, resumed_iter, logical_iter, resume_lrs,
                       optimizer_step_audit):
        elapsed = float(finished - started)
        gpu = dict(available=bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu.update(
                logical_device=int(device),
                name=torch.cuda.get_device_name(device),
                allocated_at_end_bytes=int(torch.cuda.memory_allocated(device)),
                reserved_at_end_bytes=int(torch.cuda.memory_reserved(device)),
                peak_allocated_bytes=int(
                    torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(
                    torch.cuda.max_memory_reserved(device)))
        return dict(
            rank=int(rank),
            world_size=int(world_size),
            epoch_at_start=int(self.expected_start_epoch),
            checkpoint_meta_iter=int(resumed_iter),
            logical_iter_at_start=int(logical_iter),
            global_iter_at_end=int(self.iter),
            resume_lrs=resume_lrs,
            optimizer_step_audit=optimizer_step_audit,
            samples_per_gpu=int(data_loader.batch_size),
            workers_per_gpu=int(data_loader.num_workers),
            persistent_workers=bool(data_loader.persistent_workers),
            warmup_iters=int(self.benchmark_warmup_iters),
            measured_iters=int(self.benchmark_measure_iters),
            elapsed_seconds=elapsed,
            measurement_started_unix=float(started_unix),
            measurement_finished_unix=float(finished_unix),
            iteration_seconds=iteration_times,
            data_seconds=data_times,
            step_submit_seconds=step_times,
            loss_anomalies=anomalies,
            gpu=gpu)

    def _write_results(self, payload):
        rank = payload['rank']
        # Gather first. A rank must not publish a seemingly complete artifact
        # while another rank is missing, malformed, or stuck.
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(payload['world_size'])]
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]

        measured = self.benchmark_measure_iters
        for item in gathered:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f'Rank payload is missing or malformed: {item!r}')
            if len(item['iteration_seconds']) != measured:
                raise RuntimeError(
                    f"Rank {item['rank']} recorded "
                    f"{len(item['iteration_seconds'])}/{measured} iterations")

        rank_path = os.path.join(
            self.work_dir, f'throughput_rank{rank}.json')
        _atomic_json_dump(payload, rank_path)

        all_anomalies = []
        for item in gathered:
            for anomaly in item['loss_anomalies']:
                annotated = dict(anomaly)
                annotated['rank'] = item['rank']
                all_anomalies.append(annotated)

        if rank != 0:
            if all_anomalies:
                raise RuntimeError(
                    f'Non-finite/malformed loss telemetry detected: '
                    f'{len(all_anomalies)} records')
            return

        # DDP advances at the pace of the slowest rank.  Per-iteration maxima
        # expose the critical path without pretending CPU submit time is a
        # synchronized CUDA kernel duration.
        critical_iteration = [
            max(item['iteration_seconds'][index] for item in gathered)
            for index in range(measured)]
        critical_data = [
            max(item['data_seconds'][index] for item in gathered)
            for index in range(measured)]
        critical_step = [
            max(item['step_submit_seconds'][index] for item in gathered)
            for index in range(measured)]
        elapsed = max(item['elapsed_seconds'] for item in gathered)
        measurement_started_unix = max(
            item['measurement_started_unix'] for item in gathered)
        measurement_finished_unix = min(
            item['measurement_finished_unix'] for item in gathered)
        global_batch = (
            payload['samples_per_gpu'] * payload['world_size'])
        measured_samples = measured * global_batch
        samples_per_second = measured_samples / elapsed
        if samples_per_second >= SAFE_SAMPLES_PER_SECOND:
            threshold_status = 'safe'
        elif samples_per_second >= HARD_SAMPLES_PER_SECOND:
            threshold_status = 'hard_only'
        else:
            threshold_status = 'fail'

        overall_status = (
            'invalid_loss' if all_anomalies else threshold_status)

        summary = dict(
            schema_version=1,
            benchmark='VAD cached train throughput',
            warmup_iters=int(self.benchmark_warmup_iters),
            measured_iters=int(measured),
            world_size=int(payload['world_size']),
            samples_per_gpu=int(payload['samples_per_gpu']),
            global_batch_size=int(global_batch),
            workers_per_gpu=int(payload['workers_per_gpu']),
            persistent_workers=bool(payload['persistent_workers']),
            checkpoint_meta_iter=int(payload['checkpoint_meta_iter']),
            logical_iter_at_start=int(payload['logical_iter_at_start']),
            global_iter_at_end=int(payload['global_iter_at_end']),
            resume_lrs=payload['resume_lrs'],
            optimizer_step_audits=[dict(
                rank=item['rank'], audit=item['optimizer_step_audit'])
                for item in gathered],
            measured_samples=int(measured_samples),
            elapsed_seconds=float(elapsed),
            measurement_window_unix=dict(
                started=float(measurement_started_unix),
                finished=float(measurement_finished_unix)),
            aggregate_samples_per_second=float(samples_per_second),
            thresholds=dict(
                hard_samples_per_second=HARD_SAMPLES_PER_SECOND,
                safe_samples_per_second=SAFE_SAMPLES_PER_SECOND,
                status=threshold_status),
            overall_status=overall_status,
            timings=dict(
                critical_iteration_wall=_timing_summary(
                    critical_iteration),
                critical_data_wait=_timing_summary(critical_data),
                critical_step_submit=_timing_summary(critical_step),
                note=(
                    'Aggregate throughput is boundary-synchronized wall '
                    'time. Per-iteration data/step values are CPU-observed; '
                    'CUDA is not synchronized every iteration.')),
            loss_anomalies=dict(
                count=len(all_anomalies), records=all_anomalies),
            ranks=[dict(
                rank=item['rank'],
                elapsed_seconds=item['elapsed_seconds'],
                gpu=item['gpu']) for item in gathered])
        _atomic_json_dump(
            summary, os.path.join(self.work_dir, 'throughput_summary.json'))
        self.logger.info(
            'THROUGHPUT_RESULT samples/s=%.6f status=%s loss_anomalies=%d',
            samples_per_second, overall_status, len(all_anomalies))
        if all_anomalies:
            raise RuntimeError(
                f'Benchmark rejected due to non-finite/malformed loss '
                f'telemetry: {len(all_anomalies)} records')

    def train(self, data_loader, **kwargs):
        if self._benchmark_finished:
            return
        rank, world_size, resumed_iter, logical_iter = self._validate_run(
            data_loader)
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        # Capture the checkpoint LR and ensure CosineAnnealing's first epoch
        # update reproduces it. This fails closed if a raw, unadjusted epoch_4
        # checkpoint is accidentally supplied for the 48-epoch runner.
        lrs_before_epoch_hook = self._optimizer_lrs()
        self.call_hook('before_train_epoch')
        lrs_after_epoch_hook = self._optimizer_lrs()
        self._validate_no_lr_jump(
            lrs_before_epoch_hook, lrs_after_epoch_hook)
        resume_lrs = dict(
            before_epoch_hook=lrs_before_epoch_hook,
            after_epoch_hook=lrs_after_epoch_hook)
        time.sleep(2)  # Match MMCV EpochBasedRunner transition behavior.

        tracked_steps, optimizer_steps_at_start = (
            self._capture_optimizer_step_state())

        completed = 0
        measured = 0
        started = None
        previous_end = None
        iteration_times = []
        data_times = []
        step_times = []
        anomalies = []
        total = self.benchmark_warmup_iters + self.benchmark_measure_iters

        for index, data_batch in enumerate(self.data_loader):
            fetched = time.perf_counter()
            self._inner_iter = index
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            completed += 1
            phase = (
                'warmup' if completed <= self.benchmark_warmup_iters
                else 'measure')
            anomalies.extend(self._loss_anomalies(
                self.outputs, completed, phase))
            self._iter += 1

            if completed == self.benchmark_warmup_iters:
                started, started_unix = self._start_measurement()
                previous_end = started
                continue

            if completed > self.benchmark_warmup_iters:
                measured += 1
                is_final = measured == self.benchmark_measure_iters
                if is_final:
                    _cuda_synchronize()
                finished = time.perf_counter()
                finished_unix = time.time()
                data_times.append(max(0.0, fetched - previous_end))
                step_times.append(max(0.0, finished - fetched))
                iteration_times.append(max(0.0, finished - previous_end))
                previous_end = finished
                if is_final:
                    optimizer_step_audit = self._finish_optimizer_step_audit(
                        tracked_steps, optimizer_steps_at_start, total)
                    if not optimizer_step_audit['valid']:
                        anomalies.append(dict(
                            context=dict(
                                local_iteration=int(completed), phase='all'),
                            name='optimizer_step_audit',
                            value=json.dumps(
                                optimizer_step_audit, sort_keys=True)))
                    expected_end = logical_iter + total
                    if self.iter != expected_end:
                        raise RuntimeError(
                            f'Benchmark iteration accounting mismatch: '
                            f'expected end {expected_end}, got {self.iter}')
                    payload = self._local_payload(
                        rank, world_size, data_loader, started, finished,
                        started_unix, finished_unix,
                        iteration_times, data_times, step_times, anomalies,
                        resumed_iter, logical_iter, resume_lrs,
                        optimizer_step_audit)
                    self._write_results(payload)
                    self._benchmark_finished = True
                    break

            if completed >= total:
                raise AssertionError('Benchmark loop exceeded its target')

        if not self._benchmark_finished:
            raise RuntimeError(
                f'DataLoader ended after {completed}/{total} benchmark '
                'iterations')

        self.call_hook('after_train_epoch')
        self._epoch += 1
        # EpochBasedRunner.run has no stop flag.  The benchmark has produced
        # its complete result, so move to max_epochs without starting another
        # partial epoch.  No checkpoint hook is installed in the benchmark.
        self._epoch = self._max_epochs
