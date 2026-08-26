#!/usr/bin/env python3
"""Launch collision-free VAD cached-data throughput A/B benchmarks.

Run this inside the ``ad2026`` container after the complete 344-scene cache
manifest exists. Each case reads a separately prepared resume checkpoint and
uses an independent work directory. The original epoch_4.pth is never loaded
for training and is protected by before/after SHA-256 verification. No
training checkpoint or validation output is produced.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

DEFAULT_CONFIG = (
    'tools/optimization/VAD_etri_tiny_stage1_cached_benchmark.py')
DEFAULT_SOURCE_CHECKPOINT = 'work_dirs/stage1_etri/epoch_4.pth'
DEFAULT_CACHE_ROOT = '/workspace/VAD/work_dirs/etri_geometry_cache_v1'
DEFAULT_OUTPUT_ROOT = 'work_dirs/throughput_sweep'
DEFAULT_ANN_FILE = (
    '/workspace/VAD/data/etri/.causal_regen/'
    'vad_etri_infos_temporal_train_split.pkl')
EXPECTED_SCENES = 344
EXPECTED_CACHE_VERSION = 1
EXPECTED_FRAME_STRIDE = 5
EXPECTED_CROP_SIZE = [1920, 1080]
EXPECTED_SCALE = 0.4
EXPECTED_CROP_KEEP_TOP = [
    'camera_front_left', 'camera_front_right', 'camera_rear_left',
    'camera_rear_right', 'camera_rear_wide']
EXPECTED_GROUP_SIZES = [19608]
EXPECTED_WORLD_SIZE = 2
EXPECTED_CHECKPOINT_EPOCH = 4
MIN_FREE_BYTES = 10 * 1024 ** 3


def _csv_ints(value):
    try:
        result = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'Expected comma-separated integers, got {value!r}') from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            f'Expected non-negative comma-separated integers, got {value!r}')
    return result


def _sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def _validate_cache(cache_root):
    manifest_path = Path(cache_root) / 'cache_manifest.json'
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f'Complete cache manifest is required: {manifest_path}') from error
    requested = manifest.get('requested_scene_count')
    completed = manifest.get('completed_scene_count')
    listed = len(manifest.get('scenes', []))
    expected_ann = os.path.realpath(DEFAULT_ANN_FILE)
    actual_ann = os.path.realpath(str(manifest.get('ann_file', '')))
    invariants = dict(
        cache_version=(
            manifest.get('cache_version') == EXPECTED_CACHE_VERSION),
        ann_file=(actual_ann == expected_ann),
        frame_stride=(
            manifest.get('frame_stride') == EXPECTED_FRAME_STRIDE),
        crop_size=(manifest.get('crop_size') == EXPECTED_CROP_SIZE),
        crop_keep_top=(
            manifest.get('crop_keep_top') == EXPECTED_CROP_KEEP_TOP),
        scale=(manifest.get('scale') == EXPECTED_SCALE),
        scene_counts=(
            requested == completed == listed == EXPECTED_SCENES))
    if not all(invariants.values()):
        raise RuntimeError(
            f'Cache manifest failed strict preflight: {invariants}; '
            f'requested={requested}, '
            f'completed={completed}, listed={listed}, '
            f'expected={EXPECTED_SCENES}')
    scene_entries = manifest['scenes']
    tokens = [str(item.get('scene_token')) for item in scene_entries]
    if len(tokens) != len(set(tokens)):
        raise RuntimeError('Cache manifest contains duplicate scene tokens')
    missing_shards = 0
    missing_metadata = 0
    bytes_mismatch = 0
    invalid_metadata = 0
    for item in scene_entries:
        stem = hashlib.sha1(
            str(item['scene_token']).encode('utf-8')).hexdigest()
        shard_path = Path(cache_root) / f'{stem}.npy'
        metadata_path = Path(cache_root) / f'{stem}.json'
        if not shard_path.is_file():
            missing_shards += 1
        elif shard_path.stat().st_size != int(item['bytes']):
            bytes_mismatch += 1
        if not metadata_path.is_file():
            missing_metadata += 1
        else:
            try:
                with open(metadata_path) as handle:
                    metadata = json.load(handle)
                metadata_valid = (
                    metadata.get('cache_version') == EXPECTED_CACHE_VERSION
                    and metadata.get('complete') is True
                    and str(metadata.get('scene_token'))
                    == str(item['scene_token'])
                    and metadata.get('dtype') == 'uint8'
                    and metadata.get('crop_size') == EXPECTED_CROP_SIZE
                    and metadata.get('crop_keep_top')
                    == EXPECTED_CROP_KEEP_TOP
                    and metadata.get('scale') == EXPECTED_SCALE
                    and metadata.get('shape') == [60, 6, 432, 768, 3]
                    and len(metadata.get('frame_indices', [])) == 60
                    and len(metadata.get('camera_names', [])) == 6)
                if not metadata_valid:
                    invalid_metadata += 1
            except (OSError, ValueError, json.JSONDecodeError):
                invalid_metadata += 1
    if (missing_shards or missing_metadata or bytes_mismatch
            or invalid_metadata):
        raise RuntimeError(
            'Cache files failed structural preflight: '
            f'missing_shards={missing_shards}, '
            f'missing_metadata={missing_metadata}, '
            f'bytes_mismatch={bytes_mismatch}, '
            f'invalid_metadata={invalid_metadata}')
    return dict(
        manifest=str(manifest_path),
        cache_version=manifest.get('cache_version'),
        requested_scene_count=requested,
        completed_scene_count=completed,
        frame_stride=manifest.get('frame_stride'),
        crop_size=manifest.get('crop_size'),
        scale=manifest.get('scale'),
        shard_files_verified=listed)


def _validate_checkpoint(path, *, require_resume_preparation,
                         expected_samples_per_gpu=None,
                         expected_source_sha256=None):
    # Keep --dry-run usable on the host; torch is required only in the
    # container when the real checkpoint preflight is performed.
    import torch

    checkpoint = torch.load(str(path), map_location='cpu')
    meta = checkpoint.get('meta', {})
    epoch = meta.get('epoch')
    iteration = meta.get('iter')
    resume_preparation = meta.get('resume_preparation')
    del checkpoint
    if epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError(
            f'Expected epoch_4 checkpoint metadata epoch=4, got {epoch!r}')
    if require_resume_preparation:
        if expected_samples_per_gpu is None or expected_samples_per_gpu <= 0:
            raise ValueError(
                'expected_samples_per_gpu must be positive for a prepared '
                'checkpoint')
        if not expected_source_sha256:
            raise ValueError(
                'expected_source_sha256 is required for a prepared checkpoint')
        if not isinstance(resume_preparation, dict):
            raise RuntimeError(
                'Benchmark checkpoint must be a separately prepared resume '
                'checkpoint containing meta.resume_preparation')
        denominator = EXPECTED_WORLD_SIZE * expected_samples_per_gpu
        expected_iters_per_epoch = sum(
            (size + denominator - 1) // denominator
            for size in EXPECTED_GROUP_SIZES)
        expected_iter = (
            EXPECTED_CHECKPOINT_EPOCH * expected_iters_per_epoch)
        expected = dict(
            epoch=EXPECTED_CHECKPOINT_EPOCH,
            max_epochs=48,
            runtime_world_size=EXPECTED_WORLD_SIZE,
            samples_per_gpu=expected_samples_per_gpu,
            group_sizes=EXPECTED_GROUP_SIZES,
            iters_per_epoch=expected_iters_per_epoch,
            new_meta_iter=expected_iter,
            source_checkpoint_sha256=expected_source_sha256)
        actual = {key: resume_preparation.get(key) for key in expected}
        mismatches = {
            key: dict(expected=expected[key], actual=actual[key])
            for key in expected if actual[key] != expected[key]}
        if iteration != expected_iter:
            mismatches['meta.iter'] = dict(
                expected=expected_iter, actual=iteration)
        if mismatches:
            raise RuntimeError(
                'Prepared checkpoint does not match this batch/source: '
                f'{json.dumps(mismatches, sort_keys=True)}')
    return dict(
        epoch=int(epoch), iteration=iteration,
        resume_preparation=resume_preparation)


def _active_training_processes():
    result = subprocess.run(
        ['ps', '-eo', 'pid=,args='], capture_output=True, text=True,
        check=True)
    own_pid = os.getpid()
    records = []
    markers = (
        'tools/train.py', 'dist_train.sh', 'torch.distributed.launch',
        'torchrun')
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(' ')
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != own_pid and any(marker in command for marker in markers):
            records.append(dict(pid=pid, command=command))
    return records


def _gpu_idle_preflight(gpu_ids):
    query = subprocess.run(
        [
            'nvidia-smi',
            f"--id={','.join(str(item) for item in gpu_ids)}",
            '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, check=True)
    records = []
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(',')]
        if len(fields) != 3:
            raise RuntimeError(f'Unexpected nvidia-smi output: {line!r}')
        records.append(dict(
            gpu_index=int(fields[0]), memory_used_mib=float(fields[1]),
            utilization_percent=float(fields[2])))
    if len(records) != len(gpu_ids):
        raise RuntimeError(
            f'Expected {len(gpu_ids)} GPU rows, got {len(records)}')

    processes = subprocess.run(
        [
            'nvidia-smi',
            f"--id={','.join(str(item) for item in gpu_ids)}",
            '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, check=True)
    compute_processes = [
        line.strip() for line in processes.stdout.splitlines()
        if line.strip()]
    # On an otherwise idle display-capable GPU a few MiB may be reserved by
    # the driver. Any CUDA compute process or meaningful allocation is a hard
    # stop; utilization is sampled and reported but can be transient.
    busy_memory = [item for item in records if item['memory_used_mib'] > 512]
    busy_utilization = [
        item for item in records if item['utilization_percent'] > 10]
    if compute_processes or busy_memory or busy_utilization:
        raise RuntimeError(
            f'GPUs are not idle: gpu_records={records}, '
            f'compute_processes={compute_processes}')
    return dict(gpus=records, compute_processes=compute_processes)


def _system_preflight(output_root, gpu_ids):
    training = _active_training_processes()
    if training:
        raise RuntimeError(
            f'Another training/DDP process is active: {training}')
    disk = shutil_disk_usage(output_root)
    if disk.free < MIN_FREE_BYTES:
        raise RuntimeError(
            f'Insufficient work-dir disk: free={disk.free:,}, '
            f'required={MIN_FREE_BYTES:,}')
    return dict(
        training_processes=training,
        disk=dict(
            path=str(output_root), total_bytes=disk.total,
            used_bytes=disk.used, free_bytes=disk.free,
            required_free_bytes=MIN_FREE_BYTES),
        gpu=_gpu_idle_preflight(gpu_ids))


def _storage_identity(path):
    """Describe the cache filesystem/device without rejecting HDD storage."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity = dict(
        path=str(path), realpath=str(resolved), st_dev=int(stat.st_dev))
    findmnt = subprocess.run(
        ['findmnt', '-T', str(resolved), '-n', '-o',
         'SOURCE,FSTYPE,TARGET'],
        capture_output=True, text=True, check=False)
    if findmnt.returncode == 0 and findmnt.stdout.strip():
        fields = findmnt.stdout.strip().split(None, 2)
        if len(fields) == 3:
            source, filesystem_type, mount_target = fields
            identity.update(
                source=source, filesystem_type=filesystem_type,
                mount_target=mount_target)
            # bind-mounted subdirectories may be reported as
            # ``/dev/sda1[/path/inside/fs]``; lsblk needs the block path only.
            block_source = source.split('[', 1)[0]
            if block_source.startswith('/dev/'):
                # Containers often omit /dev/sd* nodes even though lsblk can
                # see host topology through sysfs, so resolve by NAME in JSON.
                block = subprocess.run(
                    ['lsblk', '-J', '-o',
                     'NAME,PKNAME,MODEL,ROTA,TYPE'],
                    capture_output=True, text=True, check=False)
                if block.returncode == 0 and block.stdout.strip():
                    try:
                        payload = json.loads(block.stdout)
                        nodes = {}

                        def collect(items):
                            for item in items:
                                nodes[item.get('name')] = item
                                collect(item.get('children', []))

                        collect(payload.get('blockdevices', []))
                        leaf = nodes.get(Path(block_source).name)
                        if leaf is not None:
                            parent = nodes.get(leaf.get('pkname')) or leaf
                            rotational = parent.get('rota')
                            identity['block_device'] = dict(
                                name=parent.get('name'),
                                model=(parent.get('model') or '').strip(),
                                rotational=(
                                    bool(int(rotational))
                                    if rotational is not None else None),
                                type=parent.get('type'),
                                leaf_name=leaf.get('name'))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass
    return identity


def shutil_disk_usage(path):
    # Local wrapper keeps the imported surface small and testable.
    import shutil
    return shutil.disk_usage(path)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(('127.0.0.1', 0))
        return int(handle.getsockname()[1])


def _tail(path, lines=80):
    try:
        with open(path, errors='replace') as handle:
            content = handle.readlines()
    except OSError:
        return ''
    return ''.join(content[-lines:])


class GPUMonitor:
    """Sample physical GPU utilization/VRAM without a Python NVML dep."""

    def __init__(self, gpu_ids, output_path, interval=1.0):
        self.gpu_ids = gpu_ids
        self.output_path = Path(output_path)
        self.interval = float(interval)
        self.records = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval * 2.0))
        _atomic_json_dump(dict(records=self.records), self.output_path)

    def _run(self):
        command = [
            'nvidia-smi',
            f"--id={','.join(str(item) for item in self.gpu_ids)}",
            '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits',
        ]
        while not self._stop.is_set():
            sampled_at = time.time()
            try:
                result = subprocess.run(
                    command, check=False, capture_output=True, text=True,
                    timeout=max(5.0, self.interval))
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        fields = [field.strip() for field in line.split(',')]
                        if len(fields) != 5:
                            continue
                        self.records.append(dict(
                            timestamp=sampled_at,
                            gpu_index=int(fields[0]),
                            utilization_percent=float(fields[1]),
                            memory_used_mib=float(fields[2]),
                            memory_total_mib=float(fields[3]),
                            power_watts=float(fields[4])))
            except (OSError, subprocess.SubprocessError, ValueError):
                # Torch's rank-local allocator metrics remain authoritative if
                # nvidia-smi is temporarily unavailable.
                pass
            self._stop.wait(self.interval)

    def summary(self, started=None, finished=None):
        by_gpu = {}
        for record in self.records:
            if started is not None and record['timestamp'] < started:
                continue
            if finished is not None and record['timestamp'] > finished:
                continue
            by_gpu.setdefault(record['gpu_index'], []).append(record)
        summary = {}
        for gpu_index, records in sorted(by_gpu.items()):
            utilization = [item['utilization_percent'] for item in records]
            used = [item['memory_used_mib'] for item in records]
            power = [item['power_watts'] for item in records]
            summary[str(gpu_index)] = dict(
                samples=len(records),
                mean_utilization_percent=sum(utilization) / len(utilization),
                max_memory_used_mib=max(used),
                mean_power_watts=sum(power) / len(power))
        return summary


def _case_command(args, benchmark_checkpoint, case_dir, batch, workers, port):
    command = [
        sys.executable,
        '-m',
        'torch.distributed.launch',
        '--nproc_per_node=2',
        f'--master_port={port}',
        'tools/train.py',
        args.config,
        '--launcher',
        'pytorch',
        '--work-dir',
        str(case_dir),
        '--resume-from',
        str(benchmark_checkpoint),
        '--no-validate',
        '--seed',
        str(args.seed),
        '--deterministic',
        '--cfg-options',
        f'data.samples_per_gpu={batch}',
        f'data.workers_per_gpu={workers}',
        f'data.train.pipeline.0.cache_root={args.cache_root}',
        f'data.train.history_pipeline.0.cache_root={args.cache_root}',
        'data.train_dataloader.persistent_workers=True',
        'find_unused_parameters=False',
        f'runner.benchmark_warmup_iters={args.warmup_iters}',
        f'runner.benchmark_measure_iters={args.measure_iters}',
        f'runner.expected_samples_per_gpu={batch}',
        f'runner.expected_workers_per_gpu={workers}',
    ]
    return command


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # It still exists; inability to signal it must not be reported clean.
        return True


def _terminate_process_group(process, first_signal, grace_seconds=20):
    """Terminate every DDP rank and guarantee no child survives return."""
    if process is None:
        return
    process_group_id = process.pid  # start_new_session=True makes PID == PGID.
    try:
        os.killpg(process_group_id, first_signal)
    except ProcessLookupError:
        process.poll()
        return

    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            break
        process.poll()
        time.sleep(0.1)
    if not _process_group_exists(process_group_id):
        process.poll()
        return

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    kill_deadline = time.monotonic() + 10
    while _process_group_exists(process_group_id):
        if time.monotonic() >= kill_deadline:
            raise RuntimeError(
                f'Process group {process_group_id} survived SIGKILL')
        process.poll()
        time.sleep(0.1)
    process.poll()


class _ExternalTermination(Exception):
    def __init__(self, signum):
        super().__init__(f'received signal {signum}')
        self.signum = int(signum)


class _TerminationSignalGuard:
    """Turn catchable process-termination signals into cleanup exceptions."""

    SIGNALS = (signal.SIGTERM, signal.SIGHUP)

    def __enter__(self):
        self.previous = {}

        def handler(signum, _frame):
            raise _ExternalTermination(signum)

        for signum in self.SIGNALS:
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        return False


def _run_case(args, benchmark_checkpoint, run_root, batch, workers):
    case_name = f'b{batch}_w{workers}'
    case_dir = run_root / case_name
    case_dir.mkdir()
    port = _free_port()
    command = _case_command(
        args, benchmark_checkpoint, case_dir, batch, workers, port)
    command_path = case_dir / 'command.json'
    _atomic_json_dump(dict(argv=command), command_path)
    launcher_log = case_dir / 'launcher.log'
    monitor = GPUMonitor(args.gpus, case_dir / 'gpu_monitor.json')
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(item) for item in args.gpus)
    repository = str(Path.cwd().resolve())
    existing_pythonpath = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = (
        repository if not existing_pythonpath
        else repository + os.pathsep + existing_pythonpath)

    print(f'[{case_name}] work_dir={case_dir}', flush=True)
    print(f'[{case_name}] log={launcher_log}', flush=True)
    started = time.time()
    monitor.start()
    process = None
    timed_out = False
    return_code = None
    try:
        with _TerminationSignalGuard():
            try:
                with open(launcher_log, 'w') as log_handle:
                    process = subprocess.Popen(
                        command, cwd=repository, env=environment,
                        stdout=log_handle, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    try:
                        return_code = process.wait(
                            timeout=args.case_timeout_seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        _terminate_process_group(process, signal.SIGTERM)
                        return_code = 124
                    except KeyboardInterrupt:
                        _terminate_process_group(process, signal.SIGINT)
                        raise
            finally:
                # Covers Python exceptions, Ctrl-C, SIGTERM, and SIGHUP. Only
                # SIGKILL is inherently uncatchable by any userspace cleanup.
                _terminate_process_group(process, signal.SIGTERM)
                monitor.stop()
    except _ExternalTermination as error:
        raise SystemExit(128 + error.signum) from error
    wall_seconds = time.time() - started
    result = dict(
        case=case_name,
        batch_per_gpu=batch,
        workers_per_gpu=workers,
        return_code=int(return_code),
        timed_out=bool(timed_out),
        timeout_seconds=int(args.case_timeout_seconds),
        launcher_wall_seconds=wall_seconds,
        work_dir=str(case_dir),
        gpu_monitor=monitor.summary())
    summary_path = case_dir / 'throughput_summary.json'
    if return_code == 0 and summary_path.is_file():
        with open(summary_path) as handle:
            result['throughput'] = json.load(handle)
        throughput = result['throughput']
        loss_count = throughput.get('loss_anomalies', {}).get('count')
        threshold_status = throughput.get('thresholds', {}).get('status')
        valid = (
            loss_count == 0
            and throughput.get('overall_status') == threshold_status
            and throughput.get('warmup_iters') == args.warmup_iters
            and throughput.get('measured_iters') == args.measure_iters)
        if valid:
            window = throughput.get('measurement_window_unix', {})
            result['gpu_monitor_measurement_window'] = monitor.summary(
                window.get('started'), window.get('finished'))
            result['status'] = 'complete'
        else:
            result['status'] = 'failed'
            result['summary_validation_error'] = dict(
                loss_count=loss_count,
                overall_status=throughput.get('overall_status'),
                threshold_status=threshold_status,
                warmup_iters=throughput.get('warmup_iters'),
                measured_iters=throughput.get('measured_iters'))
    else:
        result['status'] = 'failed'
        result['log_tail'] = _tail(launcher_log)
    _atomic_json_dump(result, case_dir / 'case_result.json')
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument(
        '--checkpoint', required=True,
        help=(
            'separately prepared resume checkpoint; create it with '
            'prepare_resume_checkpoint.py --write'))
    parser.add_argument(
        '--source-checkpoint', default=DEFAULT_SOURCE_CHECKPOINT,
        help='untouched epoch_4 source, used only for preservation hashing')
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--gpus', type=_csv_ints, default=[0, 1])
    parser.add_argument('--batches', type=_csv_ints, default=[2])
    parser.add_argument('--workers', type=_csv_ints, default=[2, 4])
    parser.add_argument('--warmup-iters', type=int, default=50)
    parser.add_argument('--measure-iters', type=int, default=300)
    parser.add_argument('--case-timeout-seconds', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print case commands without reading the cache/checkpoint')
    args = parser.parse_args()
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2:
        parser.error('--gpus must name exactly two distinct physical GPUs')
    if any(item < 1 for item in args.batches):
        parser.error('--batches values must be at least 1')
    if len(args.batches) != 1:
        parser.error(
            'one prepared checkpoint is valid for exactly one batch size; '
            'run separate sweeps for separate batches')
    if any(item < 1 for item in args.workers):
        parser.error('--workers values must be at least 1')
    if args.warmup_iters != 50 or args.measure_iters != 300:
        parser.error(
            'This controlled sweep requires exactly --warmup-iters=50 and '
            '--measure-iters=300')
    if args.case_timeout_seconds < 60:
        parser.error('--case-timeout-seconds must be at least 60')
    return args


def main():
    args = parse_args()
    repository = Path.cwd().resolve()
    config = (repository / args.config).resolve()
    checkpoint = (repository / args.checkpoint).resolve()
    source_checkpoint = (repository / args.source_checkpoint).resolve()
    args.cache_root = str(Path(args.cache_root).resolve())
    if not config.is_file():
        raise FileNotFoundError(config)

    if args.dry_run:
        placeholder_root = Path(args.output_root) / '<timestamp>'
        for batch in args.batches:
            for workers in args.workers:
                case_dir = placeholder_root / f'b{batch}_w{workers}'
                command = _case_command(
                    args, Path('<prepared_resume_checkpoint.pth>'), case_dir,
                    batch, workers, 29500)
                print(json.dumps(command))
        return

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if checkpoint == source_checkpoint:
        raise ValueError(
            '--checkpoint must be a separately prepared file, not the '
            'original --source-checkpoint')
    cache = _validate_cache(args.cache_root)
    cache['storage'] = _storage_identity(args.cache_root)
    source_checkpoint_meta = _validate_checkpoint(
        source_checkpoint, require_resume_preparation=False)
    source_hash = _sha256(source_checkpoint)
    checkpoint_meta = _validate_checkpoint(
        checkpoint, require_resume_preparation=True,
        expected_samples_per_gpu=args.batches[0],
        expected_source_sha256=source_hash)
    benchmark_hash = _sha256(checkpoint)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = (repository / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    system = _system_preflight(output_root, args.gpus)
    run_root = output_root / f'{timestamp}_{os.getpid()}'
    run_root.mkdir()

    preflight = dict(
        config=str(config),
        source_checkpoint=str(source_checkpoint),
        source_checkpoint_sha256=source_hash,
        source_checkpoint_meta=source_checkpoint_meta,
        benchmark_checkpoint=str(checkpoint),
        benchmark_checkpoint_sha256=benchmark_hash,
        checkpoint_meta=checkpoint_meta,
        cache=cache,
        system=system,
        gpus=args.gpus,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters)
    _atomic_json_dump(preflight, run_root / 'preflight.json')

    results = []
    for batch in args.batches:
        for workers in args.workers:
            case_preflight = _system_preflight(output_root, args.gpus)
            result = _run_case(
                args, checkpoint, run_root, batch, workers)
            result['case_preflight'] = case_preflight
            _atomic_json_dump(
                result,
                Path(result['work_dir']) / 'case_result.json')
            results.append(result)
            # Reading a checkpoint should never mutate it.  Re-hash the source
            # after every case so that preservation is an enforced invariant.
            if _sha256(source_checkpoint) != source_hash:
                raise RuntimeError(
                    f'Original checkpoint changed during {result["case"]}')
            if _sha256(checkpoint) != benchmark_hash:
                raise RuntimeError(
                    f'Prepared benchmark checkpoint changed during '
                    f'{result["case"]}')
            if result['status'] != 'complete' and not args.continue_on_error:
                break
        if results[-1]['status'] != 'complete' and not args.continue_on_error:
            break

    final = dict(preflight=preflight, results=results)
    _atomic_json_dump(final, run_root / 'sweep_summary.json')
    print(f'Sweep summary: {run_root / "sweep_summary.json"}')
    for result in results:
        if result['status'] == 'complete':
            throughput = result['throughput']
            print(
                f"{result['case']}: "
                f"{throughput['aggregate_samples_per_second']:.4f} "
                f"samples/s ({throughput['thresholds']['status']}), "
                f"loss anomalies={throughput['loss_anomalies']['count']}")
        else:
            print(f"{result['case']}: FAILED rc={result['return_code']}")
    if any(result['status'] != 'complete' for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
