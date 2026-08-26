"""Runtime gradient/update audit used by the stage-1 optimization work.

The classes in this module are loaded only by the diagnostic config.  They do
not alter the production dataset, model, checkpoint, or training config.
"""

import json
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class DiagnosticSubsetDataset:
    """Expose a deterministic prefix of the configured training dataset."""

    def __init__(self, base_config, max_samples):
        # Import lazily so the normal project plugin has finished registering
        # its datasets before we recursively build the real dataset.
        from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

        cfg = Config.fromfile(base_config)
        self.dataset = custom_build_dataset(cfg.data.train)
        self.max_samples = min(int(max_samples), len(self.dataset))
        if hasattr(self.dataset, 'flag'):
            self.flag = np.asarray(self.dataset.flag[:self.max_samples]).copy()
        else:
            self.flag = np.zeros(self.max_samples, dtype=np.uint8)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        return self.dataset[index]

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)


@HOOKS.register_module()
class GradientUpdateAuditHook(Hook):
    """Record gradient reachability and exact parameter updates by rank."""

    def __init__(self, output_dir, write_interval=50, expected_iters=None,
                 inspect_iters=(1, 2, 10, 50, 100, 300)):
        self.output_dir = output_dir
        self.write_interval = int(write_interval)
        self.expected_iters = (
            None if expected_iters is None else int(expected_iters))
        self.inspect_iters = {int(i) for i in inspect_iters}

    @staticmethod
    def _unwrap(model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def before_run(self, runner):
        self.rank, self.world_size = get_dist_info()
        self.model = self._unwrap(runner.model)
        self.params = {
            name: param for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.initial = {
            name: param.detach().cpu().clone()
            for name, param in self.params.items()
        }
        self.stats = {}
        for name, param in self.params.items():
            self.stats[name] = {
                'shape': list(param.shape),
                'numel': int(param.numel()),
                'grad_present_iters': 0,
                'grad_none_iters': 0,
                'first_grad_iter': None,
                'last_grad_iter': None,
                'inspected_nonzero_iters': 0,
                'inspected_zero_iters': 0,
                'inspected_nonfinite_iters': 0,
            }
        self.observed_iters = 0
        self.nonfinite_loss_iters = []
        os.makedirs(self.output_dir, exist_ok=True)
        self._write(runner, include_updates=False)

    def before_train_epoch(self, runner):
        loader_iters = len(runner.data_loader)
        if (self.expected_iters is not None
                and loader_iters != self.expected_iters):
            raise RuntimeError(
                f'Diagnostic loader has {loader_iters} iterations; '
                f'expected {self.expected_iters}. Refusing to run a '
                'different audit length.')
        self.loader_iters = int(loader_iters)

    def after_train_iter(self, runner):
        iteration = int(runner.iter) + 1
        self.observed_iters += 1
        loss = runner.outputs.get('loss')
        if loss is not None and not bool(torch.isfinite(loss.detach()).all().item()):
            self.nonfinite_loss_iters.append(iteration)

        inspect_values = iteration in self.inspect_iters
        for name, param in self.params.items():
            stat = self.stats[name]
            grad = param.grad
            if grad is None:
                stat['grad_none_iters'] += 1
                continue
            stat['grad_present_iters'] += 1
            if stat['first_grad_iter'] is None:
                stat['first_grad_iter'] = iteration
            stat['last_grad_iter'] = iteration
            if inspect_values:
                grad_detached = grad.detach()
                if not bool(torch.isfinite(grad_detached).all().item()):
                    stat['inspected_nonfinite_iters'] += 1
                elif bool(torch.count_nonzero(grad_detached).item()):
                    stat['inspected_nonzero_iters'] += 1
                else:
                    stat['inspected_zero_iters'] += 1

        if self.write_interval > 0 and iteration % self.write_interval == 0:
            self._write(runner, include_updates=False)

    def after_run(self, runner):
        self._write(runner, include_updates=True)

    def _write(self, runner, include_updates):
        optimizer_state = runner.optimizer.state
        records = {}
        changed_tensors = 0
        changed_numel = 0
        for name, param in self.params.items():
            record = dict(self.stats[name])
            record['optimizer_state_present'] = bool(param in optimizer_state)
            if include_updates:
                current = param.detach().cpu()
                initial = self.initial[name]
                changed = not torch.equal(current, initial)
                record['parameter_changed'] = changed
                if changed:
                    delta = current.float() - initial.float()
                    record['max_abs_delta'] = float(delta.abs().max().item())
                    record['l2_delta'] = float(delta.norm().item())
                    changed_tensors += 1
                    changed_numel += int(param.numel())
                else:
                    record['max_abs_delta'] = 0.0
                    record['l2_delta'] = 0.0
            records[name] = record

        payload = {
            'rank': int(self.rank),
            'world_size': int(self.world_size),
            'observed_iters': int(self.observed_iters),
            'loader_iters': getattr(self, 'loader_iters', None),
            'runner_iter': int(runner.iter),
            'nonfinite_loss_iters': self.nonfinite_loss_iters,
            'trainable_tensor_count': len(self.params),
            'trainable_numel': int(sum(p.numel() for p in self.params.values())),
            'changed_tensor_count': changed_tensors if include_updates else None,
            'changed_numel': changed_numel if include_updates else None,
            'final': bool(include_updates),
            'parameters': records,
        }
        final_path = os.path.join(
            self.output_dir, f'rank{self.rank}_gradient_audit.json')
        temp_path = final_path + '.tmp'
        with open(temp_path, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temp_path, final_path)
