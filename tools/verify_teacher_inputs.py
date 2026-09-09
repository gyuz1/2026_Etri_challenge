"""Verify, on real training samples, that the three privileged/ego input
channels the TP-shortcut teacher depends on actually arrive populated.

Motivated by a bug that cost a full 22h training run: ego_target_point was
accepted by VADLAW.forward_train but never forwarded to pts_bbox_head, so
VAD_head fell back to a zero goal for the entire run -- silently, because
the zero-fallback exists on purpose for history frames. This checks the
data side end to end (dataset -> pipeline -> collate), CPU-only, so it can
run while training occupies both GPUs.

Checks per sample:
  ego_target_point : present, correct shape after reshape(B, -1), non-zero
  ego_lcf_feat     : present, non-zero over the indices the teacher reads
  can_bus[7:16]    : accel / rotation-rate / velocity slice populated
                     (the img_metas-borne ego-status path that feeds the
                     BEV queries via can_bus_mlp)
"""

import argparse

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate

from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--ann-file', default=None)
    p.add_argument('--num-samples', type=int, default=8)
    p.add_argument('--lcf-idx', type=int, nargs='+',
                   default=[0, 1, 2, 3, 4, 5, 6, 7])
    return p.parse_args()


def unwrap(value):
    """Strip DataContainer / list nesting to a plain tensor."""
    while not torch.is_tensor(value):
        if hasattr(value, 'data'):
            value = value.data
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        else:
            return None
    return value


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin', False) and cfg.get('plugin_dir', None):
        import importlib
        import os.path as osp
        module = osp.dirname(cfg.plugin_dir).replace('/', '.')
        importlib.import_module(module)

    train_cfg = cfg.data.train
    if args.ann_file is not None:
        train_cfg.ann_file = args.ann_file
    dataset = build_dataset(train_cfg)
    print(f'dataset: {len(dataset)} samples')

    tp_rows, lcf_rows, canbus_rows = [], [], []
    for i in range(args.num_samples):
        sample = dataset[i * max(1, len(dataset) // (args.num_samples * 4))]
        batch = collate([sample], samples_per_gpu=1)

        tp = unwrap(batch.get('ego_target_point'))
        if tp is None:
            print('!!! ego_target_point ABSENT from the collated batch')
        else:
            tp_rows.append(tp.reshape(1, -1).float().numpy()[0])

        lcf = unwrap(batch.get('ego_lcf_feat'))
        if lcf is None:
            print('!!! ego_lcf_feat ABSENT from the collated batch')
        else:
            lcf_rows.append(lcf.reshape(-1).float().numpy())

        metas = batch['img_metas']
        while not isinstance(metas, dict):
            if hasattr(metas, 'data'):
                metas = metas.data
            elif isinstance(metas, (list, tuple)):
                metas = metas[0]
            else:
                break
        # LAW queue metas are {frame_index: meta}; take the current frame.
        if isinstance(metas, dict) and metas and isinstance(
                next(iter(metas.values())), dict):
            metas = metas[max(metas)]
        can_bus = np.asarray(metas['can_bus'], dtype=np.float64)
        canbus_rows.append(can_bus[7:16])

    def report(name, rows, extra=''):
        if not rows:
            print(f'{name}: NO DATA')
            return
        a = np.stack(rows)
        nz = int((np.abs(a).sum(axis=-1) > 1e-8).sum())
        print(f'{name}: shape={a.shape} nonzero_samples={nz}/{len(a)} '
              f'min={np.round(a.min(0), 4)} max={np.round(a.max(0), 4)}{extra}')

    print()
    report('ego_target_point', tp_rows)
    if tp_rows:
        a = np.stack(tp_rows)
        print(f'  -> width after reshape(B,-1) = {a.shape[-1]} '
              f'(target_point_encoder expects 2)')
        print(f'  -> varies across samples: '
              f'{bool(np.abs(a - a[0]).max() > 1e-6)}')
    if lcf_rows:
        a = np.stack(lcf_rows)
        sub = a[:, args.lcf_idx] if a.shape[-1] > max(args.lcf_idx) else a
        report('ego_lcf_feat[idx]', list(sub))
    report('can_bus[7:16]', canbus_rows,
           extra='  (accel[0:3], rot_rate[3:6], velocity[6:9])')


if __name__ == '__main__':
    main()
