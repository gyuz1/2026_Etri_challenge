"""Empirically compare grad_norm-nan rate with aux_long_horizon on vs off,
same everything else, to verify the hypothesis that it's conflicting with
PRISM's posterior over the same ego_feats tensor before committing to a
multi-day restart.

Runs N real train steps (forward+backward+clip, no optimizer.step() needed
since we only care whether the clip-time grad norm is nan) for each
setting, using the actual dataset and the actual merged checkpoint.
"""
import argparse
import importlib

import torch
from mmcv import Config
from mmcv.parallel import collate, MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def run(cfg_path, ckpt_path, ann_file, n_iters, aux_long_horizon):
    cfg = Config.fromfile(cfg_path)
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.train.ann_file = ann_file
    cfg.model['pts_bbox_head']['aux_long_horizon'] = aux_long_horizon

    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    nan_count = 0
    finite_norms = []
    for i in range(n_iters):
        idx = i % len(dataset)
        batch = collate([dataset[idx]], samples_per_gpu=1)
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **batch)
        total = sum(v.sum() for v in losses.values() if torch.is_tensor(v))
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=35.0)
        norm = float(norm)
        if norm != norm:  # nan check without importing math
            nan_count += 1
        else:
            finite_norms.append(norm)
        optimizer.step()

    print(f'aux_long_horizon={aux_long_horizon}: '
          f'{nan_count}/{n_iters} nan ({100*nan_count/n_iters:.1f}%), '
          f'finite norm mean={sum(finite_norms)/max(len(finite_norms),1):.2f}')
    return nan_count, n_iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--n-iters', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    print('=== aux_long_horizon = True (current recipe) ===')
    run(args.config, args.checkpoint, args.ann_file, args.n_iters, True)

    torch.manual_seed(0)
    print('\n=== aux_long_horizon = False ===')
    run(args.config, args.checkpoint, args.ann_file, args.n_iters, False)


if __name__ == '__main__':
    main()
