"""Measure the cost of a BEV-only history frame vs a full forward.

The 2-frame eval currently pays 2x the full forward cost (131.4ms,
over the 100ms T_infer penalty threshold) even though the history frame's
only job is to produce bev_embed for the next frame's temporal fusion --
detection/map/motion/planning decoders on that frame are thrown away.

VAD already has that fast path (VAD.py's obtain_history_bev:
extract_feat -> pts_bbox_head(..., only_bev=True)). This measures whether
using it for the history frame brings a 2-frame window under 100ms.
"""
import argparse
import importlib
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', required=True)
    p.add_argument('--n', type=int, default=30, help='timed repetitions')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


def timed(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return np.array(out)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fp16:
        wrap_fp16_model(model)
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()
    inner = model.module

    batch = collate([dataset[100]], samples_per_gpu=1)

    # ---- full forward (what both frames cost today) ------------------
    def reset():
        inner.prev_frame_info = {'prev_bev': None, 'scene_token': None,
                                 'prev_pos': 0, 'prev_angle': 0}

    def full_forward():
        reset()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **batch)

    full_ms = timed(full_forward, args.n, args.warmup)

    # ---- BEV-only forward (what a history frame actually needs) ------
    img = batch['img'][0].data[0].cuda(args.device)
    img_metas = batch['img_metas'][0].data[0]

    def bev_only():
        with torch.no_grad():
            feats = inner.extract_feat(img=img, img_metas=img_metas)
            inner.pts_bbox_head(feats, img_metas, None, only_bev=True)

    bev_ms = timed(bev_only, args.n, args.warmup)

    print()
    print(f'GPU: {torch.cuda.get_device_name(args.device)}')
    print(f'timed reps: {args.n} (after {args.warmup} warmup), fp16={args.fp16}')
    print()
    print(f'full forward   : mean {full_ms.mean():7.2f}ms  median {np.median(full_ms):7.2f}ms')
    print(f'BEV-only pass  : mean {bev_ms.mean():7.2f}ms  median {np.median(bev_ms):7.2f}ms')
    print(f'decoder-only   : {np.median(full_ms) - np.median(bev_ms):7.2f}ms '
          f'({100*(np.median(full_ms)-np.median(bev_ms))/np.median(full_ms):.1f}% of full)')
    print()
    today = 2 * np.median(full_ms)
    optimized = np.median(bev_ms) + np.median(full_ms)
    print('2-frame window T_infer:')
    print(f'  today (full + full)        : {today:7.2f}ms  '
          f'-> {"OK" if today <= 100 else f"PENALIZED x{1 + (today-100)/200:.4f}"}')
    print(f'  optimized (bev_only + full): {optimized:7.2f}ms  '
          f'-> {"OK" if optimized <= 100 else f"PENALIZED x{1 + (optimized-100)/200:.4f}"}')


if __name__ == '__main__':
    main()
