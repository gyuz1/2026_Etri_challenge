"""Speed read-out on the TRAINING forward path, for comparison with the
streaming test path (tools/diag_student_speed.py).

Builds the train dataset exactly as training does (geometry cache, queue of
history frames, history_sampling from the config) minus the random photometric
distortion, runs model.forward_train under no_grad, and reads
aux_bev_motion_head's output against the ego_lcf target on the same sample.

If the training path shows no bias while the test stream shows +5%, the two
paths feed the head differently, and the model learned on the one it is not
evaluated on.

Usage:
  python tools/diag_train_path_speed.py <train_cfg> <ckpt> [--n 200]
      [--train-mode] [--device 0]
"""
import argparse
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

TRAIN_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
             'vad_etri_infos_temporal_train_split.pkl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('ckpt')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--train-mode', action='store_true',
                   help='model.train() -- dropout on, as in real training')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--no-grid-mask', action='store_true',
                   help='turn GridMask off in train mode, keeping dropout on')
    p.add_argument('--only-train', default=None,
                   help='in eval mode, put only this kind of module in train mode: '
                        'dropout | bn | head (all of pts_bbox_head) | '
                        'backbone (img_backbone+neck)')
    args = p.parse_args()

    cfg = Config.fromfile(args.train_config)
    for m in (cfg.get('custom_imports') or {}).get('imports', []):
        importlib.import_module(m)
    importlib.import_module('tools.cache.etri_geometry_cache')
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    d = cfg.data.train
    d.ann_file = TRAIN_ANN
    strip = lambda pl: [q for q in pl
                        if q['type'] != 'PhotoMetricDistortionMultiViewImage']
    d.pipeline = strip(d.pipeline)
    if d.get('history_pipeline'):
        d.history_pipeline = strip(d.history_pipeline)
    dataset = build_dataset(d)

    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    model = build_model(mc, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model.cuda(args.device)
    if args.train_mode:
        model.train()
    else:
        model.eval()
    # prev_bev_dropout and modality dropout are gated on self.training; turn
    # the stochastic ones off so every sample sees its real history.
    if args.only_train:
        model.eval()
        model.use_grid_mask = False
        import torch.nn as nn
        kinds = {'dropout': (nn.Dropout,), 'bn': (nn.modules.batchnorm._BatchNorm,)}
        n_on = 0
        if args.only_train in kinds:
            for m in model.modules():
                if isinstance(m, kinds[args.only_train]):
                    m.train(); n_on += 1
        elif args.only_train.startswith('dropout:'):
            pat = args.only_train.split(':', 1)[1]
            for nm, m in model.named_modules():
                if isinstance(m, nn.Dropout) and pat in nm:
                    m.train(); n_on += 1
        elif args.only_train == 'head':
            model.pts_bbox_head.train(); n_on = 1
        elif args.only_train == 'backbone':
            model.img_backbone.train(); model.img_neck.train(); n_on = 2
        print(f'eval mode with only {args.only_train} in train mode: {n_on} modules')
    if args.no_grid_mask:
        model.use_grid_mask = False
    if hasattr(model, 'prev_bev_dropout'):
        model.prev_bev_dropout = 0.0
    head = model.pts_bbox_head
    idx = list(cfg.model.pts_bbox_head.get('aux_bev_motion_idx'))
    sc = idx.index(7)
    cap = []
    head.aux_bev_motion_head.register_forward_hook(
        lambda m, i, o: cap.append(o.detach().float().cpu()))

    n_all = len(dataset)
    picks = list(range(0, n_all, max(1, n_all // args.n)))[:args.n]
    preds, gts, ncalls = [], [], []
    for j, i in enumerate(picks):
        item = dataset[i]
        if item is None:
            continue
        batch = collate([item], samples_per_gpu=1)
        _, kw = scatter_kwargs((), batch, [args.device])
        kw = kw[0]
        cap.clear()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            try:
                model.forward_train(**kw)
            except KeyError:
                # eval mode skips train-only outputs the loss then asks for;
                # the head forward (and the hook) already ran.
                pass
        ncalls.append(len(cap))
        if not cap:
            continue
        preds.append(float(cap[-1].reshape(-1, len(idx))[0, sc]))
        lcf = kw['ego_lcf_feat']
        gts.append(float(lcf.reshape(-1)[7]))
        if (j + 1) % 50 == 0:
            print(f'  {j + 1}/{len(picks)}', flush=True)

    a, g = np.asarray(preds), np.asarray(gts)
    print(f'\ntraining path (mode={"train" if args.train_mode else "eval"}), '
          f'hook calls per sample {sorted(set(ncalls))} (last one used), n={len(a)}')
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 15), (15, 40)):
        m = (g >= lo) & (g < hi)
        if m.any():
            print(f'  {lo:>2}~{hi:<2} m/s  n={int(m.sum()):4d}  '
                  f'bias {float((a[m]-g[m]).mean()):+.3f}  '
                  f'ratio {float(a[m].mean()/max(g[m].mean(),1e-6)):.3f}')
    e = a - g
    print(f'  overall RMSE {np.sqrt((e**2).mean()):.3f}  bias {e.mean():+.3f}')


if __name__ == '__main__':
    main()
