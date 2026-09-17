"""Per-loss gradient on the planner features (ego_feats) of a goal_pred model.

Why a goal_pred model's loss_plan_reg stays above the plain control: if the
goal losses push ego_feats harder than the planning loss, and in a conflicting
direction, the features drift toward goal classification. This measures, on
real training batches, each loss's gradient norm w.r.t. the features every
goal head and the decoder read, and its cosine with loss_plan_reg's gradient.

Usage: python tools/diag_goal_grad_conflict.py <train_cfg> <ckpt> [--n 8]
Run with one visible GPU. No optimizer step; weights are not modified.
"""
import argparse
import importlib

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

TRAIN_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
             'vad_etri_infos_temporal_train_split.pkl')
NAMES = ('loss_plan_reg', 'loss_goal_select', 'loss_goal_cls',
         'loss_goal_off', 'loss_goal_follow', 'loss_aux_long_horizon',
         'loss_ego_status_decode')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--grad-scale', type=float, default=None,
                   help='override goal_head_grad_scale to preview its effect')
    args = p.parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.train.ann_file = TRAIN_ANN
    dataset = build_dataset(cfg.data.train)
    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    if args.grad_scale is not None:
        mc.pts_bbox_head.goal_head_grad_scale = args.grad_scale
    model = build_model(mc, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # Only gradients on ego_feats are needed: freezing the image path keeps
    # its activations out of the graph, so this fits beside a running job.
    for name in ('img_backbone', 'img_neck'):
        if hasattr(model, name):
            getattr(model, name).requires_grad_(False)
    model = model.cuda(0).train()
    head = model.pts_bbox_head
    captured = []
    head._diag_capture = captured

    norms = {n: [] for n in NAMES}
    cos = {n: [] for n in NAMES}
    torch.manual_seed(0)
    for j in range(args.n):
        item = dataset[(j * 7919) % len(dataset)]
        _, kw = scatter_kwargs((), collate([item], samples_per_gpu=1), [0])
        captured.clear()
        losses = model.forward_train(**kw[0])
        feats = captured[-1]
        grads = {}
        for n in NAMES:
            v = losses.get(n)
            if v is None or not v.requires_grad:
                continue
            g, = torch.autograd.grad(v, feats, retain_graph=True,
                                     allow_unused=True)
            if g is not None:
                grads[n] = g.float().flatten()
        ref = grads.get('loss_plan_reg')
        for n, g in grads.items():
            norms[n].append(float(g.norm()))
            if ref is not None:
                cos[n].append(float(torch.nn.functional.cosine_similarity(
                    g, ref, dim=0)))
        del losses, grads, captured[:]
        torch.cuda.empty_cache()
    head._diag_capture = None

    base = sum(norms['loss_plan_reg']) / max(len(norms['loss_plan_reg']), 1)
    print(f'{"loss":<26}{"|grad| on ego_feats":>20}{"x plan_reg":>12}'
          f'{"cos with plan_reg":>20}')
    for n in NAMES:
        if not norms[n]:
            print(f'{n:<26}{"(no gradient path)":>20}')
            continue
        m = sum(norms[n]) / len(norms[n])
        c = sum(cos[n]) / len(cos[n])
        print(f'{n:<26}{m:>20.3e}{m / base:>12.2f}{c:>20.3f}')


if __name__ == '__main__':
    main()
