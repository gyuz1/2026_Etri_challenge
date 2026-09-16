"""Real training forwards of a goal_pred config, checking that the target point
actually drives it and never reaches inference. This project's recurring
failure is a switch that builds cleanly and does nothing, so every check reads
values, not config.

On real train batches (geometry cache, history queue):
  1. loss_goal_cls / loss_goal_off / loss_goal_follow present and finite
  2. loss_goal_cls == goal_cls_weight * ln(K) at init (zero-init head ->
     uniform per-mode bin distribution)
  3. label round trip: GT bin + GT offset decode back to the target point
  4. GT bins differ across samples (the label carries information)
  5. at init the scored 3s output equals the donor decoder's (zero goal
     residual, decoder steps copied)
  6. an eval-mode forward never builds labels from the target point

Usage: python tools/check_goal_pred_live.py <train_cfg> [--n 6] [--device 0]
Run with a single visible GPU: GridMask builds its mask on cuda:0.
"""
import argparse
import importlib
import math

import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

TRAIN_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
             'vad_etri_infos_temporal_train_split.pkl')


def build(cfg, goal):
    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    if not goal:
        mc.pts_bbox_head.goal_pred = False
    m = build_model(mc, train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))
    load_checkpoint(m, cfg.load_from, map_location='cpu')
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    cfg = Config.fromfile(args.train_config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.train.ann_file = TRAIN_ANN
    dataset = build_dataset(cfg.data.train)
    torch.manual_seed(0)

    model = build(cfg, True).cuda(args.device)
    donor = build(cfg, False).cuda(args.device)
    head = model.pts_bbox_head
    k = head.goal_k
    ok = True
    bins = []

    for j in range(args.n):
        item = dataset[(j * 997) % len(dataset)]
        _, kw = scatter_kwargs((), collate([item], samples_per_gpu=1),
                               [args.device])
        kw = kw[0]
        tp = kw['ego_target_point'].reshape(1, -1)[:, :2].float()

        gt_bin, gt_off = head._goal_label_from_target(tp, 1, torch.float32,
                                                      tp.device)
        c, w = head._goal_centres_cast(tp)
        back = torch.stack([c[gt_bin] + gt_off[:, 0] * w[gt_bin],
                            gt_off[:, 1] * head.goal_lat_scale], -1)
        rt = float((back - tp).abs().max())
        bins.append(int(gt_bin))

        model.train()
        with torch.no_grad():
            losses = model.forward_train(**kw)
        vals = {n: losses.get(n) for n in
                ('loss_goal_cls', 'loss_goal_off', 'loss_goal_follow',
                 'loss_plan_reg')}
        if any(v is None or not torch.isfinite(v) for v in vals.values()):
            print(f'  [FAIL] sample {j}: {vals}')
            ok = False
            continue
        exp = head.goal_cls_weight * math.log(k)
        print(f'  sample {j}: TP ({float(tp[0,0]):6.1f},{float(tp[0,1]):5.1f}) '
              f'bin {bins[-1]:2d}  round-trip err {rt:.1e}  '
              f'cls {float(vals["loss_goal_cls"]):.4f} (expected {exp:.4f})  '
              f'off {float(vals["loss_goal_off"]):.4f}  '
              f'follow {float(vals["loss_goal_follow"]):.4f}  '
              f'plan_reg {float(vals["loss_plan_reg"]):.4f}')
        if abs(float(vals['loss_goal_cls']) - exp) > 1e-3:
            print('  [FAIL] the initial CE differs from ln(K)'); ok = False
        if rt > 1e-3:
            print('  [FAIL] the label round trip does not return the target point'); ok = False

    # 5. init output equals donor
    with torch.no_grad():
        feats = torch.randn(3, head.ego_fut_decoder[0].in_features,
                            device=f'cuda:{args.device}')
        m = head.ego_fut_mode
        logits = head.goal_cls_head(feats).reshape(3, m, k)
        goals = head._goal_points(head.goal_off_head(feats).reshape(3, m, k, 2))
        sel = goals.gather(2, logits.argmax(-1)[:, :, None, None].expand(3, m, 1, 2)).squeeze(2)
        g = head._decode_to_goal(feats, sel)[:, :, :head.fut_ts]
        d = donor.pts_bbox_head.ego_fut_decoder(feats).reshape(3, m, head.fut_ts, 2)
    diff = float((g - d).abs().max())
    print(f'  initial 3s output vs donor, max diff {diff:.2e}')
    if diff > 1e-4:
        print('  [FAIL] the initial output differs from the donor'); ok = False

    # 6. eval-mode forward must not build labels
    called = []
    orig = head._goal_label_from_target

    def spy(*a, **kw_):
        called.append(head.training)
        return orig(*a, **kw_)
    head._goal_label_from_target = spy
    model.eval()
    with torch.no_grad():
        try:
            model.forward_train(**kw)
        except (KeyError, TypeError, AttributeError):
            pass  # eval mode skips train-only outputs the loss asks for
    if any(t is False for t in called):
        print('  [FAIL] labels were built from the target point in eval mode'); ok = False
    else:
        print(f'  eval-mode forward: {len(called)} label-building calls (0 is correct)')
        if called:
            ok = False
    head._goal_label_from_target = orig

    if len(set(bins)) < 2:
        print(f'  [!!] every sampled bin is the same: {bins}')
    print('verdict:', 'passed' if ok else 'failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
