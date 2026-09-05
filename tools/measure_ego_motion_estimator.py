"""How accurate is aux_bev_motion_head's ego-motion estimate at EVAL time,
in the cold-start condition the submission actually runs in?

The whole case for feeding that estimate to the planner
(aux_bev_motion_feedback, allowed by the 2026-09-04 organizer Q&A) rests
on the estimate being good. The only accuracy number we have so far is
the TRAINING loss, measured with prev_bev present on half the steps.
Evaluation replays each window from a stream reset, so the scored frame
of a 1-frame window has prev_bev=None -- a condition training only sees
through prev_bev_dropout.

This measures the estimate against ground-truth ego_lcf_feat over the
val set under exactly that condition, and reports it next to two
reference points:

  - predict-the-mean: what a model that learned nothing frame-specific
    would score. The estimator must beat this by a lot to be carrying
    real information.
  - relative error on speed: the number that matters for planning, since
    "how far forward in 3s" is essentially speed x time.
"""
import argparse
import importlib

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', required=True)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--min-frame', type=int, default=HIS_FRAMES)
    p.add_argument('--max-samples', type=int, default=600)
    p.add_argument(
        '--frame-offsets', default='0',
        help='comma-separated raw-frame offsets of the replayed window, '
             'ending at 0 (the scored frame) -- "0" is the cold-start '
             '1-frame condition, "-5,0" a 2-frame window, "-10,-5,0" a '
             '3-frame one. History frames are streamed exactly as '
             'forward_test does, including the can_bus prev_pos/prev_angle '
             'delta bookkeeping the encoder needs to align prev_bev.')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--device', type=int, default=0)
    return p.parse_args()


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
    model = MMDataParallel(model.cuda(args.device), device_ids=[args.device])
    model.eval()
    head = model.module.pts_bbox_head
    idx = list(head.aux_bev_motion_idx)
    print(f'aux_bev_motion_idx = {idx}  (ego_lcf_feat channels)')
    print(f'aux_bev_motion_temporal = {head.aux_bev_motion_temporal}')

    frame_lookup = {}
    for gi, info in enumerate(dataset.data_infos):
        frame_lookup[(info['scene_token'], info['frame_idx'])] = gi

    offsets = sorted(int(x) for x in args.frame_offsets.split(','))
    assert offsets[-1] == 0, f'--frame-offsets must end at 0: {offsets}'
    print(f'window offsets = {offsets}  ({len(offsets)} frame(s))')

    gis = [gi for gi, info in enumerate(dataset.data_infos)
           if info['frame_idx'] >= args.min_frame
           and info['frame_idx'] % args.stride == 0
           and info.get('fut_valid_flag', False)
           and all((info['scene_token'], info['frame_idx'] + off)
                   in frame_lookup for off in offsets)]
    if args.max_samples:
        gis = gis[::max(1, len(gis) // args.max_samples)][:args.max_samples]
    print(f'{len(gis)} frames sampled from the val set\n')

    preds, gts = [], []
    for gi in mmcv.track_iter_progress(gis):
        info = dataset.data_infos[gi]
        lcf = None

        with torch.no_grad():
            # Stream the window exactly as forward_test does: prev_bev and
            # the can_bus prev_pos/prev_angle deltas carry frame to frame,
            # and the first frame starts cold.
            prev_bev, prev_desc = None, None
            prev_pos, prev_angle = None, None
            for off in offsets:
                fgi = frame_lookup[(info['scene_token'],
                                    info['frame_idx'] + off)]
                fc = collate([dataset[fgi]], samples_per_gpu=1)
                fmetas = fc['img_metas'][0].data[0]
                tmp_pos = np.array(fmetas[0]['can_bus'][:3], copy=True)
                tmp_angle = float(fmetas[0]['can_bus'][-1])
                if prev_bev is None:
                    fmetas[0]['can_bus'][:3] = 0
                    fmetas[0]['can_bus'][-1] = 0
                else:
                    fmetas[0]['can_bus'][:3] -= prev_pos
                    fmetas[0]['can_bus'][-1] -= prev_angle

                # Take the previous descriptor BEFORE the encoder rotates
                # prev_bev in place (see VAD_head.bev_motion_descriptor).
                if prev_bev is not None and head.aux_bev_motion_temporal:
                    prev_desc = head.bev_motion_descriptor(prev_bev)

                feats = model.module.extract_feat(
                    img=fc['img'][0].data[0].cuda(args.device),
                    img_metas=fmetas)
                bev = head(feats, fmetas, prev_bev, only_bev=True)
                if bev.shape[1] == head.bev_h * head.bev_w:
                    bev = bev.permute(1, 0, 2)      # -> [N, B, D]

                prev_bev, prev_pos, prev_angle = bev, tmp_pos, tmp_angle
                lcf = np.asarray(
                    fc['ego_lcf_feat'][0].data[0]).reshape(-1)

            if head.aux_bev_motion_temporal:
                cur = head.bev_motion_descriptor(bev)
                delta = (cur - prev_desc if prev_desc is not None
                         else torch.zeros_like(cur))
                pooled = torch.cat([cur, delta], dim=-1)
            else:
                pooled = bev.mean(dim=0)
            pred = head.aux_bev_motion_head(pooled)

        preds.append(pred.float().cpu().numpy().reshape(-1))
        gts.append(lcf[idx])

    preds = np.stack(preds)
    gts = np.stack(gts)
    err = np.abs(preds - gts)
    mean_baseline = np.abs(gts - gts.mean(axis=0, keepdims=True))

    names = {0: 'vel_x', 1: 'vel_y', 2: 'accel_x', 3: 'accel_y',
             4: 'yaw_rate', 7: 'speed'}
    print()
    header = (f'{"channel":<12}{"MAE":>10}{"predict-mean MAE":>20}'
              f'{"GT std":>10}{"GT |mean|":>12}')
    print(header)
    print('-' * len(header))
    for j, c in enumerate(idx):
        print(f'{names.get(c, f"idx{c}"):<12}{err[:, j].mean():>10.4f}'
              f'{mean_baseline[:, j].mean():>20.4f}'
              f'{gts[:, j].std():>10.4f}{np.abs(gts[:, j]).mean():>12.4f}')

    if 7 in idx:
        j = idx.index(7)
        spd = np.abs(gts[:, j])
        keep = spd > 1.0          # relative error is meaningless near rest
        rel = err[keep, j] / spd[keep]
        print()
        print(f'speed relative error (over {keep.sum()} frames faster than '
              f'1 m/s): mean {100*rel.mean():.2f}%  median '
              f'{100*np.median(rel):.2f}%')
        print(f'3s extrapolation error implied by that: '
              f'{3.0*err[keep, j].mean():.3f} m')


if __name__ == '__main__':
    main()
