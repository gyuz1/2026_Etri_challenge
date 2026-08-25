"""Checks whether the model is using a banned shortcut: producing the final
trajectory from ego_target_point + ego state alone, "independent of visual
recognition" -- the organizers' Q&A explicitly disallows this ("target
point와 자차 상태만으로 최종 궤적을 생성하거나 영상 인식과 무관하게 경로를
결정하는 방식은 허용되지 않습니다").

Sibling of eval_holdout_l2_prevbev_ablation.py (same isolated-window
methodology), with two independent corruption switches applied right before
the model call:

  --zero-images
      Overwrites the (already-normalized) image tensor with zeros for every
      replayed frame -- removes all visual content while ego_target_point/
      ego_lcf_feat/command stay real. If L2 barely degrades vs the
      uncorrupted baseline, the model is not actually using the images --
      i.e. exactly the banned shortcut.
  --zero-target-point
      Zeros ego_target_point instead, leaving images intact -- the
      complementary check (how much does the 5s goal point alone
      contribute, with vision doing all the work).

Run with neither flag for the uncorrupted baseline to compare both against.
"""
import argparse
import importlib

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30
STREAM_STRIDE = 5

COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN',
)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_frame_offsets(spec):
    offsets = sorted(int(x) for x in spec.split(','))
    if offsets[-1] != 0:
        raise ValueError(
            f'--frame-offsets must include the current frame (0): {spec}')
    if offsets[0] < -HIS_FRAMES:
        raise ValueError(
            f'offset {offsets[0]} exceeds the provided '
            f'{HIS_FRAMES / 10:.1f}s history range: {spec}')
    for off in offsets:
        if off % STREAM_STRIDE != 0:
            raise ValueError(
                f'offset {off} is not a multiple of {STREAM_STRIDE} raw '
                f'frames ({STREAM_STRIDE / 10:.1f}s): {spec}')
    return offsets


def corrupt(collated, zero_images, zero_target_point):
    """In-place: collated['img'] is [DataContainer], .data is [Tensor] of
    shape [1, Ncam, 3, H, W] (verified directly against this pipeline).
    ego_target_point is [DataContainer], .data is [Tensor] of shape [1, 2].
    """
    if zero_images:
        collated['img'][0].data[0].zero_()
    if zero_target_point:
        collated['ego_target_point'][0].data[0].zero_()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES)
    parser.add_argument('--frame-offsets', default='0',
                         help='default "0" -- single current frame, matches '
                              'the no-prev_bev direction already decided '
                              'from the earlier ablation')
    parser.add_argument('--zero-images', action='store_true')
    parser.add_argument('--zero-target-point', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    offsets = parse_frame_offsets(args.frame_offsets)
    print(f'--frame-offsets {args.frame_offsets} -> replaying {offsets}')
    print(f'zero_images={args.zero_images}  zero_target_point={args.zero_target_point}')

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
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    l2_sums = np.zeros(3)
    n_eval = 0
    n_skipped = 0
    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))

    n_done = 0
    for scene_token, sample_ids in mmcv.track_iter_progress(list(scenes.items())):
        frame_to_gi = {
            dataset.data_infos[gi]['frame_idx']: gi for gi in sample_ids}

        for gi in sample_ids:
            if args.max_samples and n_done >= args.max_samples:
                break
            info = dataset.data_infos[gi]
            frame_idx = info['frame_idx']
            if frame_idx < args.min_frame or frame_idx % args.stride != 0:
                continue
            if not info.get('fut_valid_flag', False):
                continue

            gt_cmd_idx = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
            cmd_all[gt_cmd_idx] += 1

            window_frames = [frame_idx + off for off in offsets]
            if any(f not in frame_to_gi for f in window_frames):
                n_skipped += 1
                continue

            reset_stream(model.module)
            result = None
            collated = None
            for f in window_frames:
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                corrupt(collated, args.zero_images, args.zero_target_point)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)
            n_done += 1

            ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
            cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
                -1, ego_fut_preds.shape[0])[0]
            pred = ego_fut_preds[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
            gt = np.array(info['gt_ego_fut_trajs'], dtype=np.float64).cumsum(0)

            dist = np.linalg.norm(pred - gt, axis=-1)
            sample_l2 = np.array([dist[:2].mean(), dist[:4].mean(), dist[:6].mean()])
            l2_sums += sample_l2
            n_eval += 1

            cmd_valid[gt_cmd_idx] += 1
            cmd_l2_sums[gt_cmd_idx] += sample_l2
        if args.max_samples and n_done >= args.max_samples:
            break

    l2 = l2_sums / max(n_eval, 1)
    print()
    print(f'evaluated samples: {n_eval}')
    print(f'skipped (incomplete lookback window): {n_skipped}')
    print('Official cumulative ADE:')
    print(f'  L2@1s : {l2[0]:.6f} m')
    print(f'  L2@2s : {l2[1]:.6f} m')
    print(f'  L2@3s : {l2[2]:.6f} m')
    print(f'  Final Planning L2 avg: {l2.mean():.6f} m')

    print()
    print('----------- ETRI Planning L2 by Command -----------')
    header = (f"{'Command':<15} {'valid/all':>12} {'L2@1s':>10} "
              f"{'L2@2s':>10} {'L2@3s':>10} {'L2_avg':>10}")
    print(header)
    print('-' * len(header))
    for i, name in enumerate(COMMAND_VOCAB):
        n_v, n_a = cmd_valid[i], cmd_all[i]
        cmd_l2 = cmd_l2_sums[i] / max(n_v, 1)
        print(f"{name:<15} {f'{n_v}/{n_a}':>12} {cmd_l2[0]:>10.6f} "
              f"{cmd_l2[1]:>10.6f} {cmd_l2[2]:>10.6f} {cmd_l2.mean():>10.6f}")
    print('-' * len(header))


if __name__ == '__main__':
    main()
