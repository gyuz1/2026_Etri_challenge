"""Frame-count/spacing sweep: hold-out L2 vs per-clip cumulative T_infer.

T_infer is the PER-CLIP CUMULATIVE time of every model forward pass from
reset_stream through the final frame's output -- confirmed by the
competition organizers' official Q&A (2026-08-24), not a single frame's
latency (see measure_t_infer.py's docstring for the full quote). The same
Q&A also confirms frame count/sampling interval within the provided 3s
history window is free (image input for every used frame is still
mandatory) -- so unlike per-frame latency, cumulative T_infer can be cut
by replaying fewer history frames per clip before the final output, not
just by making each forward pass faster. This script answers the resulting
question directly: for a given frame subset, what L2 do we give up to buy
how much T_infer?

Not a copy of eval_holdout_l2_cam_ablation.py, but built the same way --
identical streaming/scoring logic, parametrized by --frame-offsets instead
of a hardcoded 7-frame window, plus cumulative per-clip timing.

Usage:
    python tools/diagnostics/sweep_frames_l2_tinfer.py \\
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \\
        work_dirs/stage2_etri_teammate_split/epoch_12.pth \\
        --ann-file data/etri/.causal_regen_teammate_split/vad_etri_infos_temporal_val_split.pkl \\
        --frame-offsets -30,-15,0 --fp16 --max-samples 75
"""
import argparse
import importlib
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30  # raw (0.1s) frames = 3.0s -- the provided past-info range
DEFAULT_OFFSETS = '-30,-25,-20,-15,-10,-5,0'  # current 7-frame baseline

COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN',
)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_offsets(spec):
    offsets = sorted(int(x) for x in spec.split(','))
    if offsets[-1] != 0:
        raise ValueError(
            f'--frame-offsets must include the current frame (0): {spec}')
    if offsets[0] < -HIS_FRAMES:
        raise ValueError(
            f'offset {offsets[0]} exceeds the provided '
            f'{HIS_FRAMES / 10:.1f}s history range: {spec}')
    if len(set(offsets)) != len(offsets):
        raise ValueError(f'duplicate offsets: {spec}')
    return offsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument(
        '--frame-offsets', default=DEFAULT_OFFSETS,
        help='comma-separated frame_idx offsets to replay per clip, '
             'relative to the scored ("current") frame = 0. Must include '
             '0 and stay within [-30, 0] (the provided 3.0s history). '
             f'Default (current baseline): {DEFAULT_OFFSETS}')
    parser.add_argument('--stride', type=int, default=5,
                         help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    offsets = parse_offsets(args.frame_offsets)

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

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    scene_items = list(scenes.items())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(scene_items)

    l2_sums = np.zeros(3)
    clip_times_ms = []
    n_eval = 0
    n_skipped_incomplete_window = 0

    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))

    for scene_token, sample_ids in mmcv.track_iter_progress(scene_items):
        if args.max_samples is not None and n_eval >= args.max_samples:
            break
        frame_to_gi = {
            dataset.data_infos[gi]['frame_idx']: gi for gi in sample_ids}

        for gi in sample_ids:
            if args.max_samples is not None and n_eval >= args.max_samples:
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
                n_skipped_incomplete_window += 1
                continue

            reset_stream(model.module)
            result = None
            collated = None
            clip_total_ms = 0.0
            for f in window_frames:
                start = time.perf_counter()
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                torch.cuda.synchronize()
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)
                torch.cuda.synchronize()
                clip_total_ms += (time.perf_counter() - start) * 1000.0
            clip_times_ms.append(clip_total_ms)

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

    l2 = l2_sums / max(n_eval, 1)
    clip_times = np.array(clip_times_ms) if clip_times_ms else np.zeros(1)
    t_infer = float(np.median(clip_times)) if clip_times_ms else float('nan')
    penalty = max(0.0, t_infer - 100.0) / 200.0

    print()
    print(f'--frame-offsets {args.frame_offsets} ({len(offsets)} frames/clip)')
    print(f'evaluated samples: {n_eval}')
    print(f'skipped (incomplete window for this offset spec): '
          f'{n_skipped_incomplete_window}')
    print('Official cumulative ADE:')
    print(f'  L2@1s : {l2[0]:.6f} m')
    print(f'  L2@2s : {l2[1]:.6f} m')
    print(f'  L2@3s : {l2[2]:.6f} m')
    print(f'  Final Planning L2 avg: {l2.mean():.6f} m')
    print()
    print(f'T_infer (median per-clip cumulative, {len(offsets)} forward '
          f'passes/clip): {t_infer:.2f}ms')
    print(f'Error Score multiplier if this were the real GPU: '
          f'x{1.0 + penalty:.4f} '
          f'({"no penalty" if penalty == 0 else "penalized"})')

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
