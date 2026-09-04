"""Hold-out planning L2 + T_infer, measured together in one pass over the
same replayed windows -- so a 1-frame vs 2-frame comparison needs one
checkpoint load and one sweep over the val set, not two separate scripts.

Merges eval_holdout_l2.py's windowing/L2 logic (isolated reset-per-window
replay of a real val scenario, scored like an actual test clip) with
measure_t_infer_fwd_only.py's timing methodology (clock only model.forward,
bracketed by torch.cuda.synchronize since CUDA kernels launch async -- see
that script's docstring for the full model-forward-only rationale from the
2026-08-25 organizer Q&A).

Use the FAST_EVAL config (not the cached one) so the timed forward pass
matches what a real test-time inference actually costs; L2 doesn't depend
on which pipeline loads the image, so this doesn't cost anything on the
accuracy side.

Each --frame-offsets config gets its own --warmup-windows (default 5)
untimed windows before recording starts, matching
measure_t_infer_fwd_only.py's --warmup-clips. This is not just about
excluding one-time cold-start cost (cudnn algorithm search, kernel JIT,
allocator growth) from the numbers -- it is required for a fair
multi-config comparison in the first place: configs run sequentially
against the same model instance, so without per-config warmup, whichever
config is listed first absorbs the entire cold-start cost while later
configs start already warm, making e.g. "1-frame vs 2-frame" comparisons
reflect list order as much as real cost.
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

HIS_FRAMES = 30
STREAM_STRIDE = 5
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='hold-out L2 + T_infer, one or more frame configs '
                    'in a single model load')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES,
                        help='minimum frame_idx to have a full lookback window')
    parser.add_argument(
        '--frame-offsets', nargs='+', default=None,
        help='One or more window configurations, e.g. --frame-offsets 0 '
             '"-5,0" for 1-frame vs 2-frame side by side. Each is '
             'comma-separated raw-frame offsets (0.1s units), must '
             'include 0. Default: a single config using the full 7-frame '
             'stream (-30,-25,...,0).')
    parser.add_argument('--fp16', action='store_true',
                         help='run inference in fp16 via mmcv.wrap_fp16_model')
    parser.add_argument('--warmup-windows', type=int, default=5,
                         help='untimed windows run (per config) before '
                              'L2/T_infer recording starts')
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def run_config(model, dataset, scenes, stream_offsets, args):
    """One frame-offset config's full sweep over the val set. Returns L2
    aggregates (matching eval_holdout_l2.py) plus a per-window T_infer list
    (matching measure_t_infer_fwd_only.py's clip_total_ms).

    The first args.warmup_windows valid windows are run for real (so the
    model, cudnn algorithm cache, and CUDA allocator are actually warmed
    up) but excluded from every returned aggregate -- L2 sums, command
    counts, and window_total_ms all start counting only after warmup
    completes. This config's own warmup is independent of any other
    config's, so each config in a multi-config run is timed under the
    same cold-vs-warm conditions rather than whichever ran first eating
    the one-time startup cost.
    """
    l2_sums = np.zeros(3)
    n_eval = 0
    n_skipped = 0
    n_cmd = len(COMMAND_VOCAB)
    cmd_all = np.zeros(n_cmd, dtype=int)
    cmd_valid = np.zeros(n_cmd, dtype=int)
    cmd_l2_sums = np.zeros((n_cmd, 3))
    window_total_ms = []
    n_warmed = 0

    for scene_token, sample_ids in mmcv.track_iter_progress(list(scenes.items())):
        frame_to_gi = {
            dataset.data_infos[gi]['frame_idx']: gi for gi in sample_ids}

        for gi in sample_ids:
            info = dataset.data_infos[gi]
            frame_idx = info['frame_idx']
            if frame_idx < args.min_frame or frame_idx % args.stride != 0:
                continue
            if not info.get('fut_valid_flag', False):
                continue

            gt_cmd_idx = int(np.asarray(info['gt_ego_fut_cmd']).argmax())

            window_frames = [frame_idx + off for off in stream_offsets]
            if any(f not in frame_to_gi for f in window_frames):
                # Only count skips against the real (post-warmup) tally --
                # a warmup-phase skip isn't evidence about the measured set.
                if n_warmed >= args.warmup_windows:
                    n_skipped += 1
                continue

            # Isolated window: reset before every one, like a real test clip.
            reset_stream(model.module)
            result = None
            collated = None
            total_ms = 0.0
            for f in window_frames:
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                torch.cuda.synchronize()
                fwd_start = time.perf_counter()
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)
                torch.cuda.synchronize()
                total_ms += (time.perf_counter() - fwd_start) * 1000.0

            if n_warmed < args.warmup_windows:
                n_warmed += 1
                continue  # ran for real (warms cudnn/allocator), not recorded

            window_total_ms.append(total_ms)
            cmd_all[gt_cmd_idx] += 1

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

    return dict(l2_sums=l2_sums, n_eval=n_eval, n_skipped=n_skipped,
                cmd_all=cmd_all, cmd_valid=cmd_valid, cmd_l2_sums=cmd_l2_sums,
                window_total_ms=window_total_ms)


def main():
    args = parse_args()
    specs = args.frame_offsets or ['-30,-25,-20,-15,-10,-5,0']
    configs = [(f'{len(parse_frame_offsets(s))}frame', parse_frame_offsets(s))
               for s in specs]

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

    results = {}
    for label, stream_offsets in configs:
        print(f'\n=== {label} (offsets {stream_offsets}) ===')
        results[label] = run_config(model, dataset, scenes, stream_offsets, args)

    # --- combined report -----------------------------------------------
    print()
    print(f'GPU: {torch.cuda.get_device_name(args.device)}')

    print(f'(each config warmed up on {args.warmup_windows} real, '
          f'unrecorded windows first)')
    print()
    header = (f'{"frames":<10}{"n_eval":>8}{"skipped":>9}{"L2@1s":>9}'
              f'{"L2@2s":>9}{"L2@3s":>9}{"L2_avg":>9}{"T_mean":>10}'
              f'{"T_median":>10}')
    print(header)
    print('-' * len(header))
    for label, _ in configs:
        r = results[label]
        if r['n_eval'] == 0:
            print(f'{label}: 0 windows recorded after warmup -- '
                  f'--warmup-windows ({args.warmup_windows}) likely '
                  f'exceeds the val set for this config; skipping')
            continue
        l2 = r['l2_sums'] / max(r['n_eval'], 1)
        arr = np.array(r['window_total_ms'])
        print(f'{label:<10}{r["n_eval"]:>8}{r["n_skipped"]:>9}{l2[0]:>9.4f}'
              f'{l2[1]:>9.4f}{l2[2]:>9.4f}{l2.mean():>9.4f}{arr.mean():>9.2f}m'
              f'{np.median(arr):>9.2f}m')

    print()
    print('T_infer penalty (Error Score multiplier, median SUM basis):')
    for label, _ in configs:
        arr = np.array(results[label]['window_total_ms'])
        t_infer = float(np.median(arr))
        penalty = max(0.0, t_infer - 100.0) / 200.0
        status = ('OK (no penalty)' if penalty == 0
                  else f'PENALIZED x{1.0 + penalty:.4f}')
        print(f'  {label:<10}{t_infer:>8.2f}ms  ->  {status}')

    print()
    print('L2 by command:')
    cmd_header = (f'{"":<10}{"cmd":<15}{"valid/all":>12}{"L2_avg":>10}')
    print(cmd_header)
    for label, _ in configs:
        r = results[label]
        for i, name in enumerate(COMMAND_VOCAB):
            n_v, n_a = r['cmd_valid'][i], r['cmd_all'][i]
            cmd_l2 = (r['cmd_l2_sums'][i] / max(n_v, 1)).mean()
            print(f'{label:<10}{name:<15}{f"{n_v}/{n_a}":>12}{cmd_l2:>10.4f}')


if __name__ == '__main__':
    main()
