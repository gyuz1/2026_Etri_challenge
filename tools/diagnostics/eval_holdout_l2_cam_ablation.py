"""Camera-masking ablation for hold-out planning L2.

Answers one question for the ETRI briefing's "actual source of performance"
compliance test (past ego state / goal point may only be *auxiliary* to a
video-based trajectory, never the substantial source of it): if the camera
input is masked out for every frame of the 7-frame/3s replay window --
leaving ego_lcf_feat and ego_target_point untouched -- how much does L2
degrade versus the unmasked model?

If masking the camera barely hurts L2, the model is not actually deriving
its trajectory from vision and the current config is at real risk under
that rule regardless of the network's architectural form. If it hurts a
lot, that's evidence video is the substantial source.

Identical to eval_holdout_l2.py except for --mask-camera / --max-samples;
kept as a separate script so the primary eval path is untouched.
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
STREAM_OFFSETS = list(range(-HIS_FRAMES, 1, STREAM_STRIDE))

COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN',
)


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def install_camera_mask(model, mode):
    """Monkeypatch extract_img_feat to mask the raw camera tensor.

    Intercepting here (rather than the DataContainer pre-collate) masks
    exactly what the backbone sees regardless of collation details, and
    leaves ego_lcf_feat / ego_target_point -- which never pass through
    this function -- completely untouched.
    """
    if mode == 'none':
        return
    original = model.extract_img_feat

    def patched(img, img_metas, len_queue=None):
        if mode == 'zero':
            img = torch.zeros_like(img)
        elif mode == 'noise':
            img = torch.randn_like(img)
        else:
            raise ValueError(f'unknown --mask-camera mode: {mode}')
        return original(img, img_metas, len_queue=len_queue)

    model.extract_img_feat = patched


def parse_args():
    parser = argparse.ArgumentParser(description='camera-masking L2 ablation')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--stride', type=int, default=5,
                        help='evaluation interval in frame_idx')
    parser.add_argument('--min-frame', type=int, default=HIS_FRAMES,
                        help='minimum frame_idx to have a full lookback window')
    parser.add_argument('--mask-camera', choices=['none', 'zero', 'noise'],
                        default='none',
                        help='none: unmasked baseline. zero/noise: mask '
                             'every frame of the replay window\'s camera '
                             'input while keeping ego_lcf_feat/target_point.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='stop after this many evaluated samples '
                             '(quick subset run)')
    parser.add_argument('--seed', type=int, default=0,
                        help='shuffles scene order before subsetting, so '
                             '--max-samples does not always hit the same '
                             'first scenes')
    return parser.parse_args()


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
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()
    install_camera_mask(model.module, args.mask_camera)

    scenes = {}
    for gi, info in enumerate(dataset.data_infos):
        scenes.setdefault(info['scene_token'], []).append(gi)

    scene_items = list(scenes.items())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(scene_items)

    l2_sums = np.zeros(3)
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

            window_frames = [frame_idx + off for off in STREAM_OFFSETS]
            if any(f not in frame_to_gi for f in window_frames):
                n_skipped_incomplete_window += 1
                continue

            reset_stream(model.module)
            result = None
            collated = None
            for f in window_frames:
                collated = collate([dataset[frame_to_gi[f]]], samples_per_gpu=1)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)

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
    print()
    print(f'--mask-camera {args.mask_camera}')
    print(f'evaluated samples: {n_eval}')
    print(f'skipped (incomplete 3s lookback window): {n_skipped_incomplete_window}')
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
