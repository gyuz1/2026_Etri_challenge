import argparse
import importlib
import json
import os
from collections import OrderedDict

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


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_frame_offsets(spec):
    """Raw-frame offsets (e.g. "-30,-15,0") -> sequential test-clip indices
    (0=oldest=-30 ... 6=current=0), matching etri_test_converter.py's
    STREAM_FRAMES = range(-30, 1, 5) layout."""
    offsets = sorted(int(x) for x in spec.split(','))
    if offsets[-1] != 0:
        raise ValueError(
            f'--frame-offsets must include the current frame (0): {spec}')
    if offsets[0] < -HIS_FRAMES:
        raise ValueError(
            f'offset {offsets[0]} exceeds the provided '
            f'{HIS_FRAMES / 10:.1f}s history range: {spec}')
    indices = []
    for off in offsets:
        if off % STREAM_STRIDE != 0:
            raise ValueError(
                f'offset {off} is not a multiple of {STREAM_STRIDE} raw '
                f'frames ({STREAM_STRIDE / 10:.1f}s): {spec}')
        indices.append(off // STREAM_STRIDE + HIS_FRAMES // STREAM_STRIDE)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='ETRI challenge submission')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--frame-offsets', default=None,
        help='comma-separated raw-frame offsets (0.1s units) to replay per '
             'clip, e.g. "0" for the current frame only (no prev_bev '
             'history -- validated against condition1/condition2 hold-out '
             'ablation: ~0.8%% worse L2 but ~11x lower cumulative latency, '
             'enough to drop the Error Score T_infer penalty entirely). '
             'Must include 0. Default: every frame the test ann-file '
             'provides (the full 7-frame stream, -30,-25,...,0).')
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

    clips = OrderedDict()
    for gi, info in enumerate(dataset.data_infos):
        clips.setdefault(info['scene_token'], []).append(gi)

    if args.frame_offsets:
        keep_indices = set(parse_frame_offsets(args.frame_offsets))
        clips = OrderedDict(
            (token, [gi for i, gi in enumerate(gis) if i in keep_indices])
            for token, gis in clips.items())
        print(f'--frame-offsets {args.frame_offsets} -> '
              f'{len(keep_indices)} frame(s)/clip, indices {sorted(keep_indices)}')

    submission = {}
    for clip_token, sample_ids in mmcv.track_iter_progress(list(clips.items())):
        reset_stream(model.module)
        result = None
        collated = None
        for gi in sample_ids:
            collated = collate([dataset[gi]], samples_per_gpu=1)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **collated)
        ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
        cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
            -1, ego_fut_preds.shape[0])[0]
        traj = ego_fut_preds[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
        submission[clip_token] = traj.tolist()

    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    out_dict = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        if '__flops__' in existing:
            out_dict['__flops__'] = existing['__flops__']
    out_dict.update(submission)
    with open(args.out, 'w') as f:
        json.dump(out_dict, f)
    print(f'wrote {args.out} ({len(submission)} clips)')


if __name__ == '__main__':
    main()
