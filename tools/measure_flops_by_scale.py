"""Break FLOPs down into img_backbone vs everything else, and project the
total at higher scale= values, without retraining anything.

Rationale: img_backbone (ResNet50 over 6 cameras) is fully convolutional and
its cost scales ~quadratically with the input's linear scale (area). BEV
encoder / detection / map / planning heads all operate on the FIXED
bev_h=bev_w=100 grid (VADLAW_etri_tiny.py:35-36) or a fixed number of
queries -- their cost does not depend on image resolution. So:

    total(s) ~= backbone(0.4) * (s/0.4)^2 + [total(0.4) - backbone(0.4)]

This script measures backbone(0.4) and total(0.4) directly (no retraining,
uses the already-trained stage1/stage2 checkpoint's architecture -- FLOPs
depend on shapes, not weights, so no checkpoint load is even needed), then
projects the total at each requested scale and reports which ones stay
under the FLOPs cutoff.
"""
import argparse
import importlib
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.utils.module_tracker as _mt
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.flop_counter import FlopCounterMode


class _NoHandle:
    def remove(self):
        pass


_mt.register_multi_grad_hook = lambda *a, **k: _NoHandle()


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--cutoff-gflops', type=float, default=7053.0)
    parser.add_argument('--project-scales', default='0.4,0.5,0.6,0.7,0.8,1.0')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda().eval()
    model.compute_planner_metric_stp3 = lambda *a, **k: {}

    reset_stream(model)
    with torch.no_grad():
        model(return_loss=False, rescale=True,
              **scatter(collate([dataset[0]]), [0])[0])
    data = scatter(collate([dataset[0]]), [0])[0]
    img_shape = tuple(data['img'][0].shape)
    print(f'measured img shape (at whatever scale this config uses): '
          f'{img_shape}')

    # Full-model FLOPs, deep enough to see individual named modules.
    counter = FlopCounterMode(display=False, depth=10)
    with torch.no_grad(), counter:
        model(return_loss=False, rescale=True, **data)
    per_module = counter.get_flop_counts()

    total = sum(per_module.get('Global', {}).values())

    # Sum every entry whose module path contains 'img_backbone'.
    backbone_flops = 0
    for module_path, ops in per_module.items():
        if 'img_backbone' in module_path:
            backbone_flops += sum(ops.values())
    # img_backbone's own subtree is double-counted against 'Global' at
    # different depths in some flop_counter versions -- Global is the
    # authoritative total, and backbone_flops here is read from the
    # top-level 'img_backbone' key specifically (children are nested under
    # it in the returned dict, not summed separately into Global again).
    backbone_top = sum(
        sum(ops.values()) for path, ops in per_module.items()
        if path.split('.')[-1] == 'img_backbone' or path == 'img_backbone')
    if backbone_top > 0:
        backbone_flops = backbone_top

    rest = total - backbone_flops
    print(f'\ntotal FLOPs        : {total/1e9:.1f} GFLOPs')
    print(f'img_backbone FLOPs  : {backbone_flops/1e9:.1f} GFLOPs '
          f'({100*backbone_flops/total:.1f}% of total)')
    print(f'everything else     : {rest/1e9:.1f} GFLOPs '
          f'({100*rest/total:.1f}% of total)')

    # Current scale, inferred from this config's data.test.pipeline. Two
    # different transforms carry it depending on which pipeline family the
    # config uses: RandomScaleImageMultiViewImage(scales=[s]) in the
    # raw/cached pipelines, FastUndistortCropScaleMultiViewImage(scale=s) in
    # the fast_eval (T_infer / submission) pipeline.
    def _find_scale(transforms):
        found = None
        for tr in transforms:
            if tr.get('type') == 'RandomScaleImageMultiViewImage':
                found = tr['scales'][0]
            elif tr.get('type') == 'FastUndistortCropScaleMultiViewImage':
                found = tr['scale']
            elif tr.get('type') == 'MultiScaleFlipAug3D':
                nested = _find_scale(tr.get('transforms', []))
                if nested is not None:
                    found = nested
        return found

    base_scale = _find_scale(cfg.data.test.get('pipeline', []))
    print(f'\nbase scale (this config): {base_scale}')
    if base_scale is None:
        print('Could not detect base scale from pipeline; assuming 0.4.')
        base_scale = 0.4

    print(f'\nProjection: total(s) = backbone({base_scale}) * '
          f'(s/{base_scale})^2 + rest')
    print(f'{"scale":>6}  {"projected GFLOPs":>18}  {"x cutoff":>10}  '
          f'{"under cutoff?":>14}')
    for s in [float(x) for x in args.project_scales.split(',')]:
        proj = backbone_flops * (s / base_scale) ** 2 + rest
        proj_g = proj / 1e9
        ratio = proj_g / args.cutoff_gflops
        ok = 'YES' if proj_g <= args.cutoff_gflops else 'NO'
        print(f'{s:>6.2f}  {proj_g:>18.1f}  {ratio:>9.3f}x  {ok:>14}')


if __name__ == '__main__':
    main()
