"""Measure whether the BEV motion descriptor can actually see ego motion.

The whole 3-frame design rests on one claim: the descriptor changes when the
BEV content shifts. A global mean pool would not -- it is exactly
shift-invariant, so cur - prev would be identically zero no matter how fast
the ego moved, and every motion loss would be regressing from noise. That is
the failure mode aux_bev_motion_temporal=False silently produces.

This measures the claim instead of trusting it: build the real head from the
config, shift a BEV by a known number of cells, and compare the descriptor
delta against the descriptor's own scale. It also reports what a global mean
would have given, and what the configured grid buys over a coarser one.

Usage:
    python tools/check_descriptor_sensitivity.py <train_config>
"""
import argparse

import mmcv
import torch
import torch.nn.functional as F
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--speed', type=float, default=10.57,
                   help='m/s, train-split mean')
    p.add_argument('--dt', type=float, default=0.5, help='s between frames')
    args = p.parse_args()

    cfg = mmcv.Config.fromfile(args.train_config)
    h = cfg.model.pts_bbox_head
    if not h.get('aux_bev_motion_temporal'):
        print('temporal descriptor 미사용 -- 검사 불필요')
        return 0

    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    head = build_model(mc, train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg')).pts_bbox_head
    head.eval()

    H, W, D = head.bev_h, head.bev_w, head.embed_dims
    pc = cfg.model.pts_bbox_head.get('pc_range') or cfg.get('point_cloud_range')
    span_x = float(pc[3]) - float(pc[0])
    span_y = float(pc[4]) - float(pc[1])
    m_per_cell_x = span_x / W
    grid = head.aux_bev_motion_grid
    print(f'BEV {H}x{W}, {D}ch, pc_range x {pc[0]}~{pc[3]} '
          f'-> 셀당 {m_per_cell_x:.3f} m')
    print(f'descriptor grid {grid}x{grid} -> 칸당 '
          f'{span_x/grid:.2f} m x {span_y/grid:.2f} m')

    move_m = args.speed * args.dt
    shift = max(1, int(round(move_m / m_per_cell_x)))
    print(f'\n평균 속도 {args.speed} m/s, {args.dt}s -> {move_m:.2f} m '
          f'= {shift} 셀 이동\n')

    torch.manual_seed(0)
    # A structured BEV, not white noise: real features are spatially
    # correlated, and a noise field would overstate how much any shift
    # changes a pooled statistic.
    base = torch.randn(1, D, H, W)
    base = F.avg_pool2d(base, 9, stride=1, padding=4)

    def desc(x):
        seq = x.reshape(1, D, H * W).permute(2, 0, 1)  # [N, B, D]
        with torch.no_grad():
            return head.bev_motion_descriptor(seq)

    d0 = desc(base)
    scale = d0.std().item()

    print(f'{"이동(셀)":>10} {"이동(m)":>9} {"|Δdesc|":>10} {"Δ/scale":>9}')
    for s in (0, 1, shift // 2, shift, shift * 2):
        shifted = torch.roll(base, shifts=int(s), dims=3)
        dd = (desc(shifted) - d0).abs().mean().item()
        print(f'{s:>10} {s*m_per_cell_x:>9.2f} {dd:>10.5f} '
              f'{dd/scale:>9.4f}')

    # The counterfactual: what the non-temporal path actually computes.
    def global_mean(x):
        seq = x.reshape(1, D, H * W).permute(2, 0, 1)
        return seq.mean(dim=0)

    g0 = global_mean(base)
    gshift = global_mean(torch.roll(base, shifts=int(shift), dims=3))
    gd = (gshift - g0).abs().mean().item()
    print(f'\n비교 -- global mean pool ({shift}셀 이동): |Δ| = {gd:.3e}')
    print('  (roll 은 순환 이동이라 전역 평균이 원리적으로 불변이다. '
          'aux_bev_motion_temporal=False 가 만드는 상태가 정확히 이것)')

    ok = True
    dd_shift = (desc(torch.roll(base, shifts=int(shift), dims=3))
                - d0).abs().mean().item()
    if dd_shift / scale < 0.01:
        print('\n  [FAIL] 실제 주행 변위에서 descriptor 가 거의 안 변한다')
        ok = False
    else:
        print(f'\n  [OK]  실제 변위에서 descriptor 가 scale 의 '
              f'{dd_shift/scale:.1%} 만큼 변한다')

    # Does the configured grid actually beat the coarser default?
    if grid > 4:
        saved = head.aux_bev_motion_grid
        head.aux_bev_motion_grid = 4
        c0 = desc(base)
        c1 = desc(torch.roll(base, shifts=int(shift), dims=3))
        coarse = (c1 - c0).abs().mean().item() / c0.std().item()
        head.aux_bev_motion_grid = saved
        print(f'  grid 4 였다면 {coarse:.1%} -> grid {grid} 는 '
              f'{dd_shift/scale/coarse:.2f}배 민감')

    print('\n판정:', '통과' if ok else '실패')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
