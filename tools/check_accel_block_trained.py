"""Prove from a trained checkpoint that the acceleration block got gradient.

check_accel_block_live.py reads the call path; this reads the outcome. If the
descriptor's third block is zero during training, the aux heads' input columns
4096:6144 multiply zeros, receive no gradient, and stay exactly at their
initial values. Comparing their statistics against the velocity block's is a
direct measurement of whether acceleration was learned or the config merely
claimed it.

What to expect when it works: the accel columns are SMALLER than the velocity
columns (second differences are noisier and smaller in magnitude) but the same
order, and clearly different from a fresh init. A ratio near zero, or columns
matching a freshly built model's init to several digits, means the block was
dead.

Usage:
    python tools/check_accel_block_trained.py <ckpt> --config <train config>
"""
import argparse

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


def blocks(w, frames):
    """Split an aux head's first-layer weight into its per-frame blocks."""
    per = w.shape[1] // frames
    return [w[:, i * per:(i + 1) * per] for i in range(frames)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ckpt')
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    frames = cfg.model.pts_bbox_head.get('aux_bev_motion_frames') or 2
    if frames < 3:
        print('3프레임 config 가 아님 -- 검사 불필요')
        return 0

    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)

    # EMAHook runs at priority HIGH and swaps the EMA into the model at
    # after_train_epoch, before CheckpointHook (NORMAL) writes the file. So
    # the ordinary slots hold the EMA and the ema_* buffers hold the raw
    # training params. Verified on this run: between epoch 1 and 2 the
    # ordinary slots move 2.27x LESS than the ema_* ones.
    #
    # That matters here. With momentum 0.0002 the EMA lags by roughly 5000
    # iterations, so early checkpoints' ordinary slots sit near their init no
    # matter how well training is going -- reading them under-reports
    # movement and makes a healthy run look inert. The raw params are where
    # learning shows first, so prefer them when present.
    raw = {k[len('ema_'):]: v for k, v in sd.items() if k.startswith('ema_')}
    if raw:
        lookup = {}
        for k in sd:
            if not k.startswith('ema_'):
                flat = k.replace('.', '_')
                if flat in raw:
                    lookup[k] = raw[flat]
        if lookup:
            print(f'EMAHook 감지: raw 학습 파라미터 {len(lookup)}개를 '
                  'ema_* 버퍼에서 읽는다 (일반 슬롯은 EMA 라 초기값 근처에 머문다)')
            sd = {**sd, **lookup}

    model_cfg = cfg.model.copy()
    model_cfg.pop('feature_distill_teacher_cfg', None)
    model_cfg.pop('feature_distill_teacher_ckpt', None)
    fresh = build_model(model_cfg, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg')).state_dict()

    names = [k for k in sd
             if k.endswith('.0.weight')
             and ('aux_bev_motion_head' in k
                  or 'aux_bev_future_motion_head' in k
                  or 'ego_status_est_net' in k)]
    if not names:
        print('  [FAIL] aux/estimator head 가 체크포인트에 없다')
        return 1

    failed = False
    for name in sorted(names):
        w = sd[name].float()
        f0 = fresh[name].float() if name in fresh else None
        bs = blocks(w, frames)
        labels = ['현재', '속도(1차차분)', '가속도(2차차분)'][:frames]
        print(f'\n{name}  shape {tuple(w.shape)}')
        for lab, b in zip(labels, bs):
            print(f'   {lab:<14} mean|w| {b.abs().mean():.6f}   '
                  f'std {b.std():.6f}')
        accel, vel = bs[2], bs[1]
        ratio = (accel.abs().mean() / vel.abs().mean()).item()
        print(f'   가속도/속도 크기비 = {ratio:.4f}')
        if f0 is not None:
            a0 = blocks(f0, frames)[2]
            drift = (accel - a0).abs().mean().item()
            init = a0.abs().mean().item()
            print(f'   초기값 대비 이동량 = {drift:.6f}  (초기 mean|w| {init:.6f})')
            # A block that never received gradient is bit-identical to its
            # init only if seeds match; they do not across processes, so
            # compare distributions instead. A dead block keeps the init's
            # std exactly, since nothing ever updated it.
            v0 = blocks(f0, frames)[1]
            vel_drift_std = abs(vel.std().item() - v0.std().item())
            acc_drift_std = abs(accel.std().item() - a0.std().item())
            print(f'   std 변화: 속도 {vel_drift_std:.6f} / '
                  f'가속도 {acc_drift_std:.6f}')
            if vel_drift_std > 0 and acc_drift_std < vel_drift_std * 0.02:
                print('   [FAIL] 가속도 블록만 초기 분포 그대로 '
                      '-- gradient 를 못 받았다')
                failed = True
                continue
        if ratio < 0.01:
            print('   [FAIL] 가속도 블록이 사실상 0')
            failed = True
        else:
            print('   [OK]  가속도 블록이 학습됐다')

    print('\n판정:', '실패' if failed else '통과')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
