"""Check that an eval config builds the SAME network as its train config.

The failure this exists to catch: mmcv loads checkpoints with strict=False,
so an eval config that declares a module at the wrong width does not raise.
It logs a `size mismatch for ...` line and evaluates a randomly initialized
module. This project has lost a run to exactly that (prism_posterior_net),
and the A-student config was one edit away from losing another (the whole
ego_fut_decoder, via an unset ego_fut_dec_hidden_dim).

Comparing config dicts is not enough -- what matters is the built modules'
parameter shapes, since a train-only option can still change a module's
width. So this builds both models and diffs their state_dict shapes.

Usage:
    python tools/diff_eval_config.py <train_config> <eval_config> [ckpt]

With a checkpoint it also reports which of ITS keys fail to land, which is
the number that actually decides whether an eval is trustworthy.
"""
import argparse

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401  (registers VADLAW)


def build(path):
    cfg = mmcv.Config.fromfile(path)
    # The student configs load a frozen teacher at construction time; that
    # is a training-only object and its checkpoint may not exist yet.
    cfg.model.pop('feature_distill_teacher_cfg', None)
    cfg.model.pop('feature_distill_teacher_ckpt', None)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('train_config')
    p.add_argument('eval_config')
    p.add_argument('ckpt', nargs='?', default=None)
    args = p.parse_args()

    train = build(args.train_config)
    ev = build(args.eval_config)

    mismatch = sorted(k for k in train.keys() & ev.keys()
                      if train[k] != ev[k])
    train_only = sorted(train.keys() - ev.keys())
    eval_only = sorted(ev.keys() - train.keys())

    print('=== shape mismatch (치명적: 조용히 랜덤 초기화됨) ===')
    for k in mismatch:
        print(f'  {k}: train {train[k]} vs eval {ev[k]}')
    if not mismatch:
        print('  없음')

    # Train-only modules are expected (privileged heads, posteriors); an
    # eval-only module is not -- it has no checkpoint weights to load.
    print(f'\n=== train 에만 있는 모듈 ({len(train_only)}개, 보통 정상) ===')
    for k in train_only[:12]:
        print(f'  {k} {train[k]}')
    if len(train_only) > 12:
        print(f'  ... 외 {len(train_only) - 12}개')

    print(f'\n=== eval 에만 있는 모듈 ({len(eval_only)}개, 0이어야 정상) ===')
    for k in eval_only:
        print(f'  {k} {ev[k]}')

    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        sd = sd.get('state_dict', sd)
        bad = sorted(k for k in sd if k in ev and tuple(sd[k].shape) != ev[k])
        absent = sorted(k for k in sd if k not in ev)
        fresh = sorted(k for k in ev if k not in sd)
        print(f'\n=== 체크포인트 {args.ckpt} ===')
        print(f'  shape 불일치로 버려지는 키 : {len(bad)}')
        for k in bad:
            print(f'    {k}: ckpt {tuple(sd[k].shape)} vs eval {ev[k]}')
        print(f'  eval 모델에 없는 키        : {len(absent)}')
        print(f'  체크포인트에 없어 랜덤 초기화: {len(fresh)}')
        for k in fresh[:12]:
            print(f'    {k} {ev[k]}')
        if len(fresh) > 12:
            print(f'    ... 외 {len(fresh) - 12}개')

    ok = not mismatch and not eval_only
    print('\n판정:', '통과' if ok else '실패 — 위 항목을 고치기 전에 평가하지 말 것')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
