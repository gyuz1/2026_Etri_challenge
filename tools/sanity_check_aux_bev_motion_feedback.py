"""Forward+backward sanity check for aux_bev_motion_feedback.

Verifies two things with a real forward+backward pass on real data:

1. Compliance is intact: with ego_lcf_feat made a differentiable leaf,
   d(loss_plan_reg)/d(ego_lcf_feat) is still exactly 0. The feedback path
   (aux_bev_motion_head's prediction -> ego_feats -> ego_fut_decoder) is a
   function of bev_embed only, never of ego_lcf_feat/ego_lcf_target, so
   this must hold exactly as it did before the feedback was added.

2. The feedback is actually live: gradient from loss_plan_reg reaches
   aux_bev_motion_head's parameters (not just from loss_aux_bev_motion),
   proving the decoder's output actually depends on the head's own
   prediction, not merely training it via a dead-end loss.
"""
import argparse
import importlib

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--ann-file', default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    if args.ann_file:
        cfg.data.train.ann_file = args.ann_file

    head_cfg = cfg.model['pts_bbox_head']
    print('aux_bev_motion          :', head_cfg.get('aux_bev_motion'))
    print('aux_bev_motion_feedback :', head_cfg.get('aux_bev_motion_feedback'))
    assert head_cfg.get('aux_bev_motion_feedback') is True
    assert cfg.model['use_ego_lcf_status'] is False
    assert head_cfg['ego_lcf_feat_idx'] is None

    dataset = build_dataset(cfg.data.train)
    print(f'dataset built: {len(dataset)} samples')
    batch = collate([dataset[0]], samples_per_gpu=1)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.train()

    # Make ego_lcf_feat a differentiable leaf so we can check the gradient
    # reaching it directly, matching sanity_check_nolcf_aux.py's approach.
    lcf = batch['ego_lcf_feat'].data[0]
    lcf = lcf.clone().detach().cuda(0).requires_grad_(True)
    batch['ego_lcf_feat'].data[0] = lcf

    print('\n=== forward ===')
    losses = model(return_loss=True, **batch)
    for key in ('loss_plan_reg', 'loss_aux_bev_motion'):
        assert key in losses, f'{key} missing from loss dict'
        val = losses[key]
        val = val if torch.is_tensor(val) else torch.as_tensor(val)
        print(f'  {key:<24} {float(val.detach()):.6f}')

    print('\n=== backward: d(loss_plan_reg)/d(ego_lcf_feat) ===')
    plan_reg = losses['loss_plan_reg']
    plan_reg = plan_reg.sum() if torch.is_tensor(plan_reg) else plan_reg
    grad_lcf, = torch.autograd.grad(plan_reg, lcf, retain_graph=True,
                                     allow_unused=True)
    grad_lcf_norm = 0.0 if grad_lcf is None else grad_lcf.abs().sum().item()
    print(f'  |d(loss_plan_reg)/d(ego_lcf_feat)| = {grad_lcf_norm}')
    assert grad_lcf_norm == 0.0, (
        'COMPLIANCE VIOLATION: loss_plan_reg has a nonzero gradient into '
        'the raw ego_lcf_feat input -- the feedback path must depend on '
        'bev_embed only, never on ego_lcf_feat directly.')
    print('  OK: exactly zero, as required')

    print('\n=== backward: d(loss_plan_reg)/d(aux_bev_motion_head params) ===')
    aux_head = model.module.pts_bbox_head.aux_bev_motion_head
    model.zero_grad()
    plan_reg2 = losses['loss_plan_reg']
    plan_reg2 = plan_reg2.sum() if torch.is_tensor(plan_reg2) else plan_reg2
    plan_reg2.backward(retain_graph=True)
    aux_grad = sum(p.grad.abs().sum().item() for p in aux_head.parameters()
                   if p.grad is not None)
    print(f'  |d(loss_plan_reg)/d(aux_bev_motion_head)| = {aux_grad}')
    assert aux_grad > 0.0, (
        'aux_bev_motion_head gets no gradient from loss_plan_reg -- the '
        'feedback concat is not actually wired into the decoder path.')
    print('  OK: nonzero, feedback path is live')

    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
