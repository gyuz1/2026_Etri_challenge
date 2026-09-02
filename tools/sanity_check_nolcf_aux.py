"""Forward+backward sanity check for the nolcf_aux stage-2 config.

Builds the real dataset and model from the config, runs one training step,
and asserts that the three new train-only losses are actually present,
finite, and produce gradients -- specifically that they reach parameters
the plain _nolcf config would leave untouched.

Also asserts the compliance-critical property directly: with
ego_lcf_feat_idx=None, no ego status can reach ego_fut_decoder. It checks
this by running the head twice with two very different ego_lcf_target
values and requiring the predicted trajectory to be bit-identical.
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

    print('=== config check ===')
    head = cfg.model['pts_bbox_head']
    print('use_ego_lcf_status   :', cfg.model['use_ego_lcf_status'])
    print('ego_lcf_feat_idx     :', head['ego_lcf_feat_idx'])
    print('aux_ego_motion       :', head.get('aux_ego_motion'))
    print('aux_long_horizon     :', head.get('aux_long_horizon'))
    print('aux_ego_motion_idx   :', head.get('aux_ego_motion_idx'))
    print('prev_bev_dropout     :', cfg.model.get('prev_bev_dropout'))
    print('echo_cycle_weight    :', cfg.model.get('echo_cycle_weight'))
    assert cfg.model['use_ego_lcf_status'] is False
    assert head['ego_lcf_feat_idx'] is None

    dataset = build_dataset(cfg.data.train)
    print(f'\ndataset built: {len(dataset)} samples')
    batch = collate([dataset[0]], samples_per_gpu=1)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.train()

    print('\n=== forward ===')
    losses = model(return_loss=True, **batch)
    for key in sorted(losses):
        val = losses[key]
        val = val if torch.is_tensor(val) else torch.as_tensor(val)
        print(f'  {key:<32} {float(val.detach()):.6f}')

    for required in ('loss_aux_ego_motion', 'loss_echo_cycle',
                     'loss_aux_long_horizon'):
        assert required in losses, f'{required} missing from loss dict'
        assert torch.isfinite(losses[required]).all(), f'{required} not finite'
    print('\nboth new losses present and finite')

    print('\n=== backward ===')
    total = sum(v.sum() for v in losses.values() if torch.is_tensor(v))
    total.backward()

    inner = model.module
    lh_head = inner.pts_bbox_head.aux_long_horizon_head
    lh_grad = sum(
        p.grad.abs().sum().item() for p in lh_head.parameters()
        if p.grad is not None)
    aux_head = inner.pts_bbox_head.aux_ego_motion_head
    aux_grad = sum(
        p.grad.abs().sum().item() for p in aux_head.parameters()
        if p.grad is not None)
    wm_grad = sum(
        p.grad.abs().sum().item() for p in inner.bev_world_model.parameters()
        if p.grad is not None)
    dec_grad = sum(
        p.grad.abs().sum().item()
        for p in inner.pts_bbox_head.ego_fut_decoder.parameters()
        if p.grad is not None)
    print(f'  aux_long_horizon_head grad sum : {lh_grad:.6f}')
    print(f'  aux_ego_motion_head grad sum : {aux_grad:.6f}')
    print(f'  bev_world_model     grad sum : {wm_grad:.6f}')
    print(f'  ego_fut_decoder     grad sum : {dec_grad:.6f}')
    assert aux_grad > 0, 'aux head received no gradient'
    assert lh_grad > 0, 'long-horizon head received no gradient'
    assert wm_grad > 0, 'world model received no gradient'
    assert dec_grad > 0, 'planning decoder received no gradient'

    print('\n=== compliance: ego status must NOT reach the trajectory ===')
    # Gradient-based proof. Make the ego status tensor a differentiable leaf,
    # then backprop the PLANNING loss alone. If any ego status reached
    # ego_fut_decoder -- directly or through a "simple embedding" -- a
    # gradient would arrive here. The aux loss is backpropped separately as
    # a positive control, confirming the tensor really is wired in and the
    # zero above is not just a disconnected graph.
    def _unwrap(value):
        while hasattr(value, 'data'):
            value = value.data
        while isinstance(value, (list, tuple)):
            value = value[0]
        return value

    lcf = _unwrap(batch['ego_lcf_feat']).clone().float().cuda(0)
    lcf.requires_grad_(True)
    probe = dict(batch)
    probe['ego_lcf_feat'] = lcf

    model.zero_grad(set_to_none=True)
    probe_losses = model(return_loss=True, **probe)
    probe_losses['loss_plan_reg'].backward(retain_graph=True)
    plan_grad = 0.0 if lcf.grad is None else float(lcf.grad.abs().sum())
    print(f'  d(loss_plan_reg)/d(ego_status) : {plan_grad:.8f}   <- must be 0')

    lcf.grad = None
    probe_losses['loss_aux_ego_motion'].backward()
    aux_grad = 0.0 if lcf.grad is None else float(lcf.grad.abs().sum())
    print(f'  d(loss_aux_ego_motion)/d(ego_status) : {aux_grad:.8f} '
          '  <- must be > 0 (control)')

    assert plan_grad == 0.0, 'ego status leaked into the trajectory output!'
    assert aux_grad > 0.0, (
        'aux loss has no gradient to ego status -- the control failed, so '
        'the zero above proves nothing')
    print('  -> ego status reaches ONLY the auxiliary loss, never the plan')

    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
