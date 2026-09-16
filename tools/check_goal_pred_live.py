"""Real training forwards of a goal_pred config, checking that the target point
actually drives training but never candidate generation. TP-based selection
among already-generated candidates is a separate permitted caller operation.
This project's recurring
failure is a switch that builds cleanly and does nothing, so every check reads
values, not config.

On real train batches (geometry cache, history queue):
  1. loss_goal_cls / loss_goal_off / loss_goal_follow present and finite
  2. loss_goal_cls == goal_cls_weight * ln(valid anchors of the GT command)
  3. label round trip: GT bin + GT offset decode back to the target point
  4. GT bins differ across samples (the label carries information)
  5. at init the scored 3s output equals the donor decoder's (zero goal
     residual, decoder steps copied)
  6. replay the actual current-frame head inputs in eval with two different
     TPs: trajectories/candidate points/masks stay identical, labels unread
  7. tiny decoder-only autograd probes reach classifier, offset and decoder
     weights with finite gradients; padded classifier/offset rows stay zero

Usage: python tools/check_goal_pred_live.py <train_cfg> [--n 6] [--device 0]
Run with a single visible GPU: GridMask builds its mask on cuda:0.
"""
import argparse
import copy
import importlib
import math

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

TRAIN_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
             'vad_etri_infos_temporal_train_split.pkl')


def build(cfg, goal):
    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    if not goal:
        mc.pts_bbox_head.goal_pred = False
    m = build_model(mc, train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))
    load_checkpoint(m, cfg.load_from, map_location='cpu')
    return m


def clone_inputs(value):
    """Do not reuse caller BEV/meta objects: the encoder can mutate them."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_inputs(item) for item in value)
    return copy.deepcopy(value)


def check_head_backward(head):
    """Decoder-only graph checks, with no optimizer step or .grad mutation.

    The zero-initialized goal embedding deliberately makes planning's gradient
    to the goal coordinates zero initially. This is not a learned-routing
    test: it checks CE/offset supervision and the decoder/embedding graph.
    """
    weight = head.goal_cls_head.weight
    m, k, b = head.ego_fut_mode, head.goal_k, head.ego_fut_mode
    feats = torch.randn(b, weight.shape[1], device=weight.device,
                        dtype=weight.dtype)
    _, mask = head._anchors(feats)
    mask = mask.bool()
    counts = mask.sum(-1).long()
    cmd = torch.arange(b, device=weight.device)
    labels = counts - 1
    logits = head.goal_cls_head(feats).reshape(b, m, k)
    logits = logits.masked_fill(~mask[None], torch.finfo(logits.dtype).min)
    off = head.goal_off_head(feats).reshape(b, m, k, 2)
    selected_off = off[cmd, cmd, labels]
    ce = F.cross_entropy(logits[cmd, cmd].float(), labels)
    offset_loss = F.smooth_l1_loss(
        selected_off.float(), selected_off.detach().float() + 0.25)
    points = head._goal_points(off)
    chosen = points.gather(2, labels[None, :, None, None].expand(b, m, 1, 2)).squeeze(2)
    traj = head._decode_to_goal(feats, chosen)
    decoder_loss = traj.float().square().mean()
    gc, = torch.autograd.grad(ce, head.goal_cls_head.weight, retain_graph=True)
    go, = torch.autograd.grad(offset_loss, head.goal_off_head.weight, retain_graph=True)
    gd, ge = torch.autograd.grad(
        decoder_loss, (head.ego_fut_decoder[-1].weight, head.goal_embed[-1].weight))
    grads = {'classifier CE': gc, 'offset': go, 'decoder': gd, 'goal embedding': ge}
    ok = True
    for name, grad in grads.items():
        finite = bool(torch.isfinite(grad).all())
        norm = float(grad.float().norm())
        nonzero_expected = name != 'classifier CE' or bool((counts > 1).any())
        passed = finite and (norm > 0 or not nonzero_expected)
        print(f'  backward {name}: finite={finite}, norm={norm:.3e}')
        if not passed:
            print(f'  [FAIL] decoder-only {name} gradient check'); ok = False
    padded_cls = gc.reshape(m, k, -1)[~mask]
    padded_off = go.reshape(m, k, 2, -1)[~mask]
    if ((padded_cls.numel() and bool((padded_cls != 0).any()))
            or (padded_off.numel() and bool((padded_off != 0).any()))):
        print('  [FAIL] padded anchor rows received CE/offset gradients'); ok = False
    print('  Scope: decoder-only backward; no optimizer step or full BEV backward. '
          'Zero planning-to-goal gradient at zero-init is expected.')
    return ok


def check_eval_invariance(head, captured):
    """Replay a real head call, not forward_train's training loss wrapper."""
    if not captured:
        print('  [FAIL] no current-frame head inputs captured for eval replay')
        return False
    args, kwargs = captured
    calls = []
    original_label = head._goal_label_from_target
    original_expose = head.goal_expose_candidates

    def spy(*a, **kw):
        calls.append(True)
        return original_label(*a, **kw)

    head._goal_label_from_target = spy
    head.goal_expose_candidates = True
    try:
        with torch.no_grad():
            first = head(*clone_inputs(args), **clone_inputs(kwargs))
            changed = clone_inputs(kwargs)
            changed['ego_target_point'] = torch.full_like(
                changed['ego_target_point'], 1234.5)
            second = head(*clone_inputs(args), **changed)
    finally:
        head._goal_label_from_target = original_label
        head.goal_expose_candidates = original_expose
    ok = not calls
    if calls:
        print('  [FAIL] eval replay built TP labels')
    for key in ('ego_fut_preds', 'goal_pred', 'goal_cand_trajs',
                'goal_cand_points', 'goal_cand_mask'):
        x, y = first.get(key), second.get(key)
        if (not torch.is_tensor(x) or not torch.is_tensor(y)
                or not bool(torch.isfinite(x).all()) or not torch.equal(x, y)):
            print(f'  [FAIL] eval TP perturbation: {key} missing, non-finite or changed')
            ok = False
    if ok:
        print('  eval replay: TP perturbation changes no generated trajectory, '
              'candidate point or validity mask; label calls=0')
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    if args.n < 1:
        p.error('--n must be positive')
    cfg = Config.fromfile(args.train_config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.train.ann_file = TRAIN_ANN
    dataset = build_dataset(cfg.data.train)
    torch.manual_seed(0)

    model = build(cfg, True).cuda(args.device)
    donor = build(cfg, False).cuda(args.device)
    head = model.pts_bbox_head
    k = head.goal_k
    ok = True
    bins = []
    captured = []

    for j in range(args.n):
        item = dataset[(j * 997) % len(dataset)]
        _, kw = scatter_kwargs((), collate([item], samples_per_gpu=1),
                               [args.device])
        kw = kw[0]
        tp = kw['ego_target_point'].reshape(1, -1)[:, :2].float()
        cmd = kw['ego_fut_cmd'].reshape(1, -1).argmax(-1)
        gt_bin, gt_off = head._goal_label_from_target(tp, 1, torch.float32,
                                                     tp.device, cmd)
        table, mask = head._anchors(tp)
        cell = table[cmd, gt_bin]
        back = cell[:, 0::2] + gt_off * cell[:, 1::2]
        rt = float((back - tp).abs().max())
        valid_label = bool(mask[cmd, gt_bin].bool().all())
        bins.append((int(cmd[0]), int(gt_bin[0])))
        if not valid_label or not bool(torch.isfinite(gt_off).all()):
            print(f'  [FAIL] sample {j}: invalid/padded label or non-finite offset')
            ok = False

        model.train()
        original_forward = head.forward

        def capture(*a, **kw_):
            if (j == args.n - 1 and kw_.get('ego_target_point') is not None
                    and not kw_.get('only_bev', False)):
                captured[:] = [clone_inputs(a), clone_inputs(kw_)]
            return original_forward(*a, **kw_)

        head.forward = capture
        try:
            with torch.no_grad():
                losses = model.forward_train(**kw)
        finally:
            head.forward = original_forward
        names = ['loss_goal_cls', 'loss_goal_off', 'loss_plan_reg']
        if head.goal_follow_weight > 0:
            names.append('loss_goal_follow')
        if head.goal_select_weight > 0:
            names.append('loss_goal_select')
        vals = {n: losses.get(n) for n in names}
        if any(v is None or not torch.isfinite(v) for v in vals.values()):
            print(f'  [FAIL] sample {j}: {vals}')
            ok = False
            continue
        count = int(mask[cmd[0]].bool().sum())
        exp = head.goal_cls_weight * math.log(count)
        print(f'  sample {j}: TP ({float(tp[0,0]):6.1f},{float(tp[0,1]):5.1f}) '
              f'command {bins[-1][0]} bin {bins[-1][1]:2d}/{count}  round-trip err {rt:.1e}  '
              f'cls {float(vals["loss_goal_cls"]):.4f} (expected {exp:.4f})  '
              f'off {float(vals["loss_goal_off"]):.4f}  '
              f'follow {float(vals.get("loss_goal_follow", 0.0)):.4f}  '
              f'plan_reg {float(vals["loss_plan_reg"]):.4f}')
        if abs(float(vals['loss_goal_cls']) - exp) > 1e-3:
            print('  [FAIL] the initial CE differs from ln(valid command anchors)'); ok = False
        if not math.isfinite(rt) or rt > 1e-3:
            print('  [FAIL] the label round trip does not return the target point'); ok = False

    # 5. init output equals donor (both decode in deterministic eval mode)
    model.eval(); donor.eval()
    with torch.no_grad():
        feats = torch.randn(3, head.ego_fut_decoder[0].in_features,
                            device=f'cuda:{args.device}')
        m = head.ego_fut_mode
        logits = head.goal_cls_head(feats).reshape(3, m, k)
        _, mask = head._anchors(feats)
        logits = logits.masked_fill(~mask.bool()[None], torch.finfo(logits.dtype).min)
        goals = head._goal_points(head.goal_off_head(feats).reshape(3, m, k, 2))
        sel = goals.gather(2, logits.argmax(-1)[:, :, None, None].expand(3, m, 1, 2)).squeeze(2)
        g = head._decode_to_goal(feats, sel)[:, :, :head.fut_ts]
        d = donor.pts_bbox_head.ego_fut_decoder(feats).reshape(3, m, head.fut_ts, 2)
    diff = float((g - d).abs().max())
    print(f'  initial 3s output vs donor, max diff {diff:.2e}')
    if not math.isfinite(diff) or diff > 1e-4:
        print('  [FAIL] the initial output differs from the donor'); ok = False

    ok = check_eval_invariance(head, captured) and ok
    ok = check_head_backward(head) and ok

    if len(set(bins)) < 2:
        print(f'  [!!] every sampled (command, bin) is the same: {bins}')
    print('verdict:', 'passed' if ok else 'failed')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
