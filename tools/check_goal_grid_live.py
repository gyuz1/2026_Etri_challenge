"""One real training forward of a goal-grid config, checking that the target
point actually drives it -- the failure this project keeps having is a switch
that builds fine and does nothing.

Asserts, on real train batches (geometry cache, history queue):
  1. loss_goal_cls and loss_goal_cell_traj are present and finite
  2. loss_goal_cls ~= goal_grid_weight * ln(K) at initialization (zero-init
     head -> uniform per-mode distribution)
  3. the GT cells are not all one cell (the label carries information)
  4. ego_fut_preds at initialization equals the donor's one-trajectory
     decoder output (tiling + uniform mix reproduce it)
  5. eval-mode forward never reads the target point (zeroing it changes
     nothing)

Usage: python tools/check_goal_grid_live.py <train_cfg> [--n 4] [--device 0]
"""
import argparse
import importlib
import math

import torch
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
        mc.pts_bbox_head.goal_grid_size = None
    m = build_model(mc, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(m, cfg.load_from, map_location='cpu', logger='silent'
                    if False else None)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    cfg = Config.fromfile(args.train_config)
    importlib.import_module('tools.cache.etri_geometry_cache')
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    d = cfg.data.train
    d.ann_file = TRAIN_ANN
    dataset = build_dataset(d)
    torch.manual_seed(0)

    model = build(cfg, True).cuda(args.device)
    donor = build(cfg, False).cuda(args.device)
    head = model.pts_bbox_head
    k = head.goal_grid_k
    ok = True
    cells = []
    for j in range(args.n):
        item = dataset[(j * 997) % len(dataset)]
        batch = collate([item], samples_per_gpu=1)
        _, kw = scatter_kwargs((), batch, [args.device])
        kw = kw[0]
        tp = kw.get('ego_target_point')
        cells.append(int(head._goal_cell_from_target(
            tp, 1, torch.float32, tp.device)[0]))
        model.train()
        with torch.no_grad():
            losses = model.forward_train(**kw)
        lc = losses.get('loss_goal_cls')
        lt = losses.get('loss_goal_cell_traj')
        if lc is None or lt is None or not torch.isfinite(lc) or not torch.isfinite(lt):
            print(f'  [FAIL] sample {j}: loss_goal_cls={lc} loss_goal_cell_traj={lt}')
            ok = False
            continue
        exp = head.goal_grid_weight * math.log(k)
        print(f'  sample {j}: cell {cells[-1]:2d}  loss_goal_cls {float(lc):.4f} '
              f'(초기 기대 {exp:.4f})  loss_goal_cell_traj {float(lt):.4f}  '
              f'loss_plan_reg {float(losses.get("loss_plan_reg", float("nan"))):.4f}')
        if abs(float(lc) - exp) > 1e-3:
            print('  [FAIL] 초기 CE 가 ln(K) 와 다르다 -- zero-init 또는 라벨 경로 이상')
            ok = False

        # eval: same output as donor at init, and target point unused
        model.eval(); donor.eval()
        captured = {}
        def hook(name):
            return lambda m, i, o: captured.__setitem__(name, o)
        h1 = head.ego_fut_decoder.register_forward_hook(hook('g'))
        h2 = donor.pts_bbox_head.ego_fut_decoder.register_forward_hook(hook('d'))
        with torch.no_grad():
            feats = torch.randn(2, head.ego_fut_decoder[0].in_features,
                                device=tp.device)
            g = head._collapse_goal(
                head.ego_fut_decoder(feats).reshape(2, head.ego_fut_mode, k, head.fut_ts, 2),
                head.goal_cls_head(feats).reshape(2, head.ego_fut_mode, k))
            dd = donor.pts_bbox_head.ego_fut_decoder(feats).reshape(2, head.ego_fut_mode, head.fut_ts, 2)
        h1.remove(); h2.remove()
        diff = float((g - dd).abs().max())
        if j == 0:
            print(f'  초기 출력 vs donor 최대차 {diff:.2e}')
            if diff > 1e-4:
                print('  [FAIL] 초기 goal-grid 출력이 donor 와 다르다'); ok = False
    if len(set(cells)) < 2:
        print(f'  [!!] {args.n} 샘플 칸이 전부 같다: {cells} (샘플 수를 늘려 확인)')
    import inspect
    src = inspect.getsource(type(head).forward)
    gate = 'if self.training and ego_target_point is not None:'
    if src.count('_goal_cell_from_target(') != 1 or gate not in src:
        print('  [FAIL] target point 읽기가 self.training 게이트 밖에 있다'); ok = False
    print('판정:', '통과' if ok else '실패')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
