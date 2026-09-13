"""Does the planner's status slot already carry the ego state, or not?

The slot ego_fut_decoder consumes is produced by ego_status_est_net and was,
until ego_status_decode was added, trained only by a cosine against the
teacher's embedding. Cosine is scale-invariant, so nothing in that signal
pins down absolute speed -- and L2 is brutally sensitive to absolute speed
(tools/speed_error_to_l2.py: 0.25 m/s RMSE costs 0.5039 against the 0.2708
oracle).

This measures whether the slot ended up carrying the state anyway. It runs a
trained checkpoint over val samples, captures the slot, and fits a linear
probe from slot to the real ego_lcf columns, train/test split so the number
is a generalization error rather than a fit.

Reading the result:
  low speed RMSE   the slot already encodes speed; ego_status_decode is
                   mostly redundant and its weight should stay small
  high speed RMSE  the cosine left the physical state out, which is exactly
                   the gap ego_status_decode targets

Runs on CPU by default so it cannot OOM a training job sharing the GPU.

Usage:
    python tools/probe_status_slot.py <eval_config> <ckpt> [--n 100]
"""
import argparse

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401

VAL_ANN = ('data/etri/.causal_regen_split_301_75_10hz/'
           'vad_etri_infos_temporal_val_split.pkl')
NAMES = ['vx', 'vy', 'ax', 'ay', 'yaw_rate', 'speed']
COLS = [0, 1, 2, 3, 4, 7]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('ckpt')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.ann_file = VAL_ANN
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)

    mc = cfg.model.copy()
    mc.pop('feature_distill_teacher_cfg', None)
    mc.pop('feature_distill_teacher_ckpt', None)
    model = build_model(mc, test_cfg=cfg.get('test_cfg'))
    sd = torch.load(args.ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    missing = model.load_state_dict(sd, strict=False)
    print(f'체크포인트 로드: 누락 {len(missing.missing_keys)}, '
          f'예상 밖 {len(missing.unexpected_keys)}')
    net = model.pts_bbox_head.ego_status_est_net
    if net is None:
        raise SystemExit('이 config 에는 ego_status_est_net 이 없다')
    # The metric path assigns predictions to GT and is irrelevant here; it
    # also assumes a GPU-scattered batch shape. Both eval tools stub it out
    # the same way.
    # simple_test_pts computes detection/motion/planning metrics on the way
    # out. None of that is needed here, and assign_pred_to_gt_vip3d assumes
    # the GPU-scattered batch shape, so stub the whole metric chain.
    import numpy as _np
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model.assign_pred_to_gt_vip3d = lambda *a, **k: _np.zeros(0, dtype=int)
    model.compute_motion_metric_vip3d = lambda *a, **k: {}
    model.eval().to(args.device)
    # collate leaves DataContainer wrappers that the model cannot read;
    # MMDataParallel normally strips them in scatter. Calling scatter
    # directly does the same thing and works on CPU (target -1), whereas
    # MMDataParallel requires a real device id. CPU on purpose: this must not
    # be able to OOM a training job sharing the GPU.
    target = [-1] if args.device == 'cpu' else [0]

    def unwrap(batch):
        _, kw = scatter_kwargs((), batch, target)
        return kw[0]

    captured = []
    net.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().float().cpu()))

    # Stream two frames per window so prev_bev exists -- the descriptor the
    # estimator reads is a difference, so a single frame would make it
    # meaningless in a way real inference never is.
    frame_to_gi = {}
    for gi, inf in enumerate(dataset.data_infos):
        frame_to_gi[(inf['scene_token'], inf['frame_idx'])] = gi

    # Spread the sample across the whole val set, not the first scene. Taken
    # in order, consecutive frames share a speed and the probe's target has
    # almost no variance -- the number it produces then means nothing.
    candidates = [gi for gi, inf in enumerate(dataset.data_infos)
                  if (inf['scene_token'], inf['frame_idx'] - 5) in frame_to_gi]
    stride = max(1, len(candidates) // args.n)
    picked = candidates[::stride][:args.n]
    print(f'후보 {len(candidates)}개 중 {len(picked)}개를 {stride} 간격으로 추출')

    slots, targets = [], []
    done = 0
    for gi in picked:
        inf = dataset.data_infos[gi]
        prev = frame_to_gi[(inf['scene_token'], inf['frame_idx'] - 5)]
        model.prev_frame_info = {
            'prev_bev': None, 'prev_bev2': None, 'prev_bev_pristine': None,
            'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}
        captured.clear()
        with torch.no_grad():
            for j, idx in enumerate((prev, gi)):
                batch = unwrap(collate([dataset[idx]], samples_per_gpu=1))
                model(return_loss=False, rescale=True, **batch)
        if not captured:
            continue
        slots.append(captured[-1].reshape(-1).numpy())
        targets.append(np.asarray(inf['gt_ego_lcf_feat'],
                                  dtype=np.float64)[COLS])
        done += 1
        if done % 10 == 0:
            print(f'  {done}/{args.n}', flush=True)
        if done >= args.n:
            break

    X = np.stack(slots)
    Y = np.stack(targets)
    print(f'\n슬롯 {X.shape}, 타깃 {Y.shape}')

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_tr = len(X) // 2
    Xtr, Xte = X[:n_tr], X[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]
    Xtr_b = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])
    # Ridge, because the slot can be wider than the sample count here.
    lam = 1e-3 * len(Xtr)
    W = np.linalg.solve(Xtr_b.T @ Xtr_b + lam * np.eye(Xtr_b.shape[1]),
                        Xtr_b.T @ Ytr)
    pred = Xte_b @ W

    print(f'\n선형 probe (학습 {len(Xtr)} / 평가 {len(Xte)})')
    print(f"{'':>10} {'RMSE':>10} {'타깃 std':>10} {'설명력':>8}")
    for j, nm in enumerate(NAMES):
        rmse = float(np.sqrt(((pred[:, j] - Yte[:, j]) ** 2).mean()))
        sd_ = float(Yte[:, j].std())
        print(f'{nm:>10} {rmse:>10.4f} {sd_:>10.4f} '
              f'{1 - rmse ** 2 / max(sd_ ** 2, 1e-12):>8.3f}')

    sp = float(np.sqrt(((pred[:, 5] - Yte[:, 5]) ** 2).mean()))
    print(f'\n속도 RMSE = {sp:.4f} m/s')
    print('  speed_error_to_l2.py 대조: 0.10 -> L2 0.3311, '
          '0.25 -> 0.5039, 0.50 -> 0.8571')
    print('  (probe 는 선형이라 슬롯이 담은 정보의 하한이다 -- '
          '플래너는 비선형으로 더 뽑아낼 수 있다)')


if __name__ == '__main__':
    main()
