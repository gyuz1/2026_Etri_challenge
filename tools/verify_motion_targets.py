"""Check the two constants the motion losses are scaled by against the data.

Both are plain numbers written into VAD_head.py and a config. Neither is
derived at runtime, so if the data changed under them nothing would complain
-- the loss would simply optimize toward a wrong target at a wrong weight.

  FUT_TS_INTERVAL_S      divides the per-step GT delta to get a speed. If the
                         real interval is not 0.5s the future-speed head is
                         regressing a rescaled quantity.
  aux_bev_motion_norm    per-component std used to balance the L1. These were
                         computed once; if they no longer match the split's
                         actual std, one component silently dominates again
                         (before normalization vx and speed took 98.7% and
                         yaw_rate 0.1%).

Usage:
    python tools/verify_motion_targets.py [--ann <train pkl>]
"""
import argparse
import pickle

import numpy as np


LCF_NAMES = ['vx', 'vy', 'ax', 'ay', 'yaw_rate', 'length', 'width', 'speed',
             'unused']
# Order must match aux_bev_motion_idx=(0, 1, 2, 3, 4, 7).
CONFIG_IDX = (0, 1, 2, 3, 4, 7)
CONFIG_NORM = (5.7040, 0.1715, 0.4625, 0.3579, 0.0547, 5.7050)
FUT_TS_INTERVAL_S = 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann', default='data/etri/.causal_regen_split_301_75_10hz/'
                                    'vad_etri_infos_temporal_train_split.pkl')
    args = p.parse_args()

    with open(args.ann, 'rb') as fh:
        infos = pickle.load(fh)['infos']
    print(f'{len(infos)} samples from {args.ann}\n')

    lcf = np.stack([np.asarray(i['gt_ego_lcf_feat'], dtype=np.float64)
                    for i in infos])

    print('1. aux_bev_motion_norm vs 실제 std')
    bad = False
    for slot, idx in enumerate(CONFIG_IDX):
        actual = lcf[:, idx].std()
        cfgval = CONFIG_NORM[slot]
        ratio = actual / cfgval if cfgval else float('inf')
        flag = '' if 0.8 <= ratio <= 1.25 else '   <-- 불일치'
        if flag:
            bad = True
        print(f'   {LCF_NAMES[idx]:<9} config {cfgval:8.4f}   '
              f'실제 {actual:8.4f}   비율 {ratio:5.2f}{flag}')

    # What the normalized L1 share actually becomes. The point of the norm is
    # that no component takes the whole loss; report it rather than assume.
    share = np.array([lcf[:, i].std() / n
                      for i, n in zip(CONFIG_IDX, CONFIG_NORM)])
    share = share / share.sum() * 100
    print('   정규화 후 예상 L1 분담(%):',
          '  '.join(f'{LCF_NAMES[i]} {s:.1f}'
                    for i, s in zip(CONFIG_IDX, share)))

    print('\n2. FUT_TS_INTERVAL_S')
    # ego_fut_trajs are per-step deltas, so |delta_0| / interval is the speed
    # over the first step. Compare against the recorded speed at that sample.
    key = ('gt_ego_fut_trajs' if 'gt_ego_fut_trajs' in infos[0]
           else 'gt_ego_long_fut_trajs')
    deltas, speeds = [], []
    for i in infos:
        t = np.asarray(i[key], dtype=np.float64).reshape(-1, 2)
        if t.shape[0] == 0:
            continue
        deltas.append(np.linalg.norm(t[0]))
        speeds.append(float(np.asarray(i['gt_ego_lcf_feat'])[7]))
    deltas = np.asarray(deltas)
    speeds = np.asarray(speeds)
    moving = speeds > 1.0
    implied = (deltas[moving] / speeds[moving])
    print(f'   사용 키: {key}, 움직이는 샘플 {moving.sum()}개')
    print(f'   |delta_0| / speed 의 중앙값 = {np.median(implied):.4f} s')
    print(f'   코드의 FUT_TS_INTERVAL_S    = {FUT_TS_INTERVAL_S} s')
    est = np.median(implied)
    if abs(est - FUT_TS_INTERVAL_S) / FUT_TS_INTERVAL_S > 0.15:
        print(f'   [FAIL] 실제 간격은 약 {est:.3f}s -- 미래 속도 타깃이 '
              f'{FUT_TS_INTERVAL_S / est:.2f}배로 잘못 스케일된다')
        bad = True
    else:
        print('   [OK]  일치')

    print('\n판정:', '실패' if bad else '통과')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
