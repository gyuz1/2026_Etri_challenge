"""Build the per-command goal anchors a goal_pred config uses, from TRAIN only.

An anchor is one candidate 5s goal: (x_centre, x_width, y_centre, y_width) in
metres, in the ego frame. At inference the head predicts, per anchor, an offset
from its centre in units of its width, so the anchors only have to bracket the
distribution -- they do not have to be exact.

Rules, all measured on the train split:
  * forward bins are uniform inside a per-command range fitted to that command
  * a forward bin is split laterally only where the data actually splits:
    lateral std < 1m -> 1 bin, < 5m -> 2, < 10m -> 3, else 4 (capped per
    command). A 2-way split uses the midpoint of a 1-D 2-means, not a quantile,
    so a bin is only cut where there really are two groups.
  * left/right commands share one fitted table, mirrored, which doubles the
    data behind every turn and lane-change anchor
  * STOP is fitted on frames the inference rule fires on (speed < 0.1 m/s), not
    on STOP-labelled frames: 11% of those have a target point beyond 7m because
    the car is stopped now and moves within the next 5s.

Usage:
    python tools/make_goal_anchors.py [--out projects/configs/VAD/_goal_anchors.py]
                                      [--lean]
"""
import argparse
import pickle

import numpy as np

TRAIN = ('data/etri/.causal_regen_split_301_75_10hz/'
         'vad_etri_infos_temporal_train_split.pkl')
CMD = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT',
       'TURN_RIGHT', 'U_TURN', 'STOP']

# group -> (command ids, mirrored, forward edges, max lateral bins)
FULL = [
    ('LANE_KEEP', [0], False, np.linspace(-1, 118, 25), 2),
    ('LANE_CHANGE', [1, 2], True, np.linspace(5, 116, 17), 2),
    ('TURN', [3, 4], True, np.linspace(-1, 46, 13), 4),
    ('U_TURN', [5], False, np.linspace(-1, 23, 6), 2),
    ('STOP', [6], False, np.array([-1, 0.5, 2, 5, 10, 17, 25.]), 1),
]
# Leaner variant: forward resolution only, no lateral splits.
LEAN = [
    ('LANE_KEEP', [0], False, np.linspace(-1, 118, 17), 1),
    ('LANE_CHANGE', [1, 2], True, np.linspace(5, 116, 11), 1),
    ('TURN', [3, 4], True, np.linspace(-1, 46, 9), 1),
    ('U_TURN', [5], False, np.linspace(-1, 23, 5), 1),
    ('STOP', [6], False, np.array([-1, 0.5, 2, 5, 10, 17, 25.]), 1),
]


def two_means(y, iters=50):
    c = np.array([np.quantile(y, 0.2), np.quantile(y, 0.8)])
    for _ in range(iters):
        a = np.abs(y[:, None] - c[None, :]).argmin(1)
        for j in (0, 1):
            if (a == j).any():
                c[j] = y[a == j].mean()
    return float(np.sort(c).mean())


def lateral_split(ys, max_bins):
    """Edges for one forward bin, or [] when the data does not split."""
    if max_bins == 1 or len(ys) < 20:
        return []
    sd = float(ys.std())
    k = 1 if sd < 1.0 else (2 if sd < 5.0 else (3 if sd < 10.0 else 4))
    k = min(k, max_bins)
    if k == 1:
        return []
    if k == 2:
        return [two_means(ys)]
    return [float(np.quantile(ys, (j + 1) / k)) for j in range(k - 1)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='projects/configs/VAD/_goal_anchors.py')
    p.add_argument('--lean', action='store_true',
                   help='forward resolution only, no lateral splits')
    p.add_argument('--ann', default=TRAIN)
    args = p.parse_args()

    d = pickle.load(open(args.ann, 'rb'))
    infos = [i for i in (d['infos'] if isinstance(d, dict) else d)
             if i.get('fut_valid_flag')]
    tp = np.stack([np.asarray(i['gt_ego_target_point'], np.float64
                              ).reshape(-1)[:2] for i in infos])
    cmd = np.array([int(np.asarray(i['gt_ego_fut_cmd']).argmax())
                    for i in infos])
    speed = np.array([float(np.asarray(i['gt_ego_lcf_feat'])[7])
                      for i in infos])

    anchors = [None] * 7
    plan = LEAN if args.lean else FULL
    for name, cs, mirror, xe, max_lat in plan:
        m = np.isin(cmd, cs)
        if name == 'STOP':          # the population the inference rule selects
            m = speed < 0.1
        x, y = tp[m, 0].copy(), tp[m, 1].copy()
        if mirror:
            y[np.isin(cmd[m], [2, 4])] *= -1
        kf = len(xe) - 1
        bx = np.clip(np.digitize(x, xe[1:-1], right=True), 0, kf - 1)

        cells, support = [], []
        for i in range(kf):
            s = bx == i
            ys = y[s] if s.sum() >= 20 else y
            ed = lateral_split(ys, max_lat)
            bounds = [-1e4] + ed + [1e4]
            by = (np.clip(np.digitize(y[s], ed, right=True), 0, len(ed))
                  if (ed and s.sum()) else np.zeros(int(s.sum()), int))
            for j in range(len(ed) + 1):
                sel = y[s][by == j] if (s.sum() and (by == j).any()) else ys
                yc = float(np.median(sel))
                lo, hi = bounds[j], bounds[j + 1]
                # width: the real bracket where finite, else twice the spread
                yw = (hi - lo) if (abs(lo) < 1e3 and abs(hi) < 1e3) else \
                    max(2.0 * float(np.std(sel)) if len(sel) > 2 else 2.0, 1.0)
                cells.append([float((xe[i] + xe[i + 1]) / 2),
                              float(xe[i + 1] - xe[i]), yc, float(yw)])
                support.append(int((by == j).sum()) if s.sum() else 0)

        for c in cs:
            table = [list(v) for v in cells]
            if mirror and c in (2, 4):      # right-hand commands: mirror y
                table = [[a, b, -cc, dd] for a, b, cc, dd in table]
            anchors[c] = table
        print(f'{name:<12} {len(cells):3d} anchors  support min {min(support):5d} '
              f'median {int(np.median(support)):6d}  -> {[CMD[c] for c in cs]}')

    with open(args.out, 'w') as f:
        f.write('"""Goal anchors per command: [x_centre, x_width, y_centre, '
                'y_width] in metres.\n\nGenerated by tools/make_goal_anchors.py '
                'from the train split only. Do not edit by hand.\n"""\n\n')
        f.write('goal_anchors = [\n')
        for c in range(7):
            f.write(f'    # {CMD[c]} ({len(anchors[c])} anchors)\n    [\n')
            for a, b, cc, dd in anchors[c]:
                f.write(f'        [{a:.2f}, {b:.2f}, {cc:.2f}, {dd:.2f}],\n')
            f.write('    ],\n')
        f.write(']\n')
    total = sum(len(a) for a in anchors)
    print(f'wrote {args.out}: {total} anchors, max per command '
          f'{max(len(a) for a in anchors)}')


if __name__ == '__main__':
    main()
