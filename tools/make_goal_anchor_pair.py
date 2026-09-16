"""Fit a TRAIN-only, forward-matched pair of goal-anchor tables.

Both variants share forward edges and per-command distance scales. The lat1
variant has one lateral centre per forward bin. The adaptive variant may split
a centre, but only when its children have enough TRAIN frames and distinct
scenes under the SAME normalized-nearest assignment used by the model.

Two-means is a quantization heuristic, NOT evidence of two behavioural modes.
All thresholds below are configurable engineering safeguards, not tuned optima.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


TRAIN = ('data/etri/.causal_regen_split_301_75_10hz/'
         'vad_etri_infos_temporal_train_split.pkl')
COMMANDS = ['LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT',
            'TURN_RIGHT', 'U_TURN', 'STOP']
# SAME starting ranges/resolution as make_goal_anchors.py FULL; neither
# variant is allowed to change these independently.
GROUPS = [
    ('LANE_KEEP', (0,), False, np.linspace(-1, 118, 25), 2),
    ('LANE_CHANGE', (1, 2), True, np.linspace(5, 116, 17), 2),
    ('TURN', (3, 4), True, np.linspace(-1, 46, 13), 4),
    ('U_TURN', (5,), False, np.linspace(-1, 23, 6), 2),
    ('STOP', (6,), False, np.array([-1., .5, 2., 5., 10., 17., 25.]), 1),
]


def nearest_labels(points, table):
    """Float32, width-normalized squared distance; first anchor wins a tie."""
    points = np.asarray(points, dtype=np.float32)
    table = np.asarray(table, dtype=np.float32)
    rel = (points[:, None, :] - table[None, :, 0::2]) / table[None, :, 1::2]
    return np.square(rel).sum(axis=-1).argmin(axis=-1)


def mirror_table(table):
    result = np.array(table, dtype=np.float32, copy=True)
    result[:, 2] *= -1
    return result


def support_report(points, commands, scenes, table, command_ids, mirror):
    """Measure each original command separately, never hide scarcity in pooling."""
    result = {}
    for cmd in command_ids:
        selected = commands == cmd
        cmd_table = mirror_table(table) if mirror and cmd in (2, 4) else table
        labels = nearest_labels(points[selected], cmd_table)
        cmd_scenes = scenes[selected]
        result[COMMANDS[cmd]] = [
            dict(frames=int(np.count_nonzero(labels == cell)),
                 scenes=int(len(np.unique(cmd_scenes[labels == cell]))))
            for cell in range(len(table))]
    return result


def weak_cells(support, min_frames, min_scenes):
    return sorted({cell for rows in support.values()
                   for cell, row in enumerate(rows)
                   if row['frames'] < min_frames or row['scenes'] < min_scenes})


def lateral_two_means(y):
    """Deterministic candidate split and its quantization gain, not a mode test."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return None
    centres = np.quantile(y, [.2, .8])
    if centres[0] == centres[1]:
        centres = np.array([y.min(), y.max()])
    if centres[0] == centres[1]:
        return None
    for _ in range(50):
        labels = np.abs(y[:, None] - centres).argmin(axis=-1)
        if not all(np.any(labels == j) for j in range(2)):
            return None
        updated = np.array([y[labels == j].mean() for j in range(2)])
        if np.allclose(updated, centres, rtol=0, atol=1e-10):
            centres = updated
            break
        centres = updated
    centres.sort()
    before = float(np.square(y - y.mean()).sum())
    after = float(np.square(y[:, None] - centres).min(axis=-1).sum())
    return centres, (before - after) / max(before, 1e-12)


def base_table(edges, canonical_points, y_width):
    labels = np.clip(np.digitize(canonical_points[:, 0], edges[1:-1],
                                right=True), 0, len(edges) - 2)
    fallback_y = float(canonical_points[:, 1].mean())
    cells = []
    for cell in range(len(edges) - 1):
        ys = canonical_points[labels == cell, 1]
        cells.append([(edges[cell] + edges[cell + 1]) / 2,
                      edges[cell + 1] - edges[cell],
                      float(ys.mean()) if len(ys) else fallback_y, y_width])
    return np.asarray(cells, dtype=np.float32)


def fit_group(points, commands, scenes, name, command_ids, mirror,
              initial_edges, max_lateral, min_frames=50, min_scenes=3,
              min_separation=1., min_relative_sse_gain=.2):
    """Merge forward bins jointly, then conservatively add lateral capacity."""
    selected = np.isin(commands, command_ids)
    pts, cmds, scs = points[selected], commands[selected], scenes[selected]
    if not len(pts) or any(not np.any(cmds == cmd) for cmd in command_ids):
        raise ValueError('missing TRAIN examples for command group ' + name)
    canonical = pts.copy()
    if mirror:
        canonical[np.isin(cmds, [2, 4]), 1] *= -1
    # A COMMON lateral metric prevents broad anchors from winning merely by
    # making their denominator larger. It is identical in both variants.
    y_width = max(2. * float(np.std(canonical[:, 1])), 1.)
    edges = list(np.asarray(initial_edges, dtype=float))
    merges = []
    while True:
        base = base_table(edges, canonical, y_width)
        support = support_report(pts, cmds, scs, base, command_ids, mirror)
        weak = weak_cells(support, min_frames, min_scenes)
        if not weak or len(base) == 1:
            break
        cell = min(weak, key=lambda i: (
            min(rows[i]['frames'] for rows in support.values()), i))
        # Merge with the neighbour yielding the smaller combined x interval.
        choices = []
        if cell > 0:
            choices.append((edges[cell + 1] - edges[cell - 1], cell))
        if cell + 1 < len(base):
            choices.append((edges[cell + 2] - edges[cell], cell + 1))
        _, boundary = min(choices)
        merges.append(dict(removed_edge=edges[boundary], weak_cell=cell,
                           support_before=support))
        del edges[boundary]

    lat1 = base.copy()
    adaptive = base.copy()
    bin_ids = list(range(len(base)))
    attempts = []
    # If the entire command has too few scenes, a single explicit fallback is
    # unavoidable; do not manufacture more apparent evidence by splitting it.
    if not weak:
        for forward_bin in range(len(base)):
            while bin_ids.count(forward_bin) < max_lateral:
                assigned = nearest_labels(canonical, adaptive)
                candidates = [i for i, b in enumerate(bin_ids) if b == forward_bin]
                candidates.sort(key=lambda i: (
                    -float(np.var(canonical[assigned == i, 1]))
                    if np.any(assigned == i) else 0., i))
                accepted = False
                for cell in candidates:
                    ys = canonical[assigned == cell, 1]
                    split = lateral_two_means(ys)
                    attempt = dict(forward_bin=forward_bin, parent_anchor=cell,
                                   parent_frames=int(len(ys)), accepted=False)
                    if split is None:
                        attempt['reason'] = 'no_nonempty_split'
                        attempts.append(attempt)
                        continue
                    centres, gain = split
                    attempt.update(centres=centres.tolist(), relative_sse_gain=gain)
                    if centres[1] - centres[0] < min_separation or gain < min_relative_sse_gain:
                        attempt['reason'] = 'insufficient_separation_or_gain'
                        attempts.append(attempt)
                        continue
                    children = np.repeat(adaptive[cell:cell + 1], 2, axis=0)
                    children[:, 2] = centres
                    proposed = np.concatenate([adaptive[:cell], children,
                                               adaptive[cell + 1:]], axis=0)
                    proposed_support = support_report(
                        pts, cmds, scs, proposed, command_ids, mirror)
                    bad = weak_cells(proposed_support, min_frames, min_scenes)
                    if bad:
                        attempt.update(reason='actual_nearest_support',
                                       weak_anchors=bad, support=proposed_support)
                        attempts.append(attempt)
                        continue
                    attempt.update(accepted=True, reason='supported_quantization_split')
                    attempts.append(attempt)
                    adaptive = proposed
                    bin_ids[cell:cell + 1] = [forward_bin, forward_bin]
                    accepted = True
                    break
                if not accepted:
                    break

    report = dict(
        commands=[COMMANDS[c] for c in command_ids], mirrored=mirror,
        starting_forward_edges=np.asarray(initial_edges).tolist(),
        forward_edges=edges, lateral_width=float(np.float32(y_width)),
        total_frames=int(len(pts)), total_scenes=int(len(np.unique(scs))),
        forward_merges=merges, lateral_split_attempts=attempts,
        fallback_below_min_support=bool(weak),
        variants={})
    for variant, table in [('lat1', lat1), ('adaptive', adaptive)]:
        per_cmd = support_report(pts, cmds, scs, table, command_ids, mirror)
        assigned = nearest_labels(canonical, table)
        pooled = [dict(frames=int(np.count_nonzero(assigned == i)),
                       scenes=int(len(np.unique(scs[assigned == i]))))
                  for i in range(len(table))]
        report['variants'][variant] = dict(
            anchors=int(len(table)), per_command_unmirrored=per_cmd,
            pooled_mirrored=pooled,
            weak_anchors=weak_cells(per_cmd, min_frames, min_scenes))
    return lat1, adaptive, report


def geometry_hash(tables):
    packed = json.dumps(tables, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(packed.encode('utf-8')).hexdigest()


def write_config(path, tables, metadata):
    text = ['"""TRAIN-only paired anchors; generated by make_goal_anchor_pair.py.',
            'The thresholds are safeguards, not proven optimal hyperparameters.',
            '"""', '', 'goal_anchors = [']
    for cmd, table in enumerate(tables):
        text.append('    # {} ({} anchors)'.format(COMMANDS[cmd], len(table)))
        text.append('    [')
        text.extend('        ' + repr(row) + ',' for row in table)
        text.append('    ],')
    text.extend([']', '', 'goal_anchor_pair_metadata = ' + repr(metadata), ''])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(text))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann', default=TRAIN)
    parser.add_argument('--out-dir', default='projects/configs/VAD')
    parser.add_argument('--report', default='reports/goal_anchor_pair_train.json')
    parser.add_argument('--min-frames', type=int, default=50)
    parser.add_argument('--min-scenes', type=int, default=3)
    parser.add_argument('--min-separation', type=float, default=1.)
    parser.add_argument('--min-relative-sse-gain', type=float, default=.2)
    args = parser.parse_args()
    if args.min_frames < 1 or args.min_scenes < 1 or args.min_separation < 0:
        parser.error('support thresholds must be positive; separation nonnegative')
    if not 0 <= args.min_relative_sse_gain <= 1:
        parser.error('--min-relative-sse-gain must be between 0 and 1')
    # Make accidental use of the configured validation file fail loudly.
    if 'train' not in Path(args.ann).name.lower():
        parser.error('--ann must explicitly name the TRAIN annotation file')
    with open(args.ann, 'rb') as stream:
        data = pickle.load(stream)
    infos = data['infos'] if isinstance(data, dict) else data
    selected = [i for i in infos
                if i.get('fut_valid_flag', False)
                and i.get('gt_ego_long_fut_valid_flag', False)
                and np.isfinite(np.asarray(i['gt_ego_target_point'])).all()]
    points = np.stack([np.asarray(i['gt_ego_target_point'], dtype=np.float32)
                       .reshape(-1)[:2] for i in selected])
    commands = np.array([np.asarray(i['gt_ego_fut_cmd']).argmax() for i in selected])
    scenes = np.array([str(i['scene_token']) for i in selected])
    speed = np.array([np.asarray(i['gt_ego_lcf_feat']).reshape(-1)[7] for i in selected])
    scene_ids = sorted(set(scenes.tolist()))
    thresholds = dict(min_frames=args.min_frames, min_scenes=args.min_scenes,
                      min_separation=args.min_separation,
                      min_relative_sse_gain=args.min_relative_sse_gain)
    report = dict(
        schema_version=1, train_only=True, source_path=str(Path(args.ann).resolve()),
        source_sha256=file_sha256(args.ann), source_frames=len(infos),
        retained_frames=len(selected), excluded_frames=len(infos) - len(selected),
        scene_ids=scene_ids,
        scene_ids_sha256=hashlib.sha256('\n'.join(scene_ids).encode()).hexdigest(),
        assignment='float32 sum(((tp-centre)/width)**2), argmin first tie',
        thresholds=thresholds,
        notes=[
            'No validation/test labels read or thresholds tuned against them.',
            'Two-means reduces quantization error; it does not establish multimodality.',
            'Adjacent frames are correlated; unique-scene thresholds are safeguards only.',
            'Mirror pooling proposes geometry; support is required for EACH original command.',
            'STOP fitting uses its TRAIN command labels, not privileged speed routing.',
            'One fallback anchor remains when an entire command has insufficient scenes.',
        ],
        stop_population_diagnostic=dict(
            train_stop_frames=int(np.count_nonzero(commands == 6)),
            gt_speed_below_point1_frames=int(np.count_nonzero(speed < .1)),
            train_stop_speed_above_point1=int(np.count_nonzero((commands == 6) & (speed >= .1))),
            gt_speed_below_point1_nonstop=int(np.count_nonzero((commands != 6) & (speed < .1)))),
        groups={}, geometry_sha256={})
    variants = dict(lat1=[None] * 7, adaptive=[None] * 7)
    for name, command_ids, mirror, edges, max_lateral in GROUPS:
        lat1, adaptive, group = fit_group(
            points, commands, scenes, name, command_ids, mirror, edges,
            max_lateral, **thresholds)
        report['groups'][name] = group
        for variant, table in [('lat1', lat1), ('adaptive', adaptive)]:
            for cmd in command_ids:
                cmd_table = mirror_table(table) if mirror and cmd in (2, 4) else table
                variants[variant][cmd] = cmd_table.tolist()
        print('{}: forward {} -> {}, lateral anchors {} -> {}, fallback={}'.format(
            name, len(edges) - 1, len(group['forward_edges']) - 1,
            len(lat1), len(adaptive), group['fallback_below_min_support']))
    for variant, tables in variants.items():
        digest = geometry_hash(tables)
        report['geometry_sha256'][variant] = digest
        metadata = dict(variant=variant, train_only=True,
                        geometry_sha256=digest,
                        source_sha256=report['source_sha256'],
                        scene_ids_sha256=report['scene_ids_sha256'],
                        thresholds=thresholds)
        write_config(Path(args.out_dir) / ('_goal_anchors_pair_' + variant + '.py'),
                     tables, metadata)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Wrote paired configs and ' + args.report)


if __name__ == '__main__':
    main()
