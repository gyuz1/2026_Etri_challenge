"""Prints a Python-literal list of sqrt-inverse-frequency weights for
VAD_head's command_class_weights, computed from an actual train pkl's
command distribution -- never hardcode these, the ratio drifts between
tracks (split_301_75 vs fulldata) and whenever either pkl's scene set
changes.

weight[cmd] = sqrt(count[most_common] / count[cmd]), so the most common
command (LANE_KEEP in practice) always gets weight 1.0 and rarer commands
get a >1 multiplier on loss_plan_reg. sqrt, not raw inverse frequency,
deliberately: raw inverse frequency gave U_TURN (~0.1% of ETRI train
frames) a ~540x weight, which would let ~134 samples dominate the
gradient; sqrt dampens that to ~23x while still meaningfully boosting the
rare classes.

Usage:
    python tools/compute_command_class_weights.py \\
        data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_train_split.pkl
"""
import argparse
import pickle

import numpy as np

# Must match etri_vad_converter(_10hz).py's COMMAND_VOCAB order exactly --
# index i here is mode i of the model's ego_fut_mode=7 trajectory decoder.
COMMAND_VOCAB = (
    'LANE_KEEP', 'LANE_CHANGE_L', 'LANE_CHANGE_R', 'TURN_LEFT', 'TURN_RIGHT',
    'U_TURN', 'STOP',
)


def compute_weights(ann_file):
    with open(ann_file, 'rb') as handle:
        infos = pickle.load(handle)['infos']
    counts = np.zeros(len(COMMAND_VOCAB), dtype=np.int64)
    for info in infos:
        counts[int(np.asarray(info['gt_ego_fut_cmd']).argmax())] += 1
    if (counts == 0).any():
        missing = [COMMAND_VOCAB[i] for i in range(len(counts)) if counts[i] == 0]
        raise ValueError(
            f'{ann_file} has zero samples for command(s) {missing} -- '
            'sqrt(count.max()/0) is undefined. Check the pkl or drop '
            'those commands from the weighting.')
    weights = np.sqrt(counts.max() / counts)
    return counts, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ann_file')
    args = parser.parse_args()

    counts, weights = compute_weights(args.ann_file)
    for name, count, weight in zip(COMMAND_VOCAB, counts, weights):
        print(f'{name:15s} count={count:7d} weight={weight:.4f}')
    # This exact line is what the pipeline script greps out and feeds to
    # --cfg-options model.pts_bbox_head.command_class_weights=[...]
    print('WEIGHTS=' + '[' + ','.join(f'{w:.4f}' for w in weights) + ']')


if __name__ == '__main__':
    main()
