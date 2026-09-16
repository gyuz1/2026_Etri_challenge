"""Eval config for VADLAW_etri_tiny_clean_goalanchors_adaptive.py.

Anchors, goal_scale and goal_long_ts define the head and must match training.
goal_expose_candidates emits one trajectory per anchor for
eval_holdout_l2_and_tinfer.py --select-goal-by-tp; without that flag the
submitted trajectory is the network's own argmax goal.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_clean.py',
          './_goal_anchors_pair_adaptive.py']

model = dict(
    pts_bbox_head=dict(
        goal_pred=True,
        goal_anchors={{_base_.goal_anchors}},
        goal_scale=(115.0, 25.0),
        goal_long_ts=10,
        goal_cls_weight=0.5,
        goal_off_weight=0.5,
        goal_follow_weight=0.1,
        goal_select_weight=1.0,
        goal_expose_candidates=True,
    ))
