"""Eval config for VADLAW_etri_tiny_clean_goalpred.py that also emits one
trajectory per goal bin, so eval_holdout_l2_and_tinfer.py --select-goal-by-tp
can pick among them.

Identical network and weights; goal_expose_candidates only adds K extra
decoder passes at inference and changes no training-time behaviour. The
candidates are generated from the network's own predicted goals, without the
target point. Use the plain eval config for the number that uses no target
point at all.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_clean_goalpred.py']

model = dict(pts_bbox_head=dict(goal_expose_candidates=True))
