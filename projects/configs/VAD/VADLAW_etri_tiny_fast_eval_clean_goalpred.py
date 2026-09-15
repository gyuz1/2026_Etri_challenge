"""Eval config for VADLAW_etri_tiny_clean_goalpred.py.

goal_pred adds goal_cls_head / goal_off_head / goal_embed, lengthens the
decoder to goal_long_ts, and changes forward; the bin edges and lateral scale
define what the offsets mean. All must match training.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_clean.py']

model = dict(
    pts_bbox_head=dict(
        goal_pred=True,
        goal_bin_edges=(-5.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
                        75.0, 85.0, 95.0, 105.0, 115.0),
        goal_lat_scale=25.0,
        goal_long_ts=10,
    ))
