"""Eval config for VADLAW_etri_tiny_clean_goalgrid.py.

goal_grid_size sets the decoder's last-layer width and goal_cls_head's, and
goal_grid_select decides the collapse, so both must match training.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_clean.py']

model = dict(
    pts_bbox_head=dict(
        goal_grid_size=(5, 5),
        goal_grid_range=(-5.0, 110.0, -25.0, 25.0),
        goal_grid_select='soft',
    ))
