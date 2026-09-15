"""Eval config for VADLAW_etri_tiny_nodistill_goalgrid.py.

goal_grid_size sets the decoder's last-layer width and builds goal_cls_head,
so it must be repeated here or the checkpoint loads into the wrong shape.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nodistill_best.py']

model = dict(
    pts_bbox_head=dict(
        goal_grid_size=(7, 3),
        goal_grid_range=(-5.0, 110.0, -25.0, 25.0),
        goal_grid_weight=0.5,
        goal_grid_select='soft',
    ))
