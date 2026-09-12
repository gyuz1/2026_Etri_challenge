"""Fast-eval for VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py.

aux_bev_motion_grid must be repeated: it sets the descriptor width, which
is the input width of BOTH aux_bev_motion_head and ego_status_est_net --
and the latter is on the inference path, feeding the planner's status slot.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_grid=8,
    ))
