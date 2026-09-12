"""Fast-eval for VADLAW_etri_tiny_kd_nolcf_split_distill_newstack.py.

Every descriptor setting has to be repeated: frames and grid set the width
of both aux_bev_motion_head and ego_status_est_net, and the latter is on the
inference path -- it produces the status slot the planner reads.
aux_bev_future_motion builds a head the checkpoint carries weights for.
aux_bev_motion_norm is not repeated (training-loss scaling, no parameters).
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
    ))
