"""Fast-eval for VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py.

aux_bev_motion_grid must be repeated: it sets the descriptor width, which
is the input width of BOTH aux_bev_motion_head and ego_status_est_net --
and the latter is on the inference path, feeding the planner's status slot.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut.py']

# ONE model dict. This file used to assign `model` twice, and a config is
# plain Python: the second assignment replaced the first, so
# aux_bev_motion_grid=8 never reached mmcv. The eval network came out grid 4
# (1536 wide) against a grid-8 checkpoint, and ego_status_est_net -- which
# feeds the planner's status slot -- would have loaded as random weights
# under strict=False. Caught by audit_pipeline.py's parity check 2026-09-15,
# before any stage-2 eval ran.
model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_grid=8,
        bev_residual_refine=False,
        aux_long_horizon=True,
        aux_long_horizon_residual=True,
    ))
