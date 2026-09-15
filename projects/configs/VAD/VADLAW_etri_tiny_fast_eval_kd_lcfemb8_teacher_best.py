"""Fast-eval for VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py."""

_base_ = ['./VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher.py']

# ONE model dict. A second top-level `model = dict(...)` used to follow this
# one and silently replaced it (a config is plain Python), dropping frames=3,
# grid=8 and the future-motion head from the eval network. Match the train
# config; distillation-only settings do not change the eval network but
# bev_residual_refine does.
model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
        # Set in VADLAW_etri_tiny_kd_lcfemb8_teacher.py and missing here until
        # 2026-09-15. It changes no tensor shape -- only whether the raw
        # ego_lcf columns are added back onto the embedding output
        # (VAD_head.py forward) -- so the shape parity check could not see it,
        # and the teacher would have been scored without the input it was
        # trained on.
        ego_lcf_embed_residual=True,
        bev_residual_refine=False,
        aux_long_horizon=True,
        aux_long_horizon_residual=True,
    ))
