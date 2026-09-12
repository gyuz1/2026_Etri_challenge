"""Fast-eval for VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py."""

_base_ = ['./VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
    ))
