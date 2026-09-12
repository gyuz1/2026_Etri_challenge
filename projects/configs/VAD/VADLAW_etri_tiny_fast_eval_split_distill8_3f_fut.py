"""Fast-eval for VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut.py.

aux_bev_future_motion must be repeated: it builds
aux_bev_future_motion_head, which the checkpoint carries weights for. The
head runs no decoder and is train-only in effect, but the MODULE still has
to exist at the same shape or its weights silently fail to load.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8_3f.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
    ))
