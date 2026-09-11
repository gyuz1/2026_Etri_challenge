"""Fast-eval for VADLAW_etri_tiny_kd_lcfemb8_teacher_3f.py.

aux_bev_motion_frames=3 must be repeated: it sets aux_bev_motion_head's
input width, which the checkpoint carries weights for.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_kd_lcfemb_teacher.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_embed_dim=8,
        ego_lcf_embed_hidden=64,
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_temporal=True,
        aux_bev_motion_frames=3,
    ))
