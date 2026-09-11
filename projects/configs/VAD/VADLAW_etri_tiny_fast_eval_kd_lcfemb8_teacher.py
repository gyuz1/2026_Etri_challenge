"""Fast-eval for VADLAW_etri_tiny_kd_lcfemb8_teacher.py (Scheme-A teacher v2).

Repeats every setting that builds a module the checkpoint carries weights
for: the 8-d ego_lcf embedding (which sets ego_fut_dec_in_dim to 520), and
the aux_bev_motion block, whose head width depends on both the temporal
descriptor and the number of regressed components.

aux_bev_motion_norm is not repeated -- it only scales a training loss and
creates no parameters.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_kd_lcfemb_teacher.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_embed_dim=8,
        ego_lcf_embed_hidden=64,
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_temporal=True,
    ))
