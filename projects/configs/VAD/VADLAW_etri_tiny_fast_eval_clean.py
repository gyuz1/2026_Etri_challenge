"""Eval config for VADLAW_etri_tiny_clean_student.py and _clean_nodistill.py.

prism_latent_supervision builds prism_prior_net / prism_z_proj, which add to
ego_feats at inference, so it must match the train config. The dropout
switches are inactive at eval either way.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py']

model = dict(pts_bbox_head=dict(prism_latent_supervision=False))
