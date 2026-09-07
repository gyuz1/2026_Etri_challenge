"""Fast-eval config for VADLAW_etri_tiny_cached_nolcf_best.py.

Unlike the earlier aux_* heads, aux_bev_motion_feedback is NOT train-only:
the head's estimate is concatenated into ego_feats and runs at inference
too. So the eval model must declare the same aux_bev_motion settings as
training, or ego_fut_decoder is built 6 columns too narrow and the whole
decoder silently fails to load (size mismatch -> randomly initialized
weights, and an eval that measures nothing).

ego_fut_dec_hidden_dim=520 matches the surgical donor's hidden width, and
prism_posterior_lcf_idx only affects the posterior, which eval never runs
-- it is declared here purely so the checkpoint's tensor shapes line up.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_feedback=True,
        aux_bev_motion_temporal=True,
        aux_bev_motion_grid=4,
        aux_bev_motion_proj_dim=32,
        prism_posterior_lcf_idx=(0, 1, 2, 3, 4, 7),
        ego_fut_dec_hidden_dim=520,
    ))
