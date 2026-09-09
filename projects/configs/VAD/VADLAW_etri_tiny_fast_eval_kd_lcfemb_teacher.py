"""Fast-eval config for VADLAW_etri_tiny_kd_lcfemb_teacher.py -- the
Scheme-A teacher, ego_lcf ON as a learned 64-d embedding, target-point FREE.

Diagnostic only, never submittable: ego_lcf feeds the decoder.

Same construction rules as VADLAW_etri_tiny_fast_eval_kd_lcfon_diag.py
(read its docstring for why each line is here). The one addition:

- ego_lcf_embed_dim=64: this teacher does NOT append the 8 raw columns.
  It maps them through Linear(8->64)->ReLU->Linear(64->64) first, so
  ego_fut_dec_in_dim is 512+64=576, not 512+8=520, and the checkpoint
  carries ego_lcf_embed_net weights that only exist when this is set.
  Omitting it would build a 520-wide decoder and size-mismatch on load.

plan_reg_ts_weight_mode is deliberately not repeated: it only reweights
loss_planning, which never runs here.

Verified against the training config by building both models and diffing
their pts_bbox_head arguments -- do that again before trusting any number
out of this file (tools/diff_eval_config.py).
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval.py']

model = dict(
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        ego_lcf_embed_dim=64,
        ego_fut_dec_hidden_dim=512,
        prism_posterior_lcf_idx=(0, 1, 4, 7),
        bev_refine_steps=3,
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 4, 7),
    ))
