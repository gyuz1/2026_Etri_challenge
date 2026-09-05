"""Fast-eval config for the surgical-decoder-transfer checkpoint
(VADLAW_etri_tiny_cached_nolcf_bevmotion_surgical.py) -- same as
VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py's bev_refine_steps=3 fix,
plus ego_fut_dec_hidden_dim=520 to match the trained checkpoint's
hidden width. Without this, the eval model builds ego_fut_decoder at
the default hidden_dim=512 and the whole decoder (all 3 layers) gets
silently shape-mismatched and randomly re-initialized at checkpoint
load -- exactly the kind of silent bug this repo has hit before.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        ego_fut_dec_hidden_dim=520,
    ))
