"""Fast-eval (T_infer measurement) pipeline for the ego_lcf-off ablation
checkpoint -- see VADLAW_etri_tiny_cached_nolcf.py for the compliance
rationale. Only the model's ego_lcf toggle changes; test_pipeline is
inherited from VADLAW_etri_tiny_fast_eval.py unchanged.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval.py']

model = dict(
    use_ego_lcf_status=False,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=None,
    ))
