"""Cached-geometry eval pipeline for the ego_lcf-off ablation checkpoint --
see VADLAW_etri_tiny_cached_nolcf.py for the compliance rationale.
"""

_base_ = ['./VADLAW_etri_tiny_cached_eval.py']

model = dict(
    use_ego_lcf_status=False,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=None,
    ))
