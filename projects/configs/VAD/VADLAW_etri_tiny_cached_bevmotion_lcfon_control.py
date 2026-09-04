"""Diagnostic control, NOT for submission: identical to
VADLAW_etri_tiny_cached_nolcf_bevmotion.py's full technique stack (PRISM
S=2, prev_bev_dropout, echo_cycle, cascaded refine x3, EMA,
col/dir/bound loss, aux_bev_motion) with exactly one thing flipped back
on -- ego_lcf status feeding the planner directly, matching the original
pre-compliance baseline's ego_lcf_feat_idx.

Purpose: the ego_lcf-off run (this same technique stack) lands at 0.618m
L2, far short of the pre-compliance 0.218m baseline. Those two numbers
were never a controlled comparison -- the 0.218m baseline predates every
technique in this file. This run isolates the two candidate explanations:
if L2 comes back down near 0.218-0.25m, the gap is the ego_lcf-removal
cost itself; if it stays far above that, something in the technique
stack has a regression independent of ego_lcf.

Stage-1 source for the merge is stage1_etri_split_301_75_10hz/epoch_48.pth
(ego_lcf ON, ego_fut_decoder already 520-wide) -- shapes match this
config directly, so the merge needs no
--override-prefixes-from-world-model; stage1's own ETRI-domain-adapted
decoder transfers as-is.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_LCFON_CONTROL (same stack as bevmotion, '
                      'ego_lcf back on -- diagnostic only, not '
                      'compliant)'))),
    ])
