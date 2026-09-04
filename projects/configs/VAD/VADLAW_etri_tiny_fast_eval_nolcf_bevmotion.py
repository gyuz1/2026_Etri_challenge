"""Fast-eval (T_infer measurement) pipeline for the ego_lcf-off + aux_bev_motion
checkpoint (VADLAW_etri_tiny_cached_nolcf_bevmotion.py) -- see
VADLAW_etri_tiny_cached_nolcf.py for the compliance rationale. Only
bev_refine_steps changes vs VADLAW_etri_tiny_fast_eval_nolcf.py: it's the
one architectural setting that both affects the forward (inference) pass
and differs from that config's default (1) -- the checkpoint was trained
with a 3-stage cascaded BEV refine, so eval needs the same stage count or
the extra two learned correction stages simply never run. Every other
difference between the training config and plain fast_eval_nolcf.py
(aux_bev_motion, echo_cycle_weight, prev_bev_dropout, loss_plan_bound
weight, ...) is training/loss-only and doesn't touch inference.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf.py']

model = dict(
    pts_bbox_head=dict(
        bev_refine_steps=3,
    ))
