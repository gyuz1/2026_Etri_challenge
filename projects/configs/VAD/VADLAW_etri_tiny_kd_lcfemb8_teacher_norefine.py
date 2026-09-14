"""Scheme-A teacher WITHOUT the BEV trajectory refine. Diagnostic, never submittable.

Identical to VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py except
bev_residual_refine. Run alongside it as a controlled ablation: the A5000
trains with refine, the 3090 without, and the student distills from whichever
wins. Neither run is wasted -- both produce a usable teacher.

WHY
refine_ego_trajs_with_bev turned out to cost L2 rather than buy it. On the
finished stage-1 donor, skipping the call at evaluation -- module still
built and loaded, only the call removed -- improved L2 by 21-25%, reproduced
on two checkpoints:

    epoch 24   refine on 0.6395  ->  off 0.4799   (-25.0%)
    epoch 48   refine on 0.6089  ->  off 0.4807   (-21.1%)

It costs almost nothing in time (196.2ms -> 194.9ms), so this is not a
latency trade: the module was simply making the trajectory worse.

The telling part is that the refine-off numbers barely move between epoch 24
and 48 (0.4799 -> 0.4807) while the refine-on numbers improve (0.6395 ->
0.6089). The coarse trajectory had already converged by epoch 24, and the
remaining half of training went into repairing what refine did to it. Turning
it off at evaluation recovers that for free; turning it off during training
should let those epochs go into the plan itself instead.

That last step is what this config tests -- it has never been trained without
refine, so it could also come out worse, which is why the other machine keeps
the refine version rather than both switching.

SHARED_CONTEXT previously listed bev_refine_steps=3 as "해롭기 어려움 → 유지"
on the reasoning that its MLPs are zero-init and therefore start as an exact
no-op. That reasoning was never measured and is now contradicted: starting as
a no-op says nothing about where 48 epochs of gradient take it.
"""

_base_ = ['./VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py']

model = dict(pts_bbox_head=dict(bev_residual_refine=False))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name='stage2_TEACHER_A_best NO-REFINE (ablation vs A5000)')),
    ])
