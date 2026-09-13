"""Compliant student WITHOUT distillation. Submittable.

Same as VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py in every
respect except that no teacher is attached, so it runs the moment stage 1
finishes instead of waiting the ~22h the teacher takes.

WHY IT EARNS A GPU RATHER THAN LEAVING ONE IDLE
The teacher is a hard serial dependency -- VAD_LAW loads its checkpoint in
the constructor, so the distilled student cannot start until epoch_12.pth
exists. That leaves the 3090 free for exactly one teacher-length slot, and
this is the most useful thing to put in it:

  1. It isolates what the stage-1 rebuild bought on its own. Stage 1 was
     restarted three times today for the acceleration block, the frame-gap
     sampling, and the rotated prev_bev2. Against A student v1's 0.4218 --
     which used none of them and no better teacher either -- the gap here
     is those fixes plus GT planning supervision, with distillation held
     out. Nothing else in the schedule measures that.
  2. It is a submittable fallback. If the teacher lands worse than v1's
     0.2328, or the distilled student does not clear 0.4218, this model is
     already trained and already compliant.

WHAT CHANGES
feature_distill_teacher_cfg/_ckpt dropped, and both split weights set to 0
so VAD_LAW's constructor check (which rejects 'split' mode with both
weights at zero) is not tripped -- mode goes back to the default instead.
echo_cycle_weight stays 0.1, matching the distilled student, so the two
differ only in distillation.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py']

model = dict(
    feature_distill_teacher_cfg=None,
    feature_distill_teacher_ckpt=None,
    feature_distill_mode='fused',
    scene_distill_weight=0.0,
    status_distill_weight=0.0,
    feature_distill_weight=0.0,
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_STUDENT_no-distill (rebuilt stage1 only: '
                      '3-frame, grid8, accel, future speed, GT planning)'))),
    ])
