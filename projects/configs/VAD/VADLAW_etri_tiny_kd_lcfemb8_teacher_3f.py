"""Scheme-A teacher for the 3-frame student. Diagnostic, never submittable.

Identical to VADLAW_etri_tiny_kd_lcfemb8_teacher.py except
aux_bev_motion_frames=3, matching
VADLAW_etri_tiny_kd_nolcf_split_distill8_3f.py.

Why the teacher has to match: the student's ego_status_est_net reads the
BEV motion descriptor, and the split distillation aligns that estimate
against the teacher's ego_lcf embedding. If the two sides build the
descriptor from a different number of frames, the teacher is answering
from a different observation window than the student can see, which is the
same "teacher solves an easier problem" failure the ego-state prompt had.

The teacher reads real ego_lcf either way, so the extra frame changes
little for its own accuracy -- it is here to keep the pair symmetric.
"""

_base_ = ['./VADLAW_etri_tiny_kd_lcfemb8_teacher.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_TEACHER_A_v3 (3-frame descriptor, 8-d '
                      'embedding, ego_lcf-ON KD stage1)'))),
    ])
