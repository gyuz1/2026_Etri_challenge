"""Scheme-A student v2 -- 8-d vision-estimated ego status. Compliant.

Matches VADLAW_etri_tiny_kd_lcfemb8_teacher.py's 8-d status block instead
of v1's 64-d one. See that config's docstring for the measurement behind
the change; the short version is that 64 forced a 576-wide ego_feats no
stage 1 can donate to, so those columns started at zero and reached only
0.147x the scene columns' magnitude in 11 epochs.

WHAT CHANGES VS VADLAW_etri_tiny_kd_nolcf_split_distill.py
Only ego_status_est_dim, 64 -> 8. The estimator's internal width is
unaffected -- ego_status_est_net is Linear(1024 -> 256) -> ReLU ->
Linear(256 -> est_dim), so it still expands the BEV motion descriptor
internally and only its output narrows.

NOTE ON THE STUDENT'S OWN DONOR
The student is compliant, so its ego_feats is 512 (scene) + 8 (estimated)
= 520, while its stage-1 donor (the nolcf KD stage 1) is 512 wide. Those 8
columns are therefore still zero-padded on the STUDENT side -- the
ego_lcf-ON stage 1 cannot be the student's donor, since the student must
never have been trained with real ego status feeding its decoder.

That is not the same situation as the teacher's, though: the student's 8
columns read a VISION estimate it also has to learn to produce, so there
is no pretrained "correct" value for them to inherit in any lineage. What
the change buys on this side is simply 8 columns to grow instead of 64,
and a target (the teacher's 8-d embedding) that is itself better trained.

Modality dropout stays at 0.3 -- the estimate still lands in the decoder's
input, which is the structure aux_bev_motion_feedback lost 5.5% to.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill.py']

model = dict(
    pts_bbox_head=dict(
        ego_status_est_dim=8,
    ),
    feature_distill_teacher_cfg=(
        'projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher.py'),
    feature_distill_teacher_ckpt=(
        'work_dirs/stage2_kd_lcfemb8_teacher/epoch_12.pth'),
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_STUDENT_A_v2 SPLIT_DISTILL (scene 512 + '
                      'vision-estimated status 8 <- Scheme-A v2 teacher, '
                      'w=0.3/0.5, modality dropout 0.3)'))),
    ])
