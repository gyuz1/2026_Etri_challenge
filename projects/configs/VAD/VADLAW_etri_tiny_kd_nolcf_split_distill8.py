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
        # Adds acceleration (idx 2, 3) to what the BEV branch is asked to
        # regress, and normalizes the L1 per component.
        #
        # Both changes come from measuring kinematic oracles on this val
        # split with the competition's own L2 windowing
        # (tools/kinematic_oracle_ceiling.py):
        #
        #   stay put                        12.5175
        #   perfect velocity, a assumed 0    0.5965
        #   perfect velocity AND accel       0.2708
        #
        # Acceleration is worth more than everything velocity buys on top of
        # nothing: 0.5965 -> 0.2708 is a 55% cut, and on LANE_KEEP (85% of
        # our total error) it is 0.5543 -> 0.2524. Yet idx 2/3 were excluded
        # entirely, so nothing ever pushed the BEV features to encode it.
        # The compliant student currently sits at 0.4218, i.e. between the
        # velocity-only and velocity+accel oracles -- consistent with it
        # having velocity and lacking acceleration.
        #
        # The norm values are the train-split per-component std
        # (vx 5.7040, vy 0.1715, ax 0.4625, ay 0.3579, yaw_rate 0.0547,
        # speed 5.7050), ordered to match aux_bev_motion_idx. Without them
        # vx and speed took 98.7% of the L1 and yaw_rate 0.1%, so turning
        # behaviour was effectively unsupervised -- and vx/speed are nearly
        # the same number here anyway, since speed = norm(vx, vy).
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_norm=(5.7040, 0.1715, 0.4625, 0.3579, 0.0547, 5.7050),
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

# MUST be overridden. The base (v1, 64-d) loads a 576-wide donor, but this
# config's status slot is 8-d so ego_fut_decoder is 520 wide -- and mmcv
# loads with strict=False, so a 576 donor does not raise. It silently drops
# every ego_fut_decoder weight and trains from a random planner. Caught by
# comparing the donor's shapes against the built model before launching.
#
# The 520 donor is the same 512-wide KD stage1 merge with 8 zero columns
# appended (tools/surgical_ego_fut_decoder_transfer.py --ego-lcf-n 0
# --pad-input-cols 8), so the scene half transfers intact and the status
# columns start as an exact no-op.
load_from = (
    'work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/'
    'stage2_init_merged_lcfemb8.pth')
