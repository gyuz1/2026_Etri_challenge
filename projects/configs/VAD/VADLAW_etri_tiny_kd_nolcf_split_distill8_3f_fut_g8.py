"""Scheme-A student v5: finer BEV motion descriptor (grid 8).

WHAT CHANGES VS ..._3f_fut.py
aux_bev_motion_grid 4 -> 8. Descriptor goes 512 -> 2048 per frame
(6144 total across 3 frames).

WHY
The descriptor is what ego_status_est_net reads to produce the 8-d status
slot the planner consumes, and at grid 4 it compresses a 100x100x256 BEV
into 512 numbers -- 1/5000 of it. Each cell then covers 15.0m x 7.5m of
real space.

That matters because of what the descriptor is FOR. At 10 m/s the ego moves
5m between frames; inside a 15m cell that displacement is averaged away
before the difference is taken. Grid 8 puts a cell at 7.5m x 3.8m, so the
same motion crosses cell boundaries and survives into d1 and d1 - d2.

Also relevant: can_bus injects real ego velocity/accel into bev_queries
before the encoder runs (transformer.py's
`bev_queries = bev_queries + can_bus_mlp(can_bus)`), so bev_embed genuinely
contains ego-motion structure. The estimator's job is to recover it, and a
1/5000 summary is a thin window onto it. Reading bev_embed is the shared-
feature path every head uses, so this is the compliant route -- unlike a
direct ego_lcf -> planner edge, which is what the ban covers.

Grid 16 was considered and rejected: 8192 per frame puts 6.3M parameters in
the head's first layer, too much to fit in stage 2's 12 epochs.

COST
Only the auxiliary head and ego_status_est_net widen; no decoder or
inference-path shape changes beyond the status slot's source. proj_dim
stays 32.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_grid=8,
        # Must equal the teacher's. The estimator still emits all 8 channels
        # into the decoder slot; only the 6 informative ones are compared,
        # so the cosine cannot be satisfied by echoing two constants.
        ego_status_distill_idx=(0, 1, 2, 3, 4, 7),
        # The lever. The slot the planner consumes was supervised only by a
        # cosine to the teacher, and cosine is scale-invariant -- it cannot
        # separate 10.5 m/s from 5.2 m/s. L2 is brutally sensitive to exactly
        # that: measured on this val split, a 0.25 m/s speed RMSE costs 0.5039
        # against the 0.2708 perfect-kinematics oracle, and LANE_KEEP (85% of
        # our error) is where it lands. This forces the physical state into
        # the slot itself rather than leaving it in aux_bev_motion_head, whose
        # output reaches nothing.
        ego_status_decode=True,
        ego_status_decode_weight=0.5,
    ),
    # Point at the descriptor-matched teacher. Distilling ego_scene_feats
    # from a grid-4 teacher into a grid-8 student would ask the student to
    # reproduce features shaped by a coarser motion view than its own.
    feature_distill_teacher_cfg=(
        'projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py'),
    feature_distill_teacher_ckpt=(
        'work_dirs/stage2_kd_lcfemb8_teacher_best/epoch_12.pth'),
)

# Equal frame gaps, matching the teacher and stage 1. The student reads this
# descriptor to produce the status slot its planner consumes, so an
# acceleration block that means something different in training than at test
# time lands directly in the submitted trajectory.
data = dict(train=dict(history_sampling='fixed'))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_STUDENT_A_v5 (grid8 descriptor, 3-frame, '
                      'future speed profile)'))),
    ])

# MUST be overridden, for the same reason as the teacher's: the base names the
# OLD nolcf stage 1, whose 520-wide merge is still on disk and would load
# silently while carrying a grid-4, 2-frame BEV encoder.
#
# This stays on the nolcf lineage, and that is a compliance requirement rather
# than a tuning choice -- the submitted model must never inherit weights from
# a network trained with real ego status feeding its decoder. The ego_lcf-ON
# stage 1 donates only to the teacher, which is not submitted.
#
# Built by tools/merge_stage1_world_model.py on work_dirs/stage1_best_nolcf,
# then widened 512 -> 520 with
#   tools/surgical_ego_fut_decoder_transfer.py --ego-lcf-n 0 --pad-input-cols 8
# so the scene half transfers intact and the 8 status columns start as an
# exact no-op. Those columns read a vision estimate the student also has to
# learn to produce, so no lineage has a pretrained value for them.
load_from = 'work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth'
