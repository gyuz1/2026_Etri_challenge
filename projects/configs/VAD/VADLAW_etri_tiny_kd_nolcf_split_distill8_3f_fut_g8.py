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
    ))

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
