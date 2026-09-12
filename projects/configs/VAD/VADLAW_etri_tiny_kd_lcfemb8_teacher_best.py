"""Scheme-A teacher, best known configuration. Diagnostic, never submittable.

Everything measured so far, applied at once, with the motion descriptor
matched exactly to the student that will distill from it
(VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py).

  ego_lcf_embed_dim 8 + residual   the donor's ego columns are trained on
                                   RAW values (mean|w| 0.0481 vs the scene
                                   columns' 0.0223); a zero-init residual
                                   keeps them working. Measured: iteration
                                   100 loss_plan_reg 0.0068 with it, 0.3884
                                   without.
  ego_lcf-ON KD stage1 donor       those 8 columns reach 1.564x the scene
                                   columns by stage-1 epoch 4, against
                                   0.147x after a zero-padded stage 2.
  echo_cycle 0                     pulls the plan toward what the world
                                   model can round-trip; a teacher reading
                                   real ego status gets nothing back for it.
  cumulative ts weighting          the metric cumsums per-step deltas, so
                                   early steps displace every later one.
  aux_bev_motion: 3 frames, grid 8, accel targets, normalized,
  plus the future speed profile    identical to the student's, so the
                                   ego_scene_feats it distills were shaped
                                   by the same motion representation the
                                   student has to reproduce.

Why match the descriptor when the teacher's own status slot never reads it:
ego_scene_feats -- the 512-d half the student aligns against -- comes from
ego_agent_query/ego_map_query and therefore from the BEV encoder. Shaping
that encoder differently on the two sides would ask the student to recover
structure the teacher was never pushed to encode.
"""

_base_ = ['./VADLAW_etri_tiny_kd_lcfemb8_teacher.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
        aux_bev_future_motion_weight=0.5,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_TEACHER_A_best (8d residual, 3-frame, grid8, '
                      'accel targets, future speed profile)'))),
    ])
