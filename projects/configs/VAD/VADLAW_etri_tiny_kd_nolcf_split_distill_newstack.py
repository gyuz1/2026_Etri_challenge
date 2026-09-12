"""A-scheme student, v1 lineage + the new motion-descriptor stack.

THE POINT OF THIS RUN
VADLAW_etri_tiny_kd_nolcf_split_distill.py measured 0.4218m against a
0.4885m baseline. This config changes ONLY the motion descriptor and what is
regressed from it, keeping the same teacher (A teacher v1, 0.2328m, already
trained and evaluated), the same 64-d status slot, the same distillation
weights and the same init checkpoint. So the difference it measures is
attributable to the new stack alone:

  aux_bev_motion_frames  2 -> 3     acceleration becomes observable at all
  aux_bev_motion_grid    4 -> 8     a descriptor cell stops being larger
                                    than the motion it is meant to measure
  aux_bev_motion_idx     +ax, ay    those targets now have an observable
  aux_bev_motion_norm    added      yaw_rate was 0.1% of the L1
  aux_bev_future_motion  added      supervises what the ego is ABOUT to do

Running it against the EXISTING v1 teacher rather than waiting for the v2/v3
teacher is deliberate: the descriptor settings barely affect a teacher --
its status slot is ego_lcf_embed_net(real ego_lcf) and never reads the
descriptor -- while they decide what the student's ego_status_est_net can
see. Waiting ~9h for a teacher whose relevant properties are unchanged would
buy nothing on this question.

WHY EACH PIECE (measured)
Kinematic oracles, val split, competition L2 windowing
(tools/kinematic_oracle_ceiling.py): perfect velocity 0.5965, perfect
velocity AND acceleration 0.2708. Our student at 0.4218 sits between them,
which is what a model with velocity and without acceleration looks like.

Descriptor resolution: at grid 4 a cell covers 15.0m x 7.5m, while the ego
moves ~5m between frames at 10 m/s -- the displacement averages away inside
one cell before the difference is taken. Grid 8 puts a cell at 7.5m x 3.8m.

Future speed profile: the speed 3s out differs from the current speed by
0.80 m/s on average (sd 1.32) over 4000 val samples. That is precisely what
kinematic extrapolation discards, and the 0.14 leaderboard entry is 48%
below even the perfect-kinematics oracle, so it cannot be reached without
predicting it.

Still 64-d status, matching teacher v1. The 8-d variant is a separate
comparison against the v2 teacher.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_norm=(5.7040, 0.1715, 0.4625, 0.3579, 0.0547, 5.7050),
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
                name=('stage2_STUDENT_A newstack (3-frame, grid8, accel '
                      'targets, future speed profile) vs v1 0.4218'))),
    ])
