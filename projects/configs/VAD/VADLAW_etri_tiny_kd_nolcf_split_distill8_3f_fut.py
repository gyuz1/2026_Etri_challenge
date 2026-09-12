"""Scheme-A student v4: 3-frame descriptor + future ego speed profile
supervision on the BEV encoder. Compliant.

WHAT CHANGES VS ..._3f.py
aux_bev_future_motion=True. A second head on the same BEV motion descriptor
regresses the next 3 seconds of ego speed (6 steps at 0.5s), supervised by
gt_ego_long_fut_trajs.

WHY
Kinematic oracles bound what knowing the PRESENT can do
(tools/kinematic_oracle_ceiling.py):

  perfect velocity, accel = 0      0.5965
  perfect velocity AND accel       0.2708

and the leaderboard's 0.14 sits 48% below even the second. No accuracy about
the current state closes that; it requires reading the scene for what
happens next. aux_bev_motion asks the BEV encoder what the ego is doing now.
This asks what it is about to do.

The target is exact and costs nothing: ego_long_fut_trajs is already in
forward() for PRISM, is stored as per-step deltas, and speed is
|delta| / 0.5s. Considered and rejected: VLM-generated semantic labels
("red light ahead", "lead vehicle braking") for the same purpose. That
needs a 35h fine-tune to produce labels a teacher with a measured
memorization problem (0.1798 train vs 0.3511 held out) might get wrong, in
order to approximate something the GT states outright.

Supervising the BEV encoder rather than the planner is the point --
loss_plan_reg already trains the planner on this same GT. This pushes the
representation the planner reads FROM to carry maneuver cues.

Train-only: the head's output feeds no decoder, so T_infer is unchanged.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8_3f.py']

model = dict(
    pts_bbox_head=dict(
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
                name=('stage2_STUDENT_A_v4 (3-frame + future speed profile '
                      'supervision on BEV)'))),
    ])
