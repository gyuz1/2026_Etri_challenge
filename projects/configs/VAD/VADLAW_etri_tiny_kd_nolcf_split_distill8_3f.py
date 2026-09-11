"""Scheme-A student v3: 3-frame motion descriptor, so acceleration is
actually observable. Compliant.

WHAT CHANGES VS ...split_distill8.py
aux_bev_motion_frames 2 -> 3. The descriptor becomes
cat([current, d1, d1 - d2]) with d1 = cur - prev1 and d2 = prev1 - prev2,
so the third block is the second difference -- which is what an
acceleration is. Verified in closed form against constant-acceleration
motion: d1 - d2 == a * dt^2 exactly, and 0 under constant velocity.

WHY
Kinematic oracles on this val split, scored with the competition's own L2
windowing (tools/kinematic_oracle_ceiling.py):

  stay put                       12.5175
  perfect velocity, accel = 0     0.5965
  perfect velocity AND accel      0.2708

Acceleration is the larger half of what ego state can buy, and on LANE_KEEP
(85% of our error) it is 0.5543 -> 0.2524. The v2 student already regresses
ax/ay, but with only two frames there is no observable to regress them
FROM: two positions determine a velocity and nothing more. This config is
what makes that supervision answerable.

THE LATENCY TRADE
T_infer is model-forward-only on a 4090, and the score is
L2 x (1 + max(0, T - 100) / 200). With --bev-only-history a history frame
costs 27.7ms against a scored frame's 61.9ms (measured, etri_test_submit.py):

  2 frames   89.6ms   no penalty
  3 frames  117.3ms   x1.087

So 3 frames needs only an 8% L2 improvement to break even, against a lever
measured at 55% in the oracle. Note this assumes --bev-only-history is
actually passed at eval and submission time; without it two frames alone
already cost 135ms and x1.176.

Everything else is inherited: 8-d status slot, split distillation, modality
dropout 0.3, normalized aux_bev_motion over (0,1,2,3,4,7).
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8.py']

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
                name=('stage2_STUDENT_A_v3 (3-frame motion descriptor, '
                      'acceleration observable, 8-d status, split '
                      'distill)'))),
    ])
