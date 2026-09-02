"""Stage-2 VADLAW, ego_lcf OFF + the three train-only recovery techniques.

Builds on VADLAW_etri_tiny_cached_nolcf.py (ego status removed from the
planner's input per the organizers' 2026-08-31 ruling) and adds back, as
TRAINING signal only, what that removal costs:

1. aux_ego_motion -- regresses current ego status FROM the vision-derived
   planning features, supervised by ego_lcf_feat. A constant-velocity
   oracle study on this dataset's val split puts knowing-vs-not-knowing
   ego speed at 0.67m vs 5.51m L2, i.e. speed is by far the largest single
   thing the compliance fix removes. This makes the network compute it
   instead of being handed it. Explicitly the allowed pattern: ego status
   is a regression TARGET, never an input (see VAD_head.py's
   aux_ego_motion constructor comment for the full argument).

2. prev_bev_dropout -- eval and submission run one cold frame per clip
   (eval_holdout_l2.py resets the stream before every window), while
   training always supplies a populated prev_bev from the 3-frame queue.
   That mismatch was masked while ego_lcf_feat fed ego motion in directly;
   with it gone, a model that leans on prev_bev loses its only motion cue
   at test time. Dropping prev_bev on 50% of training steps puts the
   cold-start case in-distribution.

3. echo_cycle_weight -- Echo Planning (arXiv:2505.18945) cycle
   consistency, reusing the LAW world model's own weights for both
   directions. loss_rec already supplies the forward half; this adds the
   echo (future BEV rolled back to current) so the plan has to be
   consistent with how the scene actually moves.

All three are train-only and cost nothing at inference: inference never
calls bev_world_model, never receives ego_lcf_target, and never needs
aux_ego_motion_head. So T_infer and FLOPs are identical to the plain
_nolcf config, and eval/submission can keep using
VADLAW_etri_tiny_cached_eval_nolcf.py / _fast_eval_nolcf.py unchanged.

Deliberately NOT included: command_class_weights (measured worse -- U_TURN
went 0.737 -> 0.897 while overall L2 went 0.2166 -> 0.2179), and the
resolution increase, which would need a geometry-cache rebuild and a
stage-1 rerun and so is a separate decision.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf.py']

model = dict(
    # 0.5 -> half of training steps see the cold-start (prev_bev=None) path
    # the evaluator always uses, half keep the temporal signal the world
    # model needs to stay trainable. Both matter, hence an even split
    # rather than a small dropout probability.
    prev_bev_dropout=0.5,
    # Same order of magnitude as wm_loss_weight (0.2) but lower: the cycle
    # is a regularizer on top of the forward reconstruction, not a
    # replacement for it.
    echo_cycle_weight=0.1,
    pts_bbox_head=dict(
        aux_ego_motion=True,
        # ego_lcf_feat layout is (vx, vy, ax, ay, yaw_rate, length, width,
        # speed). Regress the four genuinely dynamic fields; length/width
        # are constants of the vehicle and ax/ay are much noisier than the
        # velocity terms, so they would add label noise without adding
        # information the planner needs.
        aux_ego_motion_idx=(0, 1, 4, 7),
        aux_ego_motion_weight=0.5,
        # 4. Regress the 5s future directly, not just through PRISM's
        #    64-d latent. Every sample in this split has a valid 10-step
        #    long future, so this is free supervision the model currently
        #    throws away. Weight below loss_plan_reg (1.0): the scored 3s
        #    horizon stays the primary objective and this is a regularizer
        #    on the features behind it.
        aux_long_horizon=True,
        aux_long_horizon_weight=0.5,
    ))
