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

6. remove_auxiliary_planning_losses=False -- re-enables VAD's original
   map-boundary / collision / lane-direction planning losses
   (arXiv:2303.12077, reference [4]), previously computed by the head but
   unconditionally discarded by VAD_LAW (VAD_LAW.py:470's own comment
   invites exactly this A/B). loss_plan_col and loss_plan_dir score the
   model's OWN detection/map predictions against its OWN planned
   trajectory -- in-network, vision-grounded, zero external information,
   train-only. loss_plan_bound also gets a real weight here for the first
   time, now that its lane_bound_cls_idx bug (was reading the 'divider'
   class instead of 'boundary' -- see VADLAW_etri_tiny.py's fix comment)
   is corrected.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf.py']

model = dict(
    remove_auxiliary_planning_losses=False,
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
        # Real weight now that lane_bound_cls_idx correctly points at
        # 'boundary'. Matches VAD_head.py's own constructor default (0.1)
        # rather than re-deriving a new number with no local evidence
        # behind it.
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1,
                             dis_thresh=1.0, lane_bound_cls_idx=2,
                             point_cloud_range=[-30.0, -15.0, -2.0,
                                                 30.0, 15.0, 2.0]),
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
        # 5. Cascaded trajectory refinement. ThinkTwice refines repeatedly;
        #    we were doing one pass. Each extra stage re-samples the BEV at
        #    the positions the previous stage produced, so the features it
        #    reads are the ones actually under the corrected waypoints.
        #    Every stage is zero-init, so 3 stages start as an exact no-op
        #    and can only diverge from the 1-stage model if training finds
        #    the extra passes useful. FLOPs are not a constraint here: the
        #    measured model sits at 479 of the 7053 GFLOPs cutoff, and this
        #    adds two grid_sample + small-MLP passes on 7*6 points.
        bev_refine_steps=3,
    ))

# Weight EMA. A decayed running average of the weights is a standard,
# near-free win for regression heads -- it damps the step-to-step noise
# that batch=2 (samples_per_gpu=1 x 2 GPUs) makes unavoidable, which is
# exactly the regime where the averaged weights tend to beat the final
# ones. momentum=0.0002 gives roughly a 5000-step window, a few thousand
# steps short of one 8579-step epoch, so the average tracks the current
# solution rather than dragging in early-training weights.
#
# EMAHook.after_train_epoch swaps the averaged weights into the model and
# before_train_epoch swaps them back, so epoch_N.pth holds the EMA weights
# and eval needs no change.
#
# priority='HIGH' (30) is load-bearing, not decoration. CheckpointHook is
# registered at NORMAL (50) by register_training_hooks, and custom_hooks are
# registered afterwards (mmdet_train.py:139 vs :190); mmcv's register_hook
# inserts an equal-priority hook AFTER the existing one, so an EMAHook left
# at NORMAL would run its after_train_epoch AFTER the checkpoint was already
# written -- saving raw weights every epoch and making EMA a silent no-op.
# HIGH puts it ahead of CheckpointHook so the swap happens first.
#
# CustomSetEpochInfoHook is listed alongside it because mmcv config merge
# REPLACES list-typed fields wholesale, not appends -- overriding
# custom_hooks with just [EMAHook] would silently drop the base's
# CustomSetEpochInfoHook. Currently inert either way (the only reader of
# the epoch it sets, use_traj_lr_warmup, is False here), caught while
# fixing the same pattern in VAD_etri_tiny_stage1_cached_kd_nolcf.py --
# listing both is the correct fix, not just relying on that staying inert.
custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(type='EMAHook', momentum=0.0002, priority='HIGH'),
]
