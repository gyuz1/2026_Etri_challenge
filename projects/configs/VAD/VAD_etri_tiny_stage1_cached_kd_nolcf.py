"""Stage-1 VAD + Qwen-teacher distillation, with ego_lcf OFF so the planning
decoder KD trains is actually transferable to the compliant stage 2.

Why both changes have to happen together:

* Stage 1's loss_plan_reg/bound/col/dir are all weight 0.0, so ego_fut_decoder
  normally receives no gradient at all across its 48 epochs -- it reaches
  stage 2 exactly as randomly initialized as it started. KD (loss_plan_kd,
  weight 0.2) is the only thing that changes this: it is the sole signal
  that trains the planning decoder during stage 1.

* But with the inherited ego_lcf_feat_idx=[0..7], that decoder is built at
  embed_dims*2 + 8 = 520 wide, while the compliant stage-2 head
  (VADLAW_etri_tiny_cached_nolcf.py, ego_lcf_feat_idx=None) is 512. A 520-wide
  decoder cannot load into a 512-wide one, so everything KD taught it would be
  dropped at merge time -- which is exactly what happens today: the merge
  falls back to taking ego_fut_decoder from the nuScenes LAW checkpoint
  (--override-prefixes-from-world-model), i.e. our compliant stage 2 currently
  warm-starts its planner from another domain rather than from ETRI data.

  Turning ego_lcf off here makes stage 1's decoder 512-wide, so a
  KD-trained, ETRI-domain-adapted planning head transfers into stage 2
  directly and that override stops being needed.

Note this changes nothing about stage 1's perception weights either way:
ego_lcf_feat is only ever read at the ego_feats concat feeding
ego_fut_decoder (VAD_head.py:1171-1191), and with planning losses at 0.0 no
gradient flowed back through it -- so the existing epoch_48.pth's backbone /
BEV encoder / detection / map / motion weights were never shaped by ego_lcf
being on. Only the (untrained) planner's width was.

aux_bev_motion is also enabled here rather than only in stage 2: it
regresses ego status from bev_embed, and stage 1 is where the BEV encoder
actually gets trained (48 epochs, versus 12 in stage 2), so this is where
pushing that encoder to represent ego motion has the most epochs to act.
Same compliance shape as everywhere else -- ego status is a regression
TARGET on a branch with no path into any decoder output.

loss_plan_col/dir/bound also get real weights here for the same reason KD
does: they were left at 0.0 in the shared base because nothing trained
ego_fut_decoder anyway, so a nonzero weight would have been inert. That
premise no longer holds once KD is live -- these three losses (VAD's
original map-boundary / collision / lane-direction terms, arXiv:2303.12077)
give the same additional geometric signal added to stage 2
(VADLAW_etri_tiny_cached_nolcf_aux.py), and stage 1 gets 48 epochs to use
it against 12 in stage 2. lane_bound_cls_idx's class-index bug is also
fixed as of this file (VAD_etri_tiny_stage1.py's own copy, separate from
VADLAW_etri_tiny.py's).

bev_residual_refine (3-stage cascade) and EMA are also on here, matching
stage 2's VADLAW_etri_tiny_cached_nolcf_aux.py. Both are architecture-
agnostic (refine_ego_trajs_with_bev and EMAHook are plain VAD_head.py /
mmcv mechanisms, not VADLAW-specific), the [N,B,D] bev_embed axis bug
(VAD_head.py's refine_ego_trajs_with_bev) is already fixed in shared code,
and 48 epochs gives the cascade far more time to move away from its
zero-init no-op than stage 2's 12.

custom_hooks is overridden here rather than only adding EMAHook to the
base's list: mmcv config merge REPLACES list-typed fields wholesale, not
appends, so the naive `custom_hooks = [EMAHook]` used for the first version
of this config (and still stage 2's) silently dropped the base's
CustomSetEpochInfoHook. Harmless in practice -- the only place that hook's
epoch value is read is use_traj_lr_warmup, which is False in every config
here -- but listing both explicitly is the actually-correct fix rather
than relying on that being inert forever.
"""

_base_ = ['./VAD_etri_tiny_stage1_cached_kd.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=None,
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 4, 7),
        aux_bev_motion_weight=0.5,
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0,
                           x_dis_thresh=3.0, y_dis_thresh=1.5,
                           point_cloud_range=[-30.0, -15.0, -2.0,
                                               30.0, 15.0, 2.0]),
        loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=0.5,
                           point_cloud_range=[-30.0, -15.0, -2.0,
                                               30.0, 15.0, 2.0]),
        loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1,
                             dis_thresh=1.0, lane_bound_cls_idx=2,
                             point_cloud_range=[-30.0, -15.0, -2.0,
                                                 30.0, 15.0, 2.0]),
        bev_residual_refine=True,
        bev_refine_steps=3,
    ))

custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(type='EMAHook', momentum=0.0002, priority='HIGH'),
]

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage1_nolcf (qwen_kd, aux_bev_motion, '
                      'col_dir_bound_loss, cascade_refine_x3, ema)'))),
    ])
