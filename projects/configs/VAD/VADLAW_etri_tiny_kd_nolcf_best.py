"""Stage-2 VADLAW on the KD stage1 (ego_lcf OFF from stage1 onward), plus
the vision-inferred ego-motion techniques.

Measured on the 2-frame hold-out condition:

  plain bevmotion (decoder from nuScenes)          0.5967
  surgical decoder transfer from ego_lcf-ON stage1 0.5635
  KD stage1 (ego_lcf OFF end to end) + prismlcf    0.4885  <- this lineage

Training stage1 with ego_lcf_feat_idx=None from the start beat every
attempt to patch an ego_lcf-ON stage1's decoder into a compliant stage2:
its ego_fut_decoder is natively 512-wide AND ETRI-domain-adapted, so
nothing has to be dropped or substituted at merge time. This config keeps
that lineage and adds what the 0.4885 run did not have:

- aux_bev_motion_feedback with aux_bev_motion_idx=(0,1,2,3,4,7): the
  head's own vision-derived motion estimate reaches ego_fut_decoder,
  covering the six meaningful channels ego_lcf_feat carried before the
  ban (velocity x/y, accel x/y, yaw rate, speed). Allowed by the
  2026-09-04 Q&A -- a network's own value inferred from video may be a
  planner input.
- aux_bev_motion_temporal: the estimate is otherwise made from a single
  BEV snapshot, which measures 96% speed error at a cold start versus
  5.6% once prev_bev exists -- the information is entirely in the
  cross-frame comparison, so the head is given that contrast explicitly.
- prev_bev_dropout=0.0: submission is 2-frame, so the scored frame always
  has prev_bev. Dropping it half the time would train for a condition
  that never occurs and would zero the temporal delta on those steps.

ego_fut_dec_hidden_dim=512 matches the KD stage1 donor's hidden width
(it never had ego_lcf columns, so unlike the surgical lineage there is
nothing to slice); the six feedback channels widen only the input, and
the init checkpoint pads them with zeros so they start as a no-op.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion_prismlcf.py']

model = dict(
    prev_bev_dropout=0.0,
    pts_bbox_head=dict(
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_weight=0.5,
        aux_bev_motion_feedback=True,
        aux_bev_motion_temporal=True,
        aux_bev_motion_grid=4,
        aux_bev_motion_proj_dim=32,
        prism_posterior_lcf_idx=(0, 1, 2, 3, 4, 7),
        ego_fut_dec_hidden_dim=512,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_fb6.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_kd_nolcf BEST (KD stage1 lineage + vision '
                      'motion feedback 6ch + temporal delta)'))),
    ])
