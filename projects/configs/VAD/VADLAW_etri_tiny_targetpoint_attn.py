"""Stage-2 VADLAW, target_point shortcut fix: full retrain from stage1's
merged init (NOT a short fine-tune off the shortcut-laden stage2 checkpoint).

Background: ablation (tools/eval_holdout_l2_shortcut_ablation.py, 2026-08-25)
on the existing 301/75-split stage2 checkpoint found removing ego_target_point
raised hold-out L2 47.75x while removing images raised it only 1.06x -- the
model was navigating almost entirely off the 5s goal point. The original
suspect fix (VADLAW_etri_tiny_cached_target_point_dropout.py: binary dropout,
short fine-tune from the converged checkpoint) was judged likely insufficient
on two counts:
  1. The zero-init advantage of target_point_encoder's residual is already
     spent on that checkpoint (it converged to a large, non-zero shortcut
     weight); a short low-lr fine-tune is unlikely to unlearn it.
  2. Binary dropout alone risks the model learning two disjoint policies
     keyed off "is target_point exactly zero", rather than ever blending
     vision and target_point when both are present (see VAD_head.py's
     target_point_mode docstring for the literature this echoes: TransFuser
     v6, LEAD (CVPR 2026), "Hidden Biases of E2E Driving Models" (ICCV 2023)).

This config instead combines two changes, both in VAD_head.py, both
retrained from scratch on top of stage1's merged init:
  - model.pts_bbox_head.target_point_mode='both': target_point now also
    conditions ego_agent_decoder/ego_map_decoder's query_pos (previously a
    hardcoded, information-free torch.zeros) in addition to the existing
    zero-init residual -- so target_point can only steer the plan by
    reweighting attention over vision-derived agent_query/map_query, not by
    a vision-independent bypass alone.
  - target_point_dropout lowered to 0.1 (kept small, as a residual
    robustness case for genuinely missing input) and target_point_noise_std
    added at 0.2 (~20% of the goal distance, applied every non-dropped
    training sample) so "target_point present" no longer means "target_point
    exactly trustworthy" -- there's no clean present/absent branch to
    overfit two separate policies around.
  - bev_residual_refine=True: ThinkTwice-lite (OpenDriveLab/ThinkTwice,
    CVPR 2023 idea, not ported -- see VAD_head.py's refine_ego_trajs_with_bev
    docstring). Samples this frame's own bev_embed at each coarse waypoint's
    predicted (x, y) location via grid_sample and predicts a correction from
    it. Stricter than target_point_mode='attn' alone: attention conditioning
    ties the plan to one aggregated vision vector, this ties each waypoint's
    correction to vision content specifically at that waypoint's own
    location, so a plan that ignores vision can't produce a spatially
    coherent correction regardless of how it's trained.

Deliberately NOT using the *_cached.py geometry-cache pipeline: this
experiment folder (A5000 LAW_split) has no geometry cache built for the
301-scene split, and this is a validation run, not the final efficient
training pass -- raw image loading is slower per-iter but requires no cache
build. Port back to the cached pipeline on the original/3090 machine once
this is confirmed to actually reduce the 47x collapse.

STATUS: not yet launched. After training, re-run
eval_holdout_l2_shortcut_ablation.py against the resulting checkpoint and
compare the target_point-zeroed / images-zeroed multipliers against the
47.75x / 1.06x baseline before trusting this.
"""

_base_ = ['./VADLAW_etri_tiny.py']

# Explicit override (base VADLAW_etri_tiny.py's load_from points at the
# same file under its original, cadence-ambiguous name) -- named _2hz here
# so it can't be confused with a future 10hz-stage1-derived merge init once
# fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata_10hz.py finishes and
# gets merged. Same underlying checkpoint, just an unambiguous name for
# this experiment: work_dirs/stage1_etri_v2/stage2_init_merged_2hz.pth is a
# symlink to stage2_init_merged.pth on the original/3090 machine, and a
# plain renamed copy on A5000 LAW_split.
load_from = 'work_dirs/stage1_etri_v2/stage2_init_merged_2hz.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                               name='stage2_targetpoint_attn_split_301_75')),
    ])

model = dict(
    pts_bbox_head=dict(
        target_point_mode='both',
        target_point_dropout=0.1,
        target_point_noise_std=0.2,
        bev_residual_refine=True,
    ))

# The target_point_encoder-skip DDP crash (fixed in VAD_head.py: the module
# is now always called) was one confirmed source of unused-parameter
# desync, but VADLAW_etri_tiny_cached.py separately documents that the LAW
# world-model branch's autograd graph hasn't been verified safe with this
# off either -- inheriting from the non-cached base here means that
# safeguard wasn't inherited, so set it explicitly rather than risk a repeat
# crash from a different source of unused params.
find_unused_parameters = True
