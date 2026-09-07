"""DIAGNOSTIC-ONLY TEACHER, NOT SUBMITTABLE. ego_lcf feeds the decoder
directly AND target_point conditions generation via attention + residual
-- both explicitly prohibited by the organizer ruling.

Purpose: the strongest L2 this codebase can produce at all, used purely
as a distillation source (trajectory-level and/or feature-level) for the
compliant models. Not "revive the old shortcut code as it was" -- every
technique added since then stays on: PRISM S=2 (+ now conditioned on
ego_lcf too, since there's no compliance reason left to hold it back
here), prev_bev_dropout, echo_cycle, 3-stage cascaded BEV refine, EMA,
VAD's col/dir/bound losses, aux_bev_motion. Historical ceiling for this
lineage (git 11cb376, older codebase, 2Hz, none of the above techniques):
0.114m with target_point, collapsing to 5.45m without it -- expect this
run to beat 0.114m given everything added since, but note that 47x
collapse means the model is close to pure point-interpolation, not
vision-grounded planning with a hint.

ego_fut_dec_hidden_dim is left at its default (embed_dims*2 + 8 = 520,
matching ego_lcf_feat_idx=[0..7]) -- aux_bev_motion_feedback stays OFF so
nothing widens it further, which means stage1's own 520-wide
ego_fut_decoder (already ego_lcf-ON, no shape mismatch) transfers with a
PLAIN merge, no surgical slicing or padding needed at all.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    # This teacher is never itself evaluated/submitted -- only queried as
    # a distillation source, and always with prev_bev present (matching
    # how the compliant model's own training queue and the distillation
    # pipeline query it). The inherited 0.5 exists to make eval's cold
    # start in-distribution for a SUBMITTABLE model; spending half this
    # teacher's training on a condition it will never be asked to perform
    # in just dilutes training toward the one condition (prev_bev
    # present) that actually matters for its purpose. 0.0 here does not
    # affect the compliant models trained from this teacher's output --
    # each of those keeps its own prev_bev_dropout setting.
    prev_bev_dropout=0.0,
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        # Past ego trajectory (ego_his_encoder) was tried here too but hit
        # a real bug (ego_his_trajs arrives None at VAD_head.forward() on
        # at least one call path -- untraced, since this data is redundant
        # with ego_lcf_feat's already-fitted velocity/accel and so was
        # expected to add only marginal value anyway). Left off rather
        # than spend more time debugging a low-value addition and delaying
        # this 22h run further.
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        prism_posterior_lcf_idx=(0, 1, 2, 3, 4, 7),
        target_point_shortcut=True,
        target_point_shortcut_mode='both',
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_lcfon_teacher.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_TEACHER_NONCOMPLIANT (ego_lcf ON + TP '
                      'shortcut + full technique stack -- diagnostic '
                      'only, never submit)'))),
    ])
