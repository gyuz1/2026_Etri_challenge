"""Stage-2 VADLAW, ego_lcf OFF -- everything that has measured or argued
its way in so far, in one configuration.

The 2026-09-04 organizer Q&A allows a network's OWN vision-derived
inference as a planner input ("네트워크가 영상으로부터 직접 추론한 결과를
planner에 사용하는 것은 ... 허용됩니다"), which is the one remaining legal
route for ego-motion information to reach the planner after the 08-31
ruling closed the direct ego_lcf path. This config takes that route and
tries to make the estimate as close to the banned-but-known-good signal
as possible:

- aux_bev_motion_feedback: the head's own estimate is concatenated into
  ego_feats and actually reaches ego_fut_decoder, instead of only
  training the head against a loss. Verified by
  tools/sanity_check_aux_bev_motion_feedback.py:
  d(loss_plan_reg)/d(ego_lcf_feat) is still exactly 0, while
  d(loss_plan_reg)/d(aux_bev_motion_head) is nonzero.

- aux_bev_motion_idx=(0,1,2,3,4,7): the six MEANINGFUL motion channels of
  ego_lcf_feat -- velocity x/y, accel x/y, yaw rate, speed -- rather than
  the four the earlier runs used. ego_lcf_feat_idx=[0..7] is what scored
  0.2166 before the ban; its other two entries (ego_length, ego_width)
  are constants, so this reproduces that signal's full content, now
  estimated from vision instead of read from the dataset.

- aux_bev_motion_temporal: gives the estimator an explicit temporal
  contrast (regional-pooled BEV descriptor, current minus previous)
  instead of a single snapshot. Regional rather than global because a
  global mean is shift-invariant and ego motion IS a shift -- measured:
  a 10-cell shift moves the regional descriptor while leaving the global
  mean bit-identical.

- prism_posterior_lcf_idx: PRISM's posterior additionally conditions on
  ego status. Train-only (the posterior is discarded at inference; the
  prior never sees it), so it costs nothing at test time and only
  sharpens what the KL term pulls the vision-only prior toward.

- ego_fut_dec_hidden_dim=520 + the fb6 surgical init: keeps the
  ETRI-domain-adapted decoder transferred out of the ego_lcf-ON stage1
  (measured 0.618 -> 0.593 on its own), with the six new feedback input
  columns zero-padded so they start as an exact no-op.

Everything else is inherited from the bevmotion stack: PRISM S=2,
prev_bev_dropout, echo_cycle, 3-stage cascaded BEV refine, EMA, and
VAD's own col/dir/bound planning losses.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    # prev_bev_dropout existed to make the COLD-START case in-distribution,
    # back when evaluation replayed one frame per window and the scored
    # frame therefore had prev_bev=None. This run is evaluated and
    # submitted with a 2-frame window, so the scored frame always has
    # prev_bev -- dropping it half the time now trains for a condition
    # that never occurs at test, and worse, zeroes the temporal delta on
    # half the steps, which is exactly the signal aux_bev_motion_temporal
    # needs. Measured: with prev_bev the estimator hits 6.4% speed error,
    # without it 96% (worse than predicting the dataset mean), so the two
    # regimes are not a smooth trade -- they are different problems.
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
        ego_fut_dec_hidden_dim=520,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_surgical_fb6.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf BEST (vision-inferred ego motion fed to '
                      'planner, 6ch + temporal delta, surgical decoder, '
                      'prism_lcf_posterior)'))),
    ])
