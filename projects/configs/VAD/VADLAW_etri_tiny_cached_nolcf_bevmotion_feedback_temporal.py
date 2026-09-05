"""Stage-2 VADLAW, ego_lcf OFF: the vision-inferred ego-motion estimate is
both fed to the planner (aux_bev_motion_feedback) and given an explicit
temporal contrast to estimate from (aux_bev_motion_temporal).

The 2026-09-04 organizer Q&A allows a network's OWN vision-derived
inference as a planner input, so aux_bev_motion_head's prediction now
reaches ego_fut_decoder instead of only training against a loss (see
VADLAW_etri_tiny_cached_nolcf_bevmotion_feedback.py). That makes the
estimate's ACCURACY the thing that matters, and estimating speed from a
single BEV snapshot is intrinsically hard -- speed is a time derivative,
and the head only saw whatever the encoder's prev_bev cross-attention
happened to fold into the current frame.

aux_bev_motion_temporal hands it the contrast directly: a grid x grid
regional pool of the channel-reduced BEV for the current frame, plus the
difference against the same descriptor for the previous frame. Regional
rather than global because ego motion shifts BEV content spatially and a
global mean is nearly shift-invariant. Cold-start frames (every eval
window's first frame, plus prev_bev_dropout's training steps) zero-fill
the delta half, so single-frame estimation stays in-distribution.

Calibration, from BEV-Planner (CVPR 2024, arXiv:2312.03031) Table 1 on
nuScenes: VAD-Base scores 1.25m L2 without ego status vs 0.37m with
(3.4x), and the best ego-status-free entry in that table is BEV-Planner's
own 0.55m. So the ego-status-free regime is intrinsically far worse, and
recovering it hinges entirely on how good a vision-only motion estimate
can get.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_feedback=True,
        aux_bev_motion_temporal=True,
        aux_bev_motion_grid=4,
        aux_bev_motion_proj_dim=32,
        # Keeps the surgical decoder transfer, which measured 0.618 -> 0.593
        # on its own. The feedback concat widens ego_feats by
        # len(aux_bev_motion_idx)=4, so the donor's layer-0 weight is
        # padded with 4 zero input columns (zero = the feedback starts as
        # an exact no-op) rather than being discarded for a shape mismatch.
        ego_fut_dec_hidden_dim=520,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_surgical_fb4.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf (bevmotion stack, aux_bev_motion '
                      'FEEDBACK + TEMPORAL delta descriptor)'))),
    ])
