"""Stage-2 on the KD stage1 lineage, with privileged-expert distillation.

Built on the best compliant result to date (0.4885 on the 2-frame
hold-out: KD stage1 + bevmotion stack + prismlcf) and changing exactly
one thing -- privileged_distill.

Why this and not more estimation. can_bus already carries exact ego
speed into the BEV: zeroing its accel/rotation_rate/velocity slice takes
L2 from 0.593 to 10.4, with LANE_KEEP going 0.618 -> 11.73. So the
compliant decoder is not short of the information, it extracts it badly.
Trying to hand it a cleaner copy failed exactly as that framing predicts
-- aux_bev_motion_feedback fed the head's own 5.5%-error estimate in as
a decoder input and made things worse (0.5635 -> 0.6419), because a
noisy explicit channel displaced an accurate implicit one, and the
damage landed on LANE_KEEP (82% of frames, and the command where
distance-from-speed is nearly the whole task).

Distillation attacks extraction instead of information. A second head
sees the same vision-derived ego_feats plus the real ego status, is
trained on the same GT, and the deployed decoder is additionally pulled
toward that head's detached output. The expert is scoring what an
ego_lcf-fed planner scored before the ban (0.2166), so its trajectories
show the deployed decoder what good use of the BEV's motion content
looks like -- without ever handing it the number.

Compliance is the same shape PRISM already relies on here: ego_lcf
enters only the privileged head, that head runs only under
self.training, it is absent from the inference graph entirely, and the
distillation target is detached so no gradient reaches the deployed
decoder through the privileged input. Verified by
tools/sanity_check_aux_bev_motion_feedback.py's gradient proof, which
must still report d(loss_plan_reg)/d(ego_lcf_feat) == 0 exactly.

stage1 stays ego_lcf-OFF: the decoder trained there is the one that gets
deployed, so it should be free of any ego_lcf reliance to unlearn, and
the KD stage1's 512-wide decoder transfers with no surgery. The
privileged head is new here and has no stage1 counterpart.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion_prismlcf.py']

model = dict(
    pts_bbox_head=dict(
        privileged_distill=True,
        privileged_distill_idx=(0, 1, 2, 3, 4, 7),
        privileged_distill_weight=1.0,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_kd_nolcf DISTILL (privileged ego_lcf expert '
                      '-> compliant decoder)'))),
    ])
