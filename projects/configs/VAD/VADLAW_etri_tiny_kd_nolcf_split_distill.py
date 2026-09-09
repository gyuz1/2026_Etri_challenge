"""Scheme-A student: split distillation from the Scheme-A teacher
(VADLAW_etri_tiny_kd_lcfemb_teacher.py). Compliant -- submittable.

THE LAYOUT
Teacher : ego_feats = cat([agent 256, map 256, ego_lcf_embed_net(8) 64]) = 576
Student : ego_feats = cat([agent 256, map 256, ego_status_est_net(bev) 64]) = 576

Same width, same slot meanings, so the two halves can be aligned by two
independent losses instead of one over the concatenation. In the fused
alternative (scheme B, running on the other machine) the 512 carries ~89%
of a single cosine's gradient and the ego block barely trains; here the
status term gets its own weight.

WHERE THE STUDENT'S 64 COMES FROM
The BEV motion descriptor that aux_bev_motion already regresses
(vx, vy, yaw_rate, speed) from -- cat([current, current - previous]) over
regionally-pooled BEV, which is why aux_bev_motion_temporal is mandatory
here. That head's loss is the standing evidence this descriptor carries
real ego motion. Compliance rests on the 2026-09-04 organizer Q&A: the
network's own vision-inferred ego status may feed the planner; raw or
gated privileged data may not. bev_embed and prev_bev are the only inputs,
so the estimate exists identically at test time.

WHY THIS IS NOT aux_bev_motion_feedback AGAIN
That experiment fed bev_pred's 4 scalars into the decoder and measured
0.5635 -> 0.6419, the damage concentrated in LANE_KEEP. Two differences:

1. Supervision. Those 4 scalars were trained only by an L1 against real
   ego_lcf, in no particular relation to what the decoder wanted from
   them. This 64-d vector is trained to reproduce the vector the
   TEACHER'S decoder actually consumed to plan well -- the target is the
   representation, not the physical quantity.
2. Modality dropout (0.3). The failure mode was a ~5.5%-error explicit
   channel displacing the more accurate implicit signal can_bus already
   supplies. Zeroing the slot on 30% of training steps keeps the decoder
   able to plan from the 512 scene half alone, so the estimate can only
   add. This is the standard countermeasure for exactly this dynamic, and
   the repo already uses the same pattern for prev_bev_dropout.

The dropout is applied AFTER the estimate is captured for the loss, so
loss_status_distill still trains the estimator on dropped steps; only the
decoder's view is blanked.

WEIGHTS
scene 0.3 mirrors scheme B's fused weight, which was set so the cosine
term (which never decays -- the student can never fully match a teacher
that saw real ego status) does not dominate loss_plan_reg once that
reaches ~0.006. status 0.5 is higher because loss_status_distill is
nearly the only supervision ego_status_est_net gets: its other path, the
planning loss through the decoder, is switched off on 30% of steps by the
modality dropout. An undertrained estimate sitting in the decoder's input
is precisely the aux_bev_motion_feedback failure.

load_from reuses the Scheme-A teacher's donor checkpoint: both models have
a 576->512 first decoder Linear, and the donor (KD stage1, no ego input)
was padded with 64 zero columns for exactly that slot. The extra nets
(ego_lcf_embed_net there, ego_status_est_net here) are in neither donor
and start fresh, so the status slot begins contributing nothing.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_distill.py']

model = dict(
    pts_bbox_head=dict(
        privileged_distill=False,
        aux_bev_motion_temporal=True,
        ego_status_est_dim=64,
        ego_status_est_dropout=0.3,
        plan_reg_ts_weight_mode='cumulative',
        # MUST be set. It defaults to ego_fut_dec_in_dim, which the 64-d
        # status slot just widened to 576 -- and then every ego_fut_decoder
        # weight in the donor (512-wide) mismatches. mmcv loads with
        # strict=False, so that would not raise: the decoder would silently
        # reinitialize and 22h of training would start from a random
        # planner. Verified by instantiating this config against the donor.
        ego_fut_dec_hidden_dim=512,
    ),
    feature_distill_teacher_cfg=(
        'projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb_teacher.py'),
    feature_distill_teacher_ckpt=(
        'work_dirs/stage2_kd_lcfemb_teacher/epoch_12.pth'),
    feature_distill_mode='split',
    scene_distill_weight=0.3,
    status_distill_weight=0.5,
)

load_from = (
    'work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/'
    'stage2_init_merged_lcfemb64.pth')

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_STUDENT_A SPLIT_DISTILL (scene 512 + vision-'
                      'estimated status 64 <- Scheme-A teacher, w=0.3/0.5, '
                      'modality dropout 0.3)'))),
    ])
