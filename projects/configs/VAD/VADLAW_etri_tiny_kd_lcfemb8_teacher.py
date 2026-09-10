"""Scheme-A teacher v2 -- ego status as a learned 8-d embedding (hidden 64),
on the ego_lcf-ON KD stage1 lineage, echo_cycle off. Never submittable.

WHY 8 OUT AND NOT 64

v1 (VADLAW_etri_tiny_kd_lcfemb_teacher.py) emitted 64, making ego_feats
512+64=576. No stage 1 produces a 576-wide decoder, so all 64 columns were
zero-padded at merge time and had only stage 2's 12 epochs to become useful.
Measured, at epoch 11:

  ego_fut_decoder.0.weight, mean|w|      scene 512 cols   ego cols
    A teacher v1 (64 zero-padded)          0.021221        0.003123  (0.147x)
    B teacher   (8 zero-padded, ep12)      0.021245        0.006181  (0.29x)
    ego_lcf-ON stage 1, after 1 epoch      0.021887        0.023510  (1.07x)
    0.2166-era lineage, at stage-2 start   0.019118        0.029813  (1.56x)

Emitting 8 makes ego_feats 512+8=520, which is exactly what
VAD_etri_tiny_stage1_cached_kd_lcfon.py produces, so those columns arrive
already trained by 48 epochs of KD instead of starting at zero.

The one real argument for going wide was the transform's expressiveness --
a ReLU over 8 units is a coarse piecewise-linear map. That argument is
about the HIDDEN width, not the output width, so ego_lcf_embed_hidden=64
keeps it while ego_lcf_embed_dim=8 keeps 520. Nothing is given up.

Widening the output was redundancy rather than capacity anyway:
ego_lcf_feat is [vx, vy, ax, ay, yaw_rate, ego_length, ego_width, speed],
where ego_length/ego_width are per-vehicle constants and speed is
norm(vx, vy) -- about 5 independent varying numbers. A 64-d encoding of
them lies on a 5-d manifold in 64-d space.

The student (VADLAW_etri_tiny_kd_nolcf_split_distill8.py) must set
ego_status_est_dim=8 to match; its estimator already expands internally
(1024 -> 256 -> est_dim), so only the output width changes there.

Everything else follows VADLAW_etri_tiny_kd_lcfon_v2.py: same stage-1
lineage, echo_cycle off (the plan-vs-world-model-roundtrip trade is not
worth making for a teacher that reads real ego status), aux_bev_motion /
bev_refine / PRISM left on, ego_fut_dec_hidden_dim=512 so ego_scene_feats
and the student's stay comparable.
"""

_base_ = ['./VADLAW_etri_tiny_kd_lcfemb_teacher.py']

model = dict(
    echo_cycle_weight=0.0,
    pts_bbox_head=dict(
        ego_lcf_embed_dim=8,
        ego_lcf_embed_hidden=64,
    ))

load_from = (
    'work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon/'
    'stage2_init_merged.pth')

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_TEACHER_A_v2 (ego_lcf -> 8-d embedding, '
                      'hidden 64, ego_lcf-ON KD stage1, echo_cycle off)'))),
    ])
