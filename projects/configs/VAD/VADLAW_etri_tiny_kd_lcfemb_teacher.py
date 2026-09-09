"""Scheme-A teacher: ego_lcf enters ego_feats as a LEARNED 64-d embedding
instead of raw appended columns. Diagnostic only, never submittable.

Why this and not the raw-column teacher (VADLAW_etri_tiny_kd_lcfon_diag.py,
running on the other machine): that one appends ego_lcf verbatim as 8
columns AFTER both ego cross-attentions, so its decoder can read speed
straight out of those columns and nothing pressures its 2*D scene half to
encode motion at all. Distilling that half teaches a student the same
indifference. Here the ego status is first mapped through
Linear(8->64)->ReLU->Linear(64->64), trained end to end by the planning
loss, so the 64-d block is a REPRESENTATION of ego status -- something a
vision-derived estimate can be aligned against block-for-block.

That alignment is the point of scheme A: the student gets the same 576-wide
ego_feats layout, its 64-d slot filled by an estimate inferred from vision,
and two separate distillation losses (512 scene block, 64 ego block) rather
than one over the concatenation, where the 512 would carry 89% of the
gradient and the ego block would barely train.

Known risk, and why the student side will add modality dropout: this repo
already measured what happens when a vision-estimated ego value is fed to
the decoder as an input channel -- aux_bev_motion_feedback went 0.5635 ->
0.6419, with the damage concentrated in LANE_KEEP (0.582 -> 0.690), which
carries ~85% of total L2 error. The diagnosis was that a ~5.5%-error
explicit channel displaced the more accurate implicit signal can_bus
already supplies. Modality dropout is the standard countermeasure (it
"precludes over-reliance on the easiest modality"), and this codebase
already uses the same pattern for prev_bev_dropout.

The teacher itself carries none of that risk: its 64-d block is computed
from real ego status, not an estimate.

ego_fut_dec_hidden_dim=512 keeps the hidden width equal to the student's,
so hidden-layer activations stay comparable if we also want to match there.
load_from pads the KD stage1 donor's 512-wide first Linear with 64 zero
columns (it was trained without any ego input), so the embedding starts
contributing exactly nothing and learns from there.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion_prismlcf.py']

model = dict(
    prev_bev_dropout=0.0,
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        ego_lcf_embed_dim=64,
        ego_fut_dec_hidden_dim=512,
        plan_reg_ts_weight_mode='cumulative',
    ))

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
                name=('stage2_TEACHER_A (ego_lcf as learned 64-d embedding, '
                      'TP-free, KD stage1)'))),
    ])
