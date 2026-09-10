"""Stage-1 VAD + Qwen-teacher KD, with ego_lcf back ON -- for the Scheme-B
(fused distillation) TEACHER lineage only. Diagnostic/non-compliant; never
feeds the compliant student directly.

WHY THIS EXISTS
VADLAW_etri_tiny_kd_lcfon_diag.py (the current Scheme-B teacher) starts stage
2 from stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfon8.pth --
a 512-wide, KD-trained-for-48-epochs ETRI decoder with 8 EXTRA columns
zero-padded on for ego_lcf, which then gets only stage 2's 12 epochs to learn
to use. Measured hold-out L2 came in at 0.2542m, short of the pre-ban
(2026-08-31 Q&A) record of 0.2166m on the same split. Comparing the two
lineages (see SHARED_CONTEXT.md's B teacher entry) turned up that the 0.2166
run's decoder was 520-wide (ego_lcf ON) from the very start of ITS stage 1,
so the whole planning head -- not just 12 epochs of it -- was ever trained
with ego status present. This config reproduces that property on the
current architecture instead of the 2026-08 one: same KD/aux_bev_motion/
col-dir-bound-loss/cascade-refine/EMA setup as
VAD_etri_tiny_stage1_cached_kd_nolcf.py, minus the one flag that config
turns off.

WHAT CHANGES VS THE NOLCF STAGE 1
Only ego_lcf_feat_idx: back to [0..7]. Every other addition in the nolcf
file (aux_bev_motion, loss_plan_col/dir/bound, bev_residual_refine cascade,
EMA) is KEPT, not stripped -- the point of this run is to isolate the
stage1-lineage variable (decoder trained with ego_lcf from epoch 1 of 48,
vs. zero-padded on at epoch 49), not to reproduce 2026-08's simpler
architecture. Those extras are redundant with having real ego_lcf (aux_
bev_motion in particular now regresses a signal the decoder can read
directly), but redundant supervision is not harmful here, and keeping the
extras held constant is what makes this an isolated comparison against the
already-running Scheme-B teacher rather than a second confound.

ego_fut_decoder is now built at embed_dims*2 + 8 = 520 wide from the first
stage-1 epoch (matching VADLAW_etri_tiny_kd_lcfon_diag.py's stage-2 width),
so KD (loss_plan_kd) trains an ETRI-domain, ego_lcf-aware planning head
directly instead of training a 512-wide one that then gets 8 fresh columns
grafted on. The stage2_init_merged.pth this run produces slots into
VADLAW_etri_tiny_kd_lcfon_diag.py's load_from with NO zero-pad surgery
needed (tools/surgical_ego_fut_decoder_transfer.py's whole purpose): the
widths already match.

COMPLIANCE
Unaffected: ego_lcf feeds the DECODER here, so this checkpoint is exactly as
non-compliant as the current Scheme-B teacher it's meant to replace, and is
subject to the same never-submit rule as VADLAW_etri_tiny_kd_lcfon_diag.py.
"""

_base_ = ['./VAD_etri_tiny_stage1_cached_kd_nolcf.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage1_lcfon (qwen_kd, ego_lcf ON from epoch 1, '
                      'aux_bev_motion, col_dir_bound_loss, '
                      'cascade_refine_x3, ema) -- Scheme-B teacher '
                      'lineage fix'))),
    ])
