"""Scheme-B teacher v2 -- ego_lcf ON, TP-free, on the ego_lcf-ON KD stage1
lineage, with echo_cycle off. Diagnostic only, never submittable.

TWO CHANGES VS VADLAW_etri_tiny_kd_lcfon_diag.py (which measured 0.2542m)

1. load_from -> the new stage1 (VAD_etri_tiny_stage1_cached_kd_lcfon.py).

   Measured motivation. The 2026-08 pre-ban record of 0.2166m came from a
   lineage whose ego_fut_decoder was 520-wide all the way from stage 1, and
   whose 8 ego_lcf columns entered stage 2 carrying real nuScenes-LAW
   weights. Comparing the tensors directly:

     nuScenes LAW pretrain : scene 512 cols mean|w| 0.0216, lcf 8 cols 0.0336
     old stage1 (0.2166)   : scene 512 cols mean|w| 0.0191, lcf 8 cols 0.0298
     kd_lcfon_diag (0.2542): lcf 8 cols exactly 0 (zero-padded at merge)

   The old stage1 never trained that decoder at all -- every planning loss
   was weight 0.0 and there was no KD, and the epoch_48 decoder is the
   nuScenes one scaled by 0.887 with cosine 1.0000 to it, i.e. pure weight
   decay. So its contribution was purely as an INITIALIZATION, and one where
   the ego status columns were not just nonzero but ~1.5x larger than the
   scene columns. kd_lcfon_diag instead starts those columns at zero and
   gives them only stage 2's 12 epochs to become useful.

   The new stage1 should beat both: KD (loss_plan_kd, the only signal that
   trains a stage-1 decoder here) now trains a 520-wide decoder on ETRI
   data with ego status present, rather than inheriting nuScenes weights or
   starting from zeros.

2. echo_cycle_weight 0.1 -> 0.0.

   Echo Planning's reverse pass (VAD_LAW.py's loss_echo_cycle) sends
   gradient into the planner through the selected waypoints, pulling the
   plan toward what bev_world_model can round-trip rather than toward GT.
   That trade is worth making for a COMPLIANT model, which has no ego
   motion cue and needs the consistency constraint as a substitute. This
   teacher reads real ego status, so the compensating benefit is gone while
   the pull away from GT remains. bev_world_model is a learned 2-layer
   deformable-attention approximation, not an oracle.

   Untested: no on/off ablation of echo_cycle exists on an ego_lcf-ON
   teacher, or on any model. This is a reasoned change, not a measured one.

DELIBERATELY UNCHANGED
- aux_bev_motion stays ON. It costs this teacher a little (it regresses a
  quantity the decoder can already read) but it pushes the teacher's VISION
  features to encode ego motion, which is what makes ego_plan_hidden
  reachable by a student that only has vision. For Scheme B specifically
  that is the whole transfer story, so removing it would optimize the
  teacher's headline number at the distillation target's expense. It is
  also small in stage 2 (loss_aux_bev_motion ~0.0675 against a total ~5).
- bev_refine_steps=3 stays. Every refine MLP's last layer is zero-init
  (VAD_head.py:824, 855), so the cascade starts as an exact no-op and can
  only learn corrections; and it runs AFTER ego_fut_decoder, so it is not
  even in ego_plan_hidden's forward path.
- prism_posterior_lcf_idx stays. For a TEACHER the prior already sees ego
  status (prism_prior_net reads ego_feats, VAD_head.py:1693), so the
  prior/posterior information gap this closes is small here -- unlike in a
  student, where it is large. prism_z_proj is zero-init too.
- ego_fut_dec_hidden_dim=512 stays, and must: it is what makes this
  teacher's ego_plan_hidden the same width as the student's.

No zero-pad surgery is needed for load_from this time --
tools/surgical_ego_fut_decoder_transfer.py existed to widen a 512-wide
donor to 520. The new stage1 is already 520.
"""

_base_ = ['./VADLAW_etri_tiny_kd_lcfon_diag.py']

model = dict(
    echo_cycle_weight=0.0,
    pts_bbox_head=dict(
        # Was an accidental confound in the v1 A-vs-B comparison: A teacher
        # v1 (0.2328m) set this to 'cumulative' while B teacher v1
        # (0.2542m) stayed at the 'position' default, so that gap was never
        # a clean read on the embedding scheme. 'cumulative' reweights each
        # timestep's regression loss by delta sensitivity -- ego_fut_preds
        # are per-step deltas the metric cumsums, so an early step's error
        # displaces every later position too, which 'position' (using only
        # that step's own GT position weight) does not account for. Setting
        # it here too so a v2-vs-v2 comparison isolates the embedding
        # scheme (raw 8 columns here vs an 8-d learned embedding in
        # VADLAW_etri_tiny_kd_lcfemb8_teacher.py) instead of mixing it with
        # this independent, unrelated improvement.
        plan_reg_ts_weight_mode='cumulative',
    ),
)

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
                name=('stage2_TEACHER_B_v2 (ego_lcf-ON KD stage1 lineage, '
                      'echo_cycle off, TP-free)'))),
    ])
