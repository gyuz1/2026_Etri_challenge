# 2026 ETRI Autonomous Driving AI Challenge — E2E Driving (VAD + LAW)

An end-to-end autonomous driving model that jointly learns 3D perception
(objects, lane geometry) and ego trajectory planning from six camera views.
VAD-tiny is trained in two stages on ETRI's domain, with a LAW (latent
world model) stage refining the planning latent.

## Architecture

**Stage 1 (VAD-only).** A six-camera BEV encoder plus 3D detection, map,
and other-agent trajectory heads, domain-adapted to ETRI data.
`projects/mmdet3d_plugin/VAD/VAD.py`, `VAD_head.py`.

**Stage 2 (VADLAW).** Stage 1's perception weights are merged
(`tools/merge_stage1_world_model.py`) with a nuScenes-pretrained LAW world
model, then fine-tuned with ego-agent/ego-map cross-attention and a
world-model latent-prediction loss. `projects/mmdet3d_plugin/LAW/VAD_LAW.py`.

Both stages regress the ego future trajectory as a multi-mode structure
(`ego_fut_mode` candidates × `fut_ts` steps × (x, y)), selecting the mode
that matches the driving-intent command.

## Compliance: ego status is a training target, never a planner input

The organizers' Q&A draws a specific line: ego status (current or past)
feeding the planner **directly or as a simple embedding** is prohibited,
because it lets vision-independent extrapolation dominate the output —
regardless of whether vision is also present. **Indirect use that improves
features shared across tasks stays allowed.** The distributed baseline
ships with `ego_lcf_feat_idx=None` in every one of its configs; ours
matches that (`ego_lcf_feat_idx=None`, `use_ego_lcf_status=False`
everywhere).

This is not just a config default — it is proven at the gradient level.
`tools/sanity_check_nolcf_aux.py` makes ego status a differentiable leaf
tensor and backprops the planning loss through it directly:
`d(loss_plan_reg)/d(ego_status)` comes out **exactly 0.0**, while a control
backprop through the auxiliary losses below gives a nonzero gradient —
proving the zero reflects a real absence of any path to the trajectory,
not a disconnected graph.

Removing this input has a real, measured cost: a constant-velocity oracle
on this dataset needs the true current speed to hit 0.67 m L2; without any
speed signal at all it is 5.51 m. The techniques below exist specifically
to recover as much of that gap as compliance allows, by having the
network infer motion from vision instead of being handed it.

### Recovering the removed information, indirectly

- **`aux_bev_motion`** — a small head reads `bev_embed` (the BEV encoder's
  raw output, the one tensor every downstream head branches from) and
  regresses current ego status (velocity, yaw rate, speed) against
  `ego_lcf_feat` as an L1 loss. `ego_lcf_feat` only ever appears as a
  regression **target** here, on a branch with no path back into any
  decoder output — the compliance proof above covers exactly this case.
  Placed at `bev_embed` rather than a later planning-specific feature so
  the pressure reaches the shared encoder, matching the organizers'
  "improves common features across multiple tasks" allowance more
  directly than a late-stage auxiliary head would.
- **`prev_bev_dropout`** — evaluation and submission both run a single
  cold frame per clip (`reset_stream()` before every window), while
  training always has a populated `prev_bev` from the temporal queue. That
  mismatch was invisible while `ego_lcf_feat` fed motion in directly; once
  it is gone, a model that leans on `prev_bev` loses its only motion cue
  exactly where the evaluator tests it. Dropping `prev_bev` on half of
  training steps makes the cold-start case in-distribution instead of an
  unseen one.
- **Echo-planning cycle consistency** (`echo_cycle_weight`, following
  [Echo Planning, arXiv:2505.18945][10]) — the LAW world model already
  supplies the forward half of a cycle (`loss_rec`: previous BEV + planned
  waypoints → current BEV). This adds the echo half: roll the current BEV
  forward along the plan to a predicted future BEV, then roll that back
  with the negated plan and require it to land on the current BEV again,
  reusing the same world-model weights for both directions so the
  constraint is on the *plan* rather than two independently-fittable
  predictors.
- **Cascaded BEV-residual refinement** (`bev_residual_refine`,
  `bev_refine_steps`) — a ThinkTwice-lite ([arXiv:2305.06242][6]) module
  samples the frame's own BEV feature at each coarse waypoint's (x, y) and
  predicts a correction; cascading it (each stage re-samples at the
  previous stage's output) matches the paper's actual multi-pass recipe
  rather than our earlier single pass. Every stage is zero-init, so the
  cascade starts as an exact no-op and only departs from it if training
  finds the extra passes useful.
- **VAD's original planning losses, re-enabled** — `loss_plan_col`
  (collision avoidance) and `loss_plan_dir` (lane-direction alignment)
  score the model's *own* detection/map predictions against its *own*
  planned trajectory: in-network, vision-grounded, zero external
  information. They were computed but unconditionally discarded by
  `remove_auxiliary_planning_losses=True`; re-enabling them gives the
  planner the same geometric supervision the original VAD paper used.
  `loss_plan_bound` (map-boundary) is included too, after fixing a real
  bug in its class-index (`lane_bound_cls_idx` pointed at the *divider*
  class instead of *boundary* — see the class docstring in
  `plan_loss.py` and the config comment in `VADLAW_etri_tiny.py`).
- **Weight EMA** — a decayed running average damps the step-to-step noise
  an effective batch size of 2 makes unavoidable.

Two auxiliary heads were tried and dropped, for stated, evidence-based
reasons rather than by default: `aux_ego_motion` (predicting ego status
from `ego_feats`, a late planning-specific feature) is superseded by
`aux_bev_motion`'s earlier, more shared placement. `aux_long_horizon`
(directly regressing the 5s privileged future) was suspected of competing
with PRISM's posterior over the same `ego_feats` tensor; an isolated A/B
run showed no evidence it caused the instability it was suspected of
(comparable grad-norm-NaN rate with it removed), so its omission is an open question tracked in code, not a settled result.

`command_class_weights` (sqrt-inverse-frequency per-command loss
weighting, to counter LANE_KEEP's ~80% share of the data) was implemented
and measured, then dropped: it made the very class it targeted *worse*
(U_TURN hold-out L2 0.737 m → 0.897 m), most likely because a 23×
multiplier on only 134 training samples amplifies that class's
idiosyncrasies rather than teaching anything general.

### Two real bugs found and fixed along the way

- **`ego_fut_decoder` was silently trained from scratch.** Stage 1 was
  kept at `ego_lcf_feat_idx=[0..7]` (its own domain-adaptation training is
  unaffected by ego status either way, since stage 1's planning losses are
  all weight-0.0 and never route gradient through `ego_lcf_feat`), so its
  planning decoder is shaped 520-wide — incompatible with the 512-wide
  compliant stage-2 head. `merge_stage1_world_model.py`'s plain union left
  that shape mismatch for `mmcv`'s loader to silently skip, meaning the
  decoder trained from random init every time, undetected until the
  merged checkpoint was loaded outside the normal launch path and the
  "size mismatch" log lines were actually read. Fixed with a new
  `--override-prefixes-from-world-model` flag that sources
  `ego_fut_decoder`'s non-final layers (which *do* match at 512-wide) from
  a same-shape nuScenes LAW checkpoint instead.
- **`bev_embed` axis assumption.** The BEV-refine sampling code assumed
  `bev_embed` was `[B, N, D]`; it is actually `[N, B, D]`
  (sequence-first — see `VAD_transformer.py`'s own permute before the
  decoder call). With batch size 1 the resulting reshape produced a
  correctly-shaped tensor holding the *wrong* permutation of the same
  numbers, so it never crashed — only surfaced when `aux_bev_motion`'s
  batch-first pooling on the same tensor threw a real shape error. Fixed
  in `refine_ego_trajs_with_bev`, and the fix is verified structurally
  (a positive gradient check) rather than only re-reading the code.

## Optional: Qwen-VL teacher distillation (KD)

A Qwen2.5-VL-3B model, fine-tuned on ETRI trajectory-prediction text
tasks, serves as a teacher. Its predictions are cached offline over the
training set (`evodrive_etri_prep/generate_teacher_cache.py`), then a
challenge-metric-aligned L2 distillation loss (`loss_plan_kd`,
`projects/mmdet3d_plugin/EvoKD/vad_head_kd.py`) supplements student
training. The teacher never enters the forward graph, so inference FLOPs
and latency are identical to plain VADLAW.

KD's biggest use is in **stage 1**: with `loss_plan_reg` at weight 0.0
there, `ego_fut_decoder` normally receives no gradient across all 48
epochs — KD is the only signal that trains it. Combined with turning
`ego_lcf_feat_idx` off in stage 1 too (so the resulting decoder is
512-wide), this produces an ETRI-domain-adapted, KD-trained planning head
that transfers directly into stage 2, instead of stage 2 having to
warm-start its planner from an out-of-domain nuScenes checkpoint.

## Repository layout

```
projects/
├── configs/VAD/                 # stage-1 / stage-2 / eval / KD configs
├── mmdet3d_plugin/
│   ├── VAD/                     # VAD.py, VAD_head.py (perception + planning head)
│   ├── LAW/                     # VAD_LAW.py (world-model stage 2)
│   ├── EvoKD/                   # Qwen teacher distillation glue
│   └── datasets/                # ETRI dataset, pipeline
tools/
├── data_converter/                          # parquet -> pkl (2Hz / 10Hz variants)
├── cache/etri_geometry_cache.py             # undistort/crop image cache
├── merge_stage1_world_model.py              # stage-1 + world-model checkpoint merge
├── compute_command_class_weights.py         # (kept for reference; not used by default)
├── run_full_pipeline_split_301_75_10hz.sh   # stage1 -> merge -> stage2 -> eval -> submit
├── eval_holdout_l2.py                       # hold-out planning L2
├── eval_holdout_l2_shortcut_ablation.py     # target_point / BEV-refine shortcut diagnostics
├── etri_test_submit.py                      # test-set inference -> submission.json
├── sanity_check_nolcf_aux.py                # forward/backward + compliance gradient proof
├── measure_flops.py / measure_t_infer_fwd_only.py
evodrive_etri_prep/                          # Qwen teacher data prep / cache generation
```

## Running it

```bash
# 1. Build the 10Hz pkl (command-stratified 301/75 split)
python tools/data_converter/regenerate_causal_infos_10hz.py \
  --train-root data/train --test-root data/test \
  --output-dir data/etri/.causal_regen_split_301_75_10hz \
  --val-scenes val_scenes_301_75.txt

# 2. Full pipeline (stage1 48ep -> merge -> stage2 12ep -> eval -> submit -> FLOPs/T_infer)
bash tools/run_full_pipeline_split_301_75_10hz.sh
```

Each stage checks for its own output file before running, so the script
is safe to re-run after an interruption.

Before committing a multi-hour run, every technique in this repo was
verified on GPU with a real forward+backward pass (loss values, gradient
reaching the intended modules, and — for anything touching ego status —
the compliance gradient proof) rather than by code review alone.

## Evaluation metrics

- **Planning L2**: mean error (m) of ego position at 0.5/1.0/1.5/2.0/2.5/3.0s
- **Error Score** = `L2 × (1 + max(0, T_infer − 100) / 200)`; `T_infer` is
  the model-forward-only cumulative time for one clip (`reset_stream` →
  final frame)
- **FLOPs cutoff**: baseline (2,351.0 GFLOPs) × 3

## References

See [`ref_papers.md`](ref_papers.md).

[6]: https://arxiv.org/abs/2305.06242
[10]: https://arxiv.org/abs/2505.18945
