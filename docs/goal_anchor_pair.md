# Goal-anchor pair: preparation and explicit execution

These are new experiments, not a restart of `stage2_clean_goalpred` or
`stage2_clean_nodistill`. Nothing below stops an existing job or overwrites its
checkpoints. Implementation/readiness does not authorize starting training.

| Role | Default training machine | Work directory |
|---|---|---|
| `lat1` | A5000 (`gyuz_split2`) | `work_dirs/stage2_goalanchors_lat1_v1` |
| `adaptive` | 3090 (`gyuz_split_3090`) | `work_dirs/stage2_goalanchors_adaptive_v1` |

Both use the same forward-bin design, stage-1 donor, data split, seed 0, and
training schedule. The intended experimental variable is lateral subdivision.
Anchor generation must use the train split only. The new tables do not apply
retroactively to old goal-prediction checkpoints.

## Prepare without starting anything

```bash
bash scripts/run_goal_anchor_pair.sh both
bash scripts/sync_goal_anchor_pair.sh
bash scripts/eval_goal_anchor_pair.sh lat1 1 --select-goal-by-tp
```

All three commands default to dry-run and perform no remote access or GPU work.
The A5000 has a separate code copy. Once its existing jobs finish, explicitly
synchronize the complete source package (model, inherited configs, tools,
scripts), with SHA-256 verification and a recoverable pre-sync backup:

```bash
bash scripts/sync_goal_anchor_pair.sh --sync
```

Synchronization refuses active GPU compute/training processes. It does not
transfer datasets/checkpoints or delete extra destination files. Do not sync
while another agent is editing these files; a concurrent edit fails hash
verification and the resulting package must be regenerated.

## Preflight, then explicit training authorization

```bash
bash scripts/run_goal_anchor_pair.sh both --check
# Only when the user explicitly authorizes training:
bash scripts/run_goal_anchor_pair.sh both --train
```

`--check` performs real GPU diagnostic forwards, but no optimizer/training run.
GPU/process checks occur before these diagnostics. Each role verifies source
hashes and runs `audit_pipeline.py` with its paired eval config,
`check_accel_block_live.py`, and `check_goal_pred_live.py`. For `both`, all
preflights finish before either job starts. Existing work directories are
refused, not erased or silently resumed. Every checkpoint-loading/preflight
message is printed without filtering. Launch acknowledgement is not evidence
that iteration 100 succeeded; inspect the complete `train.log` afterwards.

After epoch 1, run the existing `check_accel_block_trained.py` with the exact
new train config and checkpoint, then inspect candidate utilization and losses.

## Matched evaluation

Use one idle machine for both results; 3090 is the default. Copy the A5000
checkpoint explicitly into its unique work directory on that machine first.
The scripts do not infer/copy a checkpoint or replace the eval config.

```bash
bash scripts/eval_goal_anchor_pair.sh lat1 1 --run
bash scripts/eval_goal_anchor_pair.sh lat1 1 --select-goal-by-tp --run
bash scripts/eval_goal_anchor_pair.sh adaptive 1 --run
bash scripts/eval_goal_anchor_pair.sh adaptive 1 --select-goal-by-tp --run
```

The fixed scoring conditions are 3-frame history (`0,-5,-10`), FP16,
BEV-only history, and `--test-commands`. Report predicted-goal and TP-selected
results separately. The latter uses TP only to choose an already generated
candidate, never to generate or correct a trajectory. Paired eval configs must
expose candidates for the TP-selection option. A reused eval log is refused.

Timing comparisons require otherwise idle GPUs; candidates may add decoder
work. Do not claim that an anchor-count change improves the contest score from
L2 alone. `EVAL_MACHINE=a5000` is an explicit override, not the default: verify
the raw val image mount there before using it (older infrastructure notes say
it was unavailable). All eval/checkpoint output is streamed unfiltered and
saved to the separate result log.
