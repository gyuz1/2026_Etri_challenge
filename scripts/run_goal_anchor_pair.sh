#!/usr/bin/env bash
# Explicit preparation/launch for lateral-one vs adaptive-lateral anchors.
set -euo pipefail
source "$(dirname "$0")/_goal_anchor_pair.sh"
cd "$PAIR_ROOT"
usage() { echo "usage: bash scripts/run_goal_anchor_pair.sh <lat1|adaptive|both> [--dry-run|--check|--train]"; }
target="${1:-}"; mode="${2:---dry-run}"
case "$target" in lat1|adaptive) roles=("$target") ;; both) roles=(lat1 adaptive) ;; *) usage; exit 2 ;; esac
case "$mode" in --dry-run|--check|--train) ;; *) usage; exit 2 ;; esac
[ "$#" -le 2 ] || { usage; exit 2; }
for role in "${roles[@]}"; do pair_plan "$role"; done
if [ "$mode" = --dry-run ]; then
  echo 'DRY RUN: no remote access, synchronization, checks, or training performed.'
  exit 0
fi

# All roles must pass before either is launched. Never stop an existing job.
for role in "${roles[@]}"; do
  pair_role "$role"
  pair_require_idle "$PAIR_MACHINE"
  pair_verify_source "$PAIR_MACHINE"
  pair_exec "$PAIR_MACHINE" bash -c '
set -euo pipefail
cd /workspace/VAD
config=$1; eval_config=$2; work_dir=$3; donor=$4
if [ -e "$work_dir" ]; then
  echo "REFUSED: $work_dir already exists; no overwrite/resume is implied." >&2; exit 1
fi
test -f "$donor"
python tools/audit_pipeline.py "$config" --eval-config "$eval_config"
python tools/check_accel_block_live.py "$config"
CUDA_VISIBLE_DEVICES=0 python tools/check_goal_pred_live.py "$config" --n 6 --device 0
' bash "$PAIR_CONFIG" "$PAIR_EVAL_CONFIG" "$PAIR_WORK_DIR" "$PAIR_DONOR" </dev/null
done
if [ "$mode" = --check ]; then
  echo 'All preflights passed. No training was started.'
  exit 0
fi

for role in "${roles[@]}"; do
  pair_role "$role"
  pair_require_idle "$PAIR_MACHINE"
  # Atomic mkdir: unlike the old shared launcher, never rm/overwrite a work_dir.
  pair_exec "$PAIR_MACHINE" bash -c '
set -euo pipefail
cd /workspace/VAD
mkdir "$1"
printf "%s\n" "$2" > "$1/launch_config.txt"
' bash "$PAIR_WORK_DIR" "$PAIR_CONFIG" </dev/null
  pair_manifest | pair_exec "$PAIR_MACHINE" bash -c \
    'cd /workspace/VAD && tee "$1/source_manifest.sha256" >/dev/null' bash "$PAIR_WORK_DIR"
  pair_exec_detached "$PAIR_MACHINE" bash -c '
set -euo pipefail
cd /workspace/VAD
config=$1; work_dir=$2; port=$3; train_ann=$4; val_ann=$5
export WANDB_INIT_TIMEOUT=600 WANDB__SERVICE_WAIT=600 WANDB_HTTP_TIMEOUT=120
exec python -m torch.distributed.launch --nproc_per_node=2 --master_port="$port" \
  tools/train.py "$config" --launcher pytorch --work-dir "$work_dir" \
  --no-validate --seed 0 --deterministic \
  --cfg-options data.workers_per_gpu=2 data.train.ann_file="$train_ann" \
    data.val.ann_file="$val_ann" data.test.ann_file="$val_ann" \
  > "$work_dir/train.log" 2>&1
' bash "$PAIR_CONFIG" "$PAIR_WORK_DIR" "$PAIR_PORT" "$PAIR_TRAIN_ANN" "$PAIR_VAL_ANN"
  echo "Launch requested for $role on $PAIR_MACHINE; this is not a first-iteration success claim."
  echo "Read the complete checkpoint-loading and training log: /workspace/VAD/$PAIR_WORK_DIR/train.log"
done
