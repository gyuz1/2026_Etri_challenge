#!/usr/bin/env bash
# Isolated holdout evaluation; does not copy a checkpoint or start training.
set -euo pipefail
source "$(dirname "$0")/_goal_anchor_pair.sh"
cd "$PAIR_ROOT"
usage() { echo "usage: bash scripts/eval_goal_anchor_pair.sh <lat1|adaptive> <epoch> [--select-goal-by-tp] [--run]"; }
[ "$#" -ge 2 ] || { usage; exit 2; }
role=$1; epoch=$2; shift 2
pair_role "$role"
case "$epoch" in ''|*[!0-9]*) usage; exit 2 ;; esac
run=false; selection=predicted; flags=()
for opt in "$@"; do
  case "$opt" in
    --run) run=true ;;
    --select-goal-by-tp) selection=tp_select; flags=(--select-goal-by-tp) ;;
    *) usage; exit 2 ;;
  esac
done
# Use one machine for both reported results; copy A5000's checkpoint explicitly.
machine="${EVAL_MACHINE:-3090}"; gpu="${EVAL_GPU:-0}"
case "$machine" in 3090|a5000) ;; *) echo 'EVAL_MACHINE must be 3090 or a5000' >&2; exit 2 ;; esac
case "$gpu" in ''|*[!0-9]*) echo 'EVAL_GPU must be a nonnegative integer' >&2; exit 2 ;; esac
ckpt="$PAIR_WORK_DIR/epoch_${epoch}.pth"
out="$PAIR_WORK_DIR/eval_ep${epoch}_3frame_testcommands_${selection}.log"
printf 'machine=%s config=%s checkpoint=%s selection=%s\nlog=%s\n' "$machine" "$PAIR_EVAL_CONFIG" "$ckpt" "$selection" "$out"
if ! "$run"; then
  echo 'DRY RUN: add --run to execute evaluation; no remote access performed.'
  exit 0
fi
pair_require_idle "$machine"
pair_verify_source "$machine"
pair_exec "$machine" bash -c '
set -euo pipefail
cd /workspace/VAD
config=$1; ckpt=$2; ann=$3; gpu=$4; out=$5; shift 5
test -f "$ckpt"
if [ -e "$out" ]; then echo "REFUSED: evaluation log already exists: $out" >&2; exit 1; fi
python tools/eval_holdout_l2_and_tinfer.py "$config" "$ckpt" \
  --ann-file "$ann" --frame-offsets 0,-5,-10 --fp16 --bev-only-history \
  --test-commands --device "$gpu" "$@" 2>&1 | tee "$out"
' bash "$PAIR_EVAL_CONFIG" "$ckpt" "$PAIR_VAL_ANN" "$gpu" "$out" "${flags[@]}" </dev/null
