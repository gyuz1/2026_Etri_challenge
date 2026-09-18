#!/usr/bin/env bash
# Evaluate one finished run on the machine that trained it.
# Usage (inside the container):
#   bash scripts/eval_after_training.sh <work_dir> <eval_config> <epoch> <select-flag>
# e.g. ... work_dirs/stage2_cellplanner_v1 projects/configs/VAD/..._cellplanner.py 12 --select-cell-by-tp
#
# L2 for the EMA weights (the ordinary slots) and the raw weights in parallel,
# 3-frame and 2-frame each; then T_infer with the machine otherwise idle.
# Existing logs are never overwritten.
set -u
cd /workspace/VAD
WORK_DIR=$1; CFG=$2; EPOCH=$3; SELECT=${4:-}
VAL=data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl
CKPT=$WORK_DIR/epoch_${EPOCH}.pth
RAW=$WORK_DIR/epoch_${EPOCH}_rawweights.pth

run() {  # run <gpu> <log> <checkpoint> [extra flags...]
  local gpu=$1 log=$2 ckpt=$3; shift 3
  if [ -e "$log" ]; then echo "skip (exists): $log"; return; fi
  echo "=== $(date '+%F %T') start gpu$gpu $log"
  CUDA_VISIBLE_DEVICES=$gpu python tools/eval_holdout_l2_and_tinfer.py "$CFG" "$ckpt" \
    --ann-file "$VAL" --frame-offsets 0,-5,-10 0,-5 --fp16 --bev-only-history \
    --device 0 $SELECT "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') done rc=$? $log"
}

test -f "$CKPT" || { echo "missing $CKPT"; exit 1; }
[ -e "$RAW" ] || python tools/make_raw_weight_ckpt.py "$CKPT" "$RAW"

run 0 "$WORK_DIR/eval_ep${EPOCH}_3f2f_ema.log" "$CKPT" &
run 1 "$WORK_DIR/eval_ep${EPOCH}_3f2f_raw.log" "$RAW" &
wait

# Both GPUs idle now, so these timings are the ones to quote.
run 0 "$WORK_DIR/timing_ep${EPOCH}_3f2f.log" "$CKPT" --stride 50 --warmup-windows 20
echo "=== $(date '+%F %T') queue finished"
