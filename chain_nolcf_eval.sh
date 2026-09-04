#!/usr/bin/env bash
# Auto-chain: wait for ego_lcf-off stage2 training to finish (epoch_12.pth),
# then run hold-out planning L2 eval against it. Runs inside gyuz_split_3090,
# cwd /workspace/VAD.
set -uo pipefail
cd /workspace/VAD

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

STAGE2_DIR=work_dirs/stage2_etri_split_301_75_10hz_nolcf
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"
REAL_VAL_PKL='data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl'
EVAL_OUT="${STAGE2_DIR}_holdout_l2.txt"

# --- 1. wait for the training process to actually start (work-dir + log) ---
log "Waiting for stage2 training log to appear..."
while [[ ! -f "$STAGE2_DIR/train.log" ]]; do
    sleep 30
done

# --- 2. wait for epoch_12.pth (or training process death) ------------------
log "Waiting for $STAGE2_CKPT (training may take ~1 day)..."
while [[ ! -f "$STAGE2_CKPT" ]]; do
    if ! pgrep -f 'tools/train.py projects/configs/VAD/VADLAW_etri_tiny_cached_nolcf.py' > /dev/null; then
        log "ERROR: nolcf stage2 training process is no longer running, and no $STAGE2_CKPT was produced."
        tail -n 60 "$STAGE2_DIR/train.log"
        exit 1
    fi
    sleep 120
done
log "Stage2 training complete, $STAGE2_CKPT found."

# --- 3. hold-out L2 eval -----------------------------------------------------
log "Running hold-out L2 evaluation (ego_lcf-off)..."
python tools/eval_holdout_l2.py \
    projects/configs/VAD/VADLAW_etri_tiny_cached_eval_nolcf.py \
    "$STAGE2_CKPT" \
    --ann-file "$REAL_VAL_PKL" \
    --frame-offsets 0 \
    | tee "$EVAL_OUT"
log "EVAL_COMPLETE: results in $EVAL_OUT"
