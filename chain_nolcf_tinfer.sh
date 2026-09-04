#!/usr/bin/env bash
# Auto-chain: wait for /data_fast/test copy + hold-out L2 eval to finish,
# then measure T_infer for the ego_lcf-off ablation checkpoint.
# Runs inside gyuz_split_3090, cwd /workspace/VAD.
set -uo pipefail
cd /workspace/VAD

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

STAGE2_DIR=work_dirs/stage2_etri_split_301_75_10hz_nolcf
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"
REAL_TEST_PKL='data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_test.pkl'
TINFER_OUT="${STAGE2_DIR}_t_infer.log"
NVME_TEST_DIR=/data_fast/test

log "Waiting for /data_fast/test copy to finish..."
while ! grep -q DONE_COPY work_dirs/copy_data_fast_test.log 2>/dev/null; do
    sleep 30
done
log "/data_fast/test ready."

log "Waiting for hold-out L2 eval (chain_nolcf_eval.log) to complete..."
while ! grep -qE 'EVAL_COMPLETE|^\[.*\] ERROR' work_dirs/chain_nolcf_eval.log 2>/dev/null; do
    sleep 60
done
if ! grep -q EVAL_COMPLETE work_dirs/chain_nolcf_eval.log; then
    log "ERROR: eval chain failed, not running T_infer."
    exit 1
fi
if [[ ! -f "$STAGE2_CKPT" ]]; then
    log "ERROR: $STAGE2_CKPT missing even though eval reported complete."
    exit 1
fi

log "Measuring T_infer (val clips, NVMe test copy, ego_lcf-off)..."
mkdir -p /data_fast/data
ln -sfn "$NVME_TEST_DIR" /data_fast/data/test
( cd /data_fast && python -u /workspace/VAD/tools/measure_t_infer_fwd_only.py \
    /workspace/VAD/projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nolcf.py \
    "/workspace/VAD/$STAGE2_CKPT" \
    --ann-file "/workspace/VAD/$REAL_TEST_PKL" \
    --frame-offsets 0 \
    --fp16 --warmup-clips 5 --num-clips 1120 ) | tee "$TINFER_OUT"
log "TINFER_COMPLETE: results in $TINFER_OUT"
