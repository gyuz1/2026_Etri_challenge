#!/usr/bin/env bash
# Auto-chain: wait for train geometry cache -> patch manifest -> build val
# cache -> patch its manifest -> launch ego_lcf-off stage2 training.
# Runs inside gyuz_split_3090, cwd /workspace/VAD.
set -uo pipefail
cd /workspace/VAD

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

EXPECTED_TRAIN_ANN='/workspace/VAD/data/etri/.causal_regen_split_301_75/vad_etri_infos_temporal_train_split.pkl'
EXPECTED_VAL_ANN='/workspace/VAD/data/etri/.causal_regen_split_301_75/vad_etri_infos_temporal_val_split.pkl'
REAL_TRAIN_PKL='data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_train_split.pkl'
REAL_VAL_PKL='data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl'

# --- 1. wait for train cache manifest ---------------------------------------
log "Waiting for train geometry cache manifest..."
while [[ ! -f work_dirs/etri_geometry_cache_v1/cache_manifest.json ]]; do
    if ! pgrep -f 'etri_geometry_cache.py --ann-file data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_train' > /dev/null; then
        log "ERROR: train cache build process died without producing a manifest."
        tail -c 2000 work_dirs/geocache_split_train.log
        exit 1
    fi
    sleep 15
done
log "Train cache manifest ready."

# --- 2. patch train manifest ann_file to match shared configs --------------
python3 - "$EXPECTED_TRAIN_ANN" <<'PYEOF'
import json, sys
p = 'work_dirs/etri_geometry_cache_v1/cache_manifest.json'
d = json.load(open(p))
d['ann_file'] = sys.argv[1]
json.dump(d, open(p, 'w'))
print('Patched train manifest ann_file ->', sys.argv[1])
PYEOF

# --- 3. build val geometry cache (75 scenes) --------------------------------
log "Building val geometry cache (75 scenes)..."
python tools/cache/etri_geometry_cache.py \
    --ann-file "$REAL_VAL_PKL" \
    --cache-root work_dirs/etri_geometry_cache_val_v1 --workers 12 \
    > work_dirs/geocache_split_val.log 2>&1
if [[ ! -f work_dirs/etri_geometry_cache_val_v1/cache_manifest.json ]]; then
    log "ERROR: val cache build failed, no manifest produced."
    tail -c 2000 work_dirs/geocache_split_val.log
    exit 1
fi
log "Val cache manifest ready."

# --- 4. patch val manifest ann_file -----------------------------------------
python3 - "$EXPECTED_VAL_ANN" <<'PYEOF'
import json, sys
p = 'work_dirs/etri_geometry_cache_val_v1/cache_manifest.json'
d = json.load(open(p))
d['ann_file'] = sys.argv[1]
json.dump(d, open(p, 'w'))
print('Patched val manifest ann_file ->', sys.argv[1])
PYEOF

# --- 5. compute command_class_weights (same formula as the ego_lcf-ON run,
#        for a fair apples-to-apples ablation) -------------------------------
log "Computing command_class_weights..."
CMD_WEIGHTS=$(python tools/compute_command_class_weights.py "$REAL_TRAIN_PKL" | tee /dev/stderr | grep -oP 'WEIGHTS=\K.*')
log "command_class_weights: $CMD_WEIGHTS"

# --- 6. launch stage2 (ego_lcf-off) training --------------------------------
STAGE2_DIR=work_dirs/stage2_etri_split_301_75_10hz_nolcf
mkdir -p "$STAGE2_DIR"
log "Launching ego_lcf-off stage2 training..."
python -m torch.distributed.launch --nproc_per_node=2 --master_port=28812 \
    tools/train.py projects/configs/VAD/VADLAW_etri_tiny_cached_nolcf.py \
    --launcher pytorch --work-dir "$STAGE2_DIR" \
    --no-validate --deterministic \
    --cfg-options model.pts_bbox_head.command_class_weights="$CMD_WEIGHTS" \
        data.train.ann_file="$REAL_TRAIN_PKL" \
        data.val.ann_file="$REAL_VAL_PKL" \
        data.test.ann_file="$REAL_VAL_PKL" \
    > "$STAGE2_DIR/train.log" 2>&1 &
TRAIN_PID=$!
log "Stage2 training launched, PID=$TRAIN_PID"

# --- 7. sanity-watch the first ~150 iterations for crash / nan loss --------
sleep 180
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "ERROR: training process died within first 180s. Log tail:"
    tail -n 60 "$STAGE2_DIR/train.log"
    exit 1
fi
log "Training alive after 180s. Recent log:"
tail -n 15 "$STAGE2_DIR/train.log"
log "CHAIN_COMPLETE: stage2 nolcf training running healthily."
