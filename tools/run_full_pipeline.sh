#!/usr/bin/env bash
# Run stage1 -> merge -> stage2 -> val cache -> eval, skipping any step
# whose output already exists. Meant to be run inside the ad2026 container,
# from /workspace/VAD, e.g.:
#
#   docker exec -it ad2026 bash
#   cd /workspace/VAD
#   nohup bash tools/run_full_pipeline.sh > work_dirs/full_pipeline.log 2>&1 &
#
# Each stage blocks until it finishes (or fails) before the next one starts.
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped.
set -euo pipefail
cd /workspace/VAD

STAGE1_DIR=work_dirs/stage1_etri_v2
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"
MERGED_CKPT="$STAGE1_DIR/stage2_init_merged.pth"
STAGE2_DIR=work_dirs/stage2_etri
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"
VAL_CACHE_ROOT=work_dirs/etri_geometry_cache_val_v1
EVAL_OUT=work_dirs/stage2_etri_holdout_l2.txt

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Stage 1: VAD-only, ETRI domain adaptation, 48 epochs -----------------
if [[ -f "$STAGE1_CKPT" ]]; then
    log "Stage 1 already complete ($STAGE1_CKPT exists), skipping."
else
    log "Stage 1: launching (48 epochs)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28801 \
        tools/train.py projects/configs/VAD/VAD_etri_tiny_stage1_cached.py \
        --launcher pytorch --work-dir "$STAGE1_DIR" \
        --no-validate --deterministic
    log "Stage 1: done."
fi

# --- Merge: stage1 perception + LAW world model ----------------------------
if [[ -f "$MERGED_CKPT" ]]; then
    log "Merged checkpoint already exists ($MERGED_CKPT), skipping."
else
    log "Merging stage1 checkpoint with LAW world model..."
    python tools/merge_stage1_world_model.py \
        --stage1 "$STAGE1_CKPT" \
        --world-model-source ckpts/law_pretrained_nus.pth \
        --output "$MERGED_CKPT"
    log "Merge: done."
fi

# --- Stage 2: VADLAW, waypoint + world model, 12 epochs --------------------
if [[ -f "$STAGE2_CKPT" ]]; then
    log "Stage 2 already complete ($STAGE2_CKPT exists), skipping."
else
    log "Stage 2: launching (12 epochs)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28802 \
        tools/train.py projects/configs/VAD/VADLAW_etri_tiny_cached.py \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic
    log "Stage 2: done."
fi

# --- Val geometry cache (needed for a fast eval_holdout_l2.py run) --------
# CPU-only; run only after stage 2 finishes so it never contends with GPU
# training for CPU/disk bandwidth (see the earlier throughput regression
# when this ran alongside stage1).
VAL_SCENE_COUNT=$(ls "$VAL_CACHE_ROOT"/*.json 2>/dev/null | grep -vc cache_manifest || true)
if [[ "$VAL_SCENE_COUNT" -ge 32 ]]; then
    log "Val geometry cache already complete ($VAL_SCENE_COUNT/32), skipping."
else
    log "Building val geometry cache ($VAL_SCENE_COUNT/32 so far)..."
    python tools/cache/etri_geometry_cache.py \
        --ann-file data/etri/.causal_regen/vad_etri_infos_temporal_val_split.pkl \
        --cache-root "$VAL_CACHE_ROOT" --workers 12
    log "Val geometry cache: done."
fi

# --- Hold-out planning L2 evaluation ---------------------------------------
if [[ -f "$EVAL_OUT" ]]; then
    log "Eval output already exists ($EVAL_OUT), skipping. Delete it to rerun."
else
    log "Running hold-out L2 evaluation..."
    python tools/eval_holdout_l2.py \
        projects/configs/VAD/VADLAW_etri_tiny_cached_eval.py \
        "$STAGE2_CKPT" \
        --ann-file data/etri/.causal_regen/vad_etri_infos_temporal_val_split.pkl \
        | tee "$EVAL_OUT"
    log "Eval: done, results in $EVAL_OUT."
fi

log "Pipeline complete."
