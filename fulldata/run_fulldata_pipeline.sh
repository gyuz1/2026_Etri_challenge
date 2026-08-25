#!/usr/bin/env bash
# Final-submission pipeline: stage1 -> merge -> stage2 -> test-set
# submission -> FLOPs -> T_infer, trained on ALL 376 scenes (no train/val
# split, see fulldata/configs/*.py's docstrings) instead of the separate
# 301/75-split track used for L2/T_infer experimentation. No hold-out L2
# step here -- there is no val split left to score against once every
# scene is in the training set.
#
# Run inside the ad2026 container, from /workspace/VAD:
#   docker exec -it ad2026 bash
#   cd /workspace/VAD
#   nohup bash fulldata/run_fulldata_pipeline.sh \
#       > work_dirs/fulldata_pipeline.log 2>&1 &
#
# Safe to re-run after an interruption -- already-finished stages are
# skipped by their output files. Uses its own work-dirs (does not touch
# the 301/75-split checkpoints from the separate L2/T_infer
# experimentation track).
set -euo pipefail
cd /workspace/VAD

DATA_ROOT=data/etri/annotations
FULL_TRAIN_ANN="$DATA_ROOT/vad_etri_infos_fulldata_train.pkl"
TEST_ANN="$DATA_ROOT/vad_etri_infos_fulldata_test.pkl"

STAGE1_DIR=work_dirs/stage1_etri_fulldata
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"
MERGED_CKPT="$STAGE1_DIR/stage2_init_merged.pth"
STAGE2_DIR=work_dirs/stage2_etri_fulldata
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"

TRAIN_CACHE_ROOT=work_dirs/etri_geometry_cache_v1
SUBMIT_OUT=work_dirs/stage2_etri_fulldata_submission.json
TINFER_OUT=work_dirs/stage2_etri_fulldata_t_infer.log
NVME_TEST_DIR=/data_fast/test

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Geometry cache: cover all 376 scenes ----------------------------------
# Idempotent (already-cached scenes log "skipped"), safe to always run.
log "Extending geometry cache to all 376 scenes..."
python tools/cache/etri_geometry_cache.py \
    --ann-file "$FULL_TRAIN_ANN" \
    --cache-root "$TRAIN_CACHE_ROOT" --workers 12
log "Geometry cache: done."

# --- Stage 1: VAD-only, full 376-scene set, 48 epochs ----------------------
if [[ -f "$STAGE1_CKPT" ]]; then
    log "Stage 1 already complete ($STAGE1_CKPT exists), skipping."
else
    log "Stage 1: launching (48 epochs, fulldata)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28811 \
        tools/train.py fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata.py \
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

# --- Stage 2: VADLAW, full 376-scene set, 12 epochs -------------------------
if [[ -f "$STAGE2_CKPT" ]]; then
    log "Stage 2 already complete ($STAGE2_CKPT exists), skipping."
else
    log "Stage 2: launching (12 epochs, fulldata)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28812 \
        tools/train.py fulldata/configs/VADLAW_etri_tiny_cached_fulldata.py \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic \
        --cfg-options load_from="$MERGED_CKPT"
    log "Stage 2: done."
fi

# --- Test-set inference / submission ----------------------------------------
# Uses the live (non-cached) fast-eval pipeline (same as the 301/75-split
# track) -- geometry cache is a local training-speed optimization that has
# no reason to exist in the grading environment. VADLAW_etri_tiny_fast_eval
# only overrides data.test, so it's unaffected by which train config was
# used -- only the checkpoint (STAGE2_CKPT) changes here.
if [[ -f "$SUBMIT_OUT" ]]; then
    log "Submission output already exists ($SUBMIT_OUT), skipping. Delete it to rerun."
else
    log "Running test-set inference (1125 clips, live fast-eval pipeline)..."
    python tools/etri_test_submit.py \
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "$STAGE2_CKPT" \
        --ann-file "$TEST_ANN" \
        --out "$SUBMIT_OUT"
    log "Test inference: done, submission in $SUBMIT_OUT."
fi

# --- FLOPs measurement (1st-round cutoff: <= 3x baseline ~= 7053 GFLOPs) ---
# Single warm frame, independent of train data / frame-count choices.
if grep -q '"__flops__"' "$SUBMIT_OUT" 2>/dev/null; then
    log "FLOPs already recorded in $SUBMIT_OUT, skipping."
else
    log "Measuring FLOPs (single warm frame)..."
    python tools/measure_flops.py \
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        --ann-file "$TEST_ANN" \
        --out "$SUBMIT_OUT"
    log "FLOPs: done, merged into $SUBMIT_OUT."
fi

# --- T_infer measurement (Error Score latency term) -------------------------
# Per-clip cumulative time (organizers' official Q&A, 2026-08-24), same
# 7-frame window etri_test_submit.py above just used -- NOT the 5-frame
# reduction from the L2/T_infer experimentation track (that change hasn't
# been applied to the live test pkl/converter yet; when it is, update
# --frame-offsets here to match what etri_test_submit.py actually replays).
if [[ -f "$TINFER_OUT" ]]; then
    log "T_infer measurement already exists ($TINFER_OUT), skipping. Delete it to rerun."
else
    log "Measuring T_infer (1125 clips, NVMe test copy)..."
    if [[ ! -d "$NVME_TEST_DIR" ]]; then
        log "ERROR: $NVME_TEST_DIR not found -- copy dataset/test there first " \
            "(mkdir -p /data_fast && cp -r data/test /data_fast/test) or point " \
            "NVME_TEST_DIR at the HDD-backed data/test (adds cold-disk-I/O " \
            "overhead on top of the real per-clip cumulative cost)."
        exit 1
    fi
    mkdir -p /data_fast/data
    ln -sfn "$NVME_TEST_DIR" /data_fast/data/test
    ( cd /data_fast && python -u /workspace/VAD/tools/measure_t_infer.py \
        /workspace/VAD/projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "/workspace/VAD/$STAGE2_CKPT" \
        --ann-file "/workspace/VAD/$TEST_ANN" \
        --frame-offsets=-30,-25,-20,-15,-10,-5,0 \
        --fp16 --warmup-clips 5 --num-clips 1120 ) | tee "$TINFER_OUT"
    log "T_infer measurement: done, results in $TINFER_OUT."
fi

log "Pipeline complete."
