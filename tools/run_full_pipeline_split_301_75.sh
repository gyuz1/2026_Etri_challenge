#!/usr/bin/env bash
# Stage1 -> merge -> stage2 -> val cache -> holdout eval -> test-set
# submission, on the YAKDEEE/ETRI-E2E branch's stratified 301/75 train/val
# split (data/etri/.causal_regen_split_301_75/), with the TRAJ_STEP-
# aligned causal ego-motion fix and the 5s target-point branch both baked
# into that pkl and enabled in every config used here. Meant to be run
# inside the ad2026 container, from /workspace/VAD, e.g.:
#
#   docker exec -it ad2026 bash
#   cd /workspace/VAD
#   nohup bash tools/run_full_pipeline_split_301_75.sh \
#       > work_dirs/full_pipeline_split_301_75.log 2>&1 &
#
# Each stage blocks until it finishes (or fails) before the next one starts.
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped. Uses NEW work-dirs (does not
# touch stage1_etri_v2/ or stage2_etri/, the old-split checkpoints).
set -euo pipefail
cd /workspace/VAD

DATA_ROOT=data/etri/.causal_regen_split_301_75
STAGE1_DIR=work_dirs/stage1_etri_split_301_75
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"
MERGED_CKPT="$STAGE1_DIR/stage2_init_merged.pth"
STAGE2_DIR=work_dirs/stage2_etri_split_301_75
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"
TRAIN_CACHE_ROOT=work_dirs/etri_geometry_cache_v1
VAL_CACHE_ROOT=work_dirs/etri_geometry_cache_val_v1
EVAL_OUT=work_dirs/stage2_etri_split_301_75_holdout_l2.txt
SUBMIT_OUT=work_dirs/stage2_etri_split_301_75_submission.json
TINFER_OUT=work_dirs/stage2_etri_split_301_75_t_infer.log
# Test images copied onto the container's NVMe-backed writable layer
# (see docker info's DockerRootDir) instead of the HDD-backed dataset/ bind
# mount -- a full-1125-clip T_infer sweep touches far more files than fit
# in the page cache, so on the HDD path most reads are genuine random
# seeks (measured 207ms vs 114ms for a small, page-cache-hot subset on the
# same code/model). Image bytes are identical to dataset/test/ (only
# annotations changed in the new pkl), so this copy is still valid.
NVME_TEST_DIR=/data_fast/test

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Geometry caches: extend to the new split's scene sets -----------------
# Scene shards are content-addressed by scene_token and independent of which
# split a scene falls into, so this reuses every already-built shard from
# the old split and only decodes the net-new scenes. Cheap/idempotent to
# re-run (each scene logs "skipped" once cached), so always run rather than
# gate behind a scene-count heuristic.
log "Extending train geometry cache to the new 301-scene train split..."
python tools/cache/etri_geometry_cache.py \
    --ann-file "$DATA_ROOT/vad_etri_infos_temporal_train_split.pkl" \
    --cache-root "$TRAIN_CACHE_ROOT" --workers 12
log "Train geometry cache: done."

log "Extending val geometry cache to the new 75-scene val split..."
python tools/cache/etri_geometry_cache.py \
    --ann-file "$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl" \
    --cache-root "$VAL_CACHE_ROOT" --workers 12
log "Val geometry cache: done."

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
    # VADLAW_etri_tiny.py's load_from is hardcoded to the OLD split's
    # stage1_etri_v2/stage2_init_merged.pth -- override it so stage 2
    # actually initializes from the merged checkpoint built above from
    # THIS run's stage 1, not the old-split one.
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28802 \
        tools/train.py projects/configs/VAD/VADLAW_etri_tiny_cached.py \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic \
        --cfg-options load_from="$MERGED_CKPT"
    log "Stage 2: done."
fi

# --- Hold-out planning L2 evaluation ---------------------------------------
if [[ -f "$EVAL_OUT" ]]; then
    log "Eval output already exists ($EVAL_OUT), skipping. Delete it to rerun."
else
    log "Running hold-out L2 evaluation..."
    python tools/eval_holdout_l2.py \
        projects/configs/VAD/VADLAW_etri_tiny_cached_eval.py \
        "$STAGE2_CKPT" \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl" \
        | tee "$EVAL_OUT"
    log "Eval: done, results in $EVAL_OUT."
fi

# --- Test-set inference / submission ----------------------------------------
# Uses the live (non-cached) fast-eval pipeline, not a geometry cache -- the
# cache is a local training-speed optimization that has no reason to exist
# in the grading environment, so submission generation must go through the
# same undistort/crop/resize path a real grading run would use.
if [[ -f "$SUBMIT_OUT" ]]; then
    log "Submission output already exists ($SUBMIT_OUT), skipping. Delete it to rerun."
else
    log "Running test-set inference (1125 clips, live fast-eval pipeline)..."
    python tools/etri_test_submit.py \
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "$STAGE2_CKPT" \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_test.pkl" \
        --out "$SUBMIT_OUT"
    log "Test inference: done, submission in $SUBMIT_OUT."
fi

# --- FLOPs measurement (1st-round cutoff: <= 3x baseline ~= 7053 GFLOPs) ---
# Single warm frame (prev_bev already populated), NOT cumulative like
# T_infer below -- confirmed by the organizers' own tools/measure_flops.py
# design (builds prev_bev on one throwaway forward pass, then FlopCounter-
# wraps a second call on the same sample) and independent of the frame-
# count/spacing choice T_infer sweeps over, since a warm frame's own cost
# doesn't depend on how many frames preceded it. Merges __flops__ into the
# same submission.json test inference already wrote.
if grep -q '"__flops__"' "$SUBMIT_OUT" 2>/dev/null; then
    log "FLOPs already recorded in $SUBMIT_OUT, skipping."
else
    log "Measuring FLOPs (single warm frame)..."
    python tools/measure_flops.py \
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_test.pkl" \
        --out "$SUBMIT_OUT"
    log "FLOPs: done, merged into $SUBMIT_OUT."
fi

# --- T_infer measurement (Error Score latency term) -------------------------
# Same config/checkpoint as the submission step above, timed per clip:
# T_infer is the PER-CLIP CUMULATIVE time of every forward pass from
# reset_stream through the final frame's output (confirmed by the
# organizers' official Q&A, 2026-08-24), not a single frame's latency --
# see measure_t_infer.py's docstring. Uses the NVMe test-image copy so
# disk I/O doesn't dominate the number the way it does against the
# HDD-backed dataset/ mount (see NVME_TEST_DIR comment).
if [[ -f "$TINFER_OUT" ]]; then
    log "T_infer measurement already exists ($TINFER_OUT), skipping. Delete it to rerun."
else
    log "Measuring T_infer (1125 clips, NVMe test copy)..."
    if [[ ! -d "$NVME_TEST_DIR" ]]; then
        log "ERROR: $NVME_TEST_DIR not found -- copy dataset/test there first " \
            "(mkdir -p /data_fast && cp -r data/test /data_fast/test) or edit " \
            "NVME_TEST_DIR to point at the HDD-backed data/test (adds cold-" \
            "disk-I/O overhead on top of the real per-clip cumulative cost, " \
            "not a code/model difference)."
        exit 1
    fi
    mkdir -p /data_fast/data
    ln -sfn "$NVME_TEST_DIR" /data_fast/data/test
    ( cd /data_fast && python -u /workspace/VAD/tools/measure_t_infer.py \
        /workspace/VAD/projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "/workspace/VAD/$STAGE2_CKPT" \
        --ann-file "/workspace/VAD/$DATA_ROOT/vad_etri_infos_temporal_test.pkl" \
        --fp16 --warmup-clips 5 --num-clips 1120 ) | tee "$TINFER_OUT"
    log "T_infer measurement: done, results in $TINFER_OUT."
fi

log "Pipeline complete."
