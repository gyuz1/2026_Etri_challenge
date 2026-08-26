#!/usr/bin/env bash
# Stage1 -> merge -> stage2 -> val cache -> holdout eval -> submission,
# with independent STAGE1_KD/STAGE2_KD toggles for the EvoDriveVLA-teacher
# distillation ablation (VADHeadKD / loss_plan_kd). Mirrors
# run_full_pipeline.sh's skip-if-output-exists behavior; does NOT touch any
# of that script's files or work_dirs, so it's safe to prepare/launch this
# without disturbing a run already in progress under run_full_pipeline.sh.
#
# First planned use (per plan): STAGE1_KD=0 STAGE2_KD=1 -- stage1 reused
# as-is (non-KD, already trained), only stage2 gets loss_plan_kd. If that
# ablation shows a real L2 win, rerun with STAGE1_KD=1 STAGE2_KD=1 for the
# "KD from epoch 1" variant -- no code changes needed, just the toggles.
#
#   docker exec -it ad2026 bash
#   cd /workspace/VAD
#   STAGE1_KD=0 STAGE2_KD=1 nohup bash tools/run_full_pipeline_kd.sh \
#       > work_dirs/full_pipeline_kd.log 2>&1 &
#
# Each stage blocks until it finishes (or fails) before the next one starts.
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped.
set -euo pipefail
cd /workspace/VAD

STAGE1_KD="${STAGE1_KD:-0}"
STAGE2_KD="${STAGE2_KD:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

TEACHER_CACHE=work_dirs/teacher_cache/etri_train_teacher_cache.json
if [[ "$STAGE1_KD" == "1" || "$STAGE2_KD" == "1" ]]; then
    if [[ ! -f "$TEACHER_CACHE" ]]; then
        log "ERROR: STAGE1_KD/STAGE2_KD requested but $TEACHER_CACHE is missing."
        log "Run evodrive_etri_prep/generate_teacher_cache.py first, or set the KD flag(s) to 0."
        exit 1
    fi
fi

# --- Stage 1: VAD-only, ETRI domain adaptation, 48 epochs -----------------
if [[ "$STAGE1_KD" == "1" ]]; then
    STAGE1_DIR=work_dirs/stage1_etri_kd
    STAGE1_CFG=projects/configs/VAD/VAD_etri_tiny_stage1_cached_kd.py
else
    STAGE1_DIR=work_dirs/stage1_etri_v2
    STAGE1_CFG=projects/configs/VAD/VAD_etri_tiny_stage1_cached.py
fi
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"

if [[ -f "$STAGE1_CKPT" ]]; then
    log "Stage 1 (KD=$STAGE1_KD) already complete ($STAGE1_CKPT exists), skipping."
elif [[ "$STAGE1_KD" == "0" ]]; then
    log "ERROR: $STAGE1_CKPT not found. Non-KD stage1 is expected to already exist"
    log "(run tools/run_full_pipeline.sh first) -- this script does not retrain it"
    log "under STAGE1_KD=0 to avoid duplicating that ~40h job by accident."
    exit 1
else
    log "Stage 1 (KD=1): launching (48 epochs)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28811 \
        tools/train.py "$STAGE1_CFG" \
        --launcher pytorch --work-dir "$STAGE1_DIR" \
        --no-validate --deterministic
    log "Stage 1 (KD=1): done."
fi

# --- Merge: stage1 perception + LAW world model ----------------------------
MERGED_CKPT="$STAGE1_DIR/stage2_init_merged.pth"
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
if [[ "$STAGE2_KD" == "1" ]]; then
    STAGE2_DIR="work_dirs/stage2_etri_kd$([[ "$STAGE1_KD" == "1" ]] && echo _s1kd)"
    STAGE2_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_kd.py
    STAGE2_EVAL_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_kd.py
else
    STAGE2_DIR="work_dirs/stage2_etri$([[ "$STAGE1_KD" == "1" ]] && echo _s1kd)"
    STAGE2_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached.py
    STAGE2_EVAL_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_eval.py
fi
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"

# load_from in the stage2 configs points at the non-KD stage1 merge path by
# default; when this run's own merge output differs (KD stage1, or a _s1kd
# suffixed dir), override it explicitly so stage2 always inits from THIS
# run's merged checkpoint rather than whatever the config file hardcodes.
STAGE2_LOAD_FROM_OVERRIDE="--cfg-options load_from=$MERGED_CKPT"

if [[ -f "$STAGE2_CKPT" ]]; then
    log "Stage 2 (KD=$STAGE2_KD) already complete ($STAGE2_CKPT exists), skipping."
else
    log "Stage 2 (KD=$STAGE2_KD): launching (12 epochs)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28812 \
        tools/train.py "$STAGE2_CFG" $STAGE2_LOAD_FROM_OVERRIDE \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic
    log "Stage 2 (KD=$STAGE2_KD): done."
fi

# --- Val geometry cache (shared across ablations, build once) -------------
VAL_CACHE_ROOT=work_dirs/etri_geometry_cache_val_v1
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

# --- Hold-out planning L2 evaluation (submission-faithful proxy metric) ---
EVAL_OUT="$STAGE2_DIR/holdout_l2.txt"
if [[ -f "$EVAL_OUT" ]]; then
    log "Eval output already exists ($EVAL_OUT), skipping. Delete it to rerun."
else
    log "Running hold-out L2 evaluation..."
    python tools/eval_holdout_l2.py \
        "$STAGE2_EVAL_CFG" \
        "$STAGE2_CKPT" \
        --ann-file data/etri/.causal_regen/vad_etri_infos_temporal_val_split.pkl \
        | tee "$EVAL_OUT"
    log "Eval: done, results in $EVAL_OUT."
fi

# --- Real submission generation (actual leaderboard inference path) -------
SUBMISSION_OUT="$STAGE2_DIR/submission.json"
if [[ -f "$SUBMISSION_OUT" ]]; then
    log "Submission already exists ($SUBMISSION_OUT), skipping."
else
    log "Generating submission..."
    python tools/etri_test_submit.py \
        "$STAGE2_EVAL_CFG" \
        "$STAGE2_CKPT" \
        --ann-file data/etri/.causal_regen/vad_etri_infos_temporal_test.pkl \
        --out "$SUBMISSION_OUT"
    log "Submission: done, written to $SUBMISSION_OUT."
fi

log "Pipeline complete (STAGE1_KD=$STAGE1_KD, STAGE2_KD=$STAGE2_KD)."
log "  stage1 ckpt:  $STAGE1_CKPT"
log "  stage2 ckpt:  $STAGE2_CKPT"
log "  holdout eval: $EVAL_OUT"
log "  submission:   $SUBMISSION_OUT"
