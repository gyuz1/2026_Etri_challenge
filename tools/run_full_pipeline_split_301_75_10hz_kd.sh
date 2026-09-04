#!/usr/bin/env bash
# 10Hz split_301_75 + EvoDriveVLA-teacher (Qwen) distillation, with
# independent STAGE1_KD/STAGE2_KD toggles -- ported from the older
# run_full_pipeline_kd.sh (which still targeted the pre-redesign 32-scene
# split, no PRISM, no 10Hz) onto the current split_301_75_10hz/PRISM setup.
# Does NOT touch run_full_pipeline_split_301_75_10hz.sh's files or work_dirs
# -- safe to prepare/launch without disturbing that run.
#
# Two independent modes, chosen by STAGE1_KD/STAGE2_KD:
#
#   STAGE1_KD=0 STAGE2_KD=1  (default, "warm-start"/stage3):
#     Reuses the already-trained, non-KD stage1+stage2 checkpoints as-is.
#     Short (5-epoch) continued fine-tune of stage2 with loss_plan_kd added,
#     warm-started from work_dirs/stage2_etri_split_301_75_10hz/epoch_12.pth.
#     Cheap (~a few hours), but KD is introduced late so its effect on
#     learned representations is weaker than training with it from the
#     start -- see VADLAW_etri_tiny_cached_kd.py's docstring.
#
#   STAGE1_KD=1 STAGE2_KD=1  ("KD from epoch 1"):
#     Retrains stage1 (48ep) AND stage2 (12ep) from scratch with loss_plan_kd
#     active throughout. Strongest expected effect (KD shapes the learned
#     features, not just a late nudge), but redoes the full ~60h job under
#     new work-dirs (stage1_etri_split_301_75_10hz_kd,
#     stage2_etri_split_301_75_10hz_kd) rather than reusing anything.
#
# Requires the teacher cache to exist first (needs the fine-tuned Qwen
# checkpoint): see evodrive_etri_prep/generate_teacher_cache.py.
#
#   docker exec gyuz_split bash -c 'cd /workspace/VAD && \
#     STAGE1_KD=0 STAGE2_KD=1 nohup bash \
#     tools/run_full_pipeline_split_301_75_10hz_kd.sh \
#     > work_dirs/full_pipeline_split_301_75_10hz_kd.log 2>&1 &'
#
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped.
set -euo pipefail
cd /workspace/VAD

STAGE1_KD="${STAGE1_KD:-0}"
STAGE2_KD="${STAGE2_KD:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

DATA_ROOT=data/etri/.causal_regen_split_301_75_10hz
ANN_OVERRIDES=(
    data.train.ann_file="$DATA_ROOT/vad_etri_infos_temporal_train_split.pkl"
    data.val.ann_file="$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl"
    data.test.ann_file="$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl"
)

TEACHER_CACHE=work_dirs/teacher_cache/etri_train_teacher_cache.json
if [[ "$STAGE1_KD" == "1" || "$STAGE2_KD" == "1" ]]; then
    if [[ ! -f "$TEACHER_CACHE" ]]; then
        log "ERROR: STAGE1_KD/STAGE2_KD requested but $TEACHER_CACHE is missing."
        log "Run evodrive_etri_prep/generate_teacher_cache.py first (needs the"
        log "fine-tuned Qwen checkpoint), or set the KD flag(s) to 0."
        exit 1
    fi
fi

# --- Stage 1: VAD-only, 48 epochs, 10Hz ego-motion -------------------------
if [[ "$STAGE1_KD" == "1" ]]; then
    STAGE1_DIR=work_dirs/stage1_etri_split_301_75_10hz_kd
    STAGE1_CFG=projects/configs/VAD/VAD_etri_tiny_stage1_cached_kd.py
else
    STAGE1_DIR=work_dirs/stage1_etri_split_301_75_10hz
    STAGE1_CFG=projects/configs/VAD/VAD_etri_tiny_stage1_cached.py
fi
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"

if [[ -f "$STAGE1_CKPT" ]]; then
    log "Stage 1 (KD=$STAGE1_KD) already complete ($STAGE1_CKPT exists), skipping."
elif [[ "$STAGE1_KD" == "0" ]]; then
    log "ERROR: $STAGE1_CKPT not found. Non-KD stage1 is expected to already"
    log "exist (run tools/run_full_pipeline_split_301_75_10hz.sh first) --"
    log "this script does not retrain it under STAGE1_KD=0 to avoid"
    log "duplicating that ~40h job by accident."
    exit 1
else
    log "Stage 1 (KD=1): launching (48 epochs, 10Hz ego-motion)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28821 \
        tools/train.py "$STAGE1_CFG" \
        --launcher pytorch --work-dir "$STAGE1_DIR" \
        --no-validate --deterministic \
        --cfg-options "${ANN_OVERRIDES[@]}"
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

# --- Stage 2: VADLAW, 10Hz ego-motion, PRISM enabled, +KD if requested ----
NONKD_STAGE2_CKPT=work_dirs/stage2_etri_split_301_75_10hz/epoch_12.pth
if [[ "$STAGE2_KD" == "1" ]]; then
    STAGE2_DIR="work_dirs/stage2_etri_split_301_75_10hz_kd$([[ "$STAGE1_KD" == "1" ]] && echo _s1kd)"
    STAGE2_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_kd.py
    STAGE2_EVAL_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_kd.py
    if [[ "$STAGE1_KD" == "1" ]]; then
        # KD-from-epoch-1: stage2 starts from THIS run's own KD stage1 merge,
        # same as the non-KD pipeline's stage1->stage2 handoff.
        STAGE2_LOAD_FROM="$MERGED_CKPT"
    else
        # Warm-start/stage3: continue from the already-converged non-KD
        # stage2 checkpoint, not the stage1 merge -- the config's own
        # hardcoded load_from points at the pre-10Hz path, so this must be
        # overridden explicitly. Fail loudly if that checkpoint doesn't
        # exist yet rather than silently falling back to the merge (which
        # would silently turn this into a from-scratch run under a
        # warm-start work-dir name).
        if [[ ! -f "$NONKD_STAGE2_CKPT" ]]; then
            log "ERROR: warm-start mode (STAGE1_KD=0 STAGE2_KD=1) needs"
            log "$NONKD_STAGE2_CKPT to already exist (run"
            log "tools/run_full_pipeline_split_301_75_10hz.sh first)."
            exit 1
        fi
        STAGE2_LOAD_FROM="$NONKD_STAGE2_CKPT"
    fi
else
    STAGE2_DIR="work_dirs/stage2_etri_split_301_75_10hz$([[ "$STAGE1_KD" == "1" ]] && echo _s1kd)"
    STAGE2_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached.py
    STAGE2_EVAL_CFG=projects/configs/VAD/VADLAW_etri_tiny_cached_eval.py
    STAGE2_LOAD_FROM="$MERGED_CKPT"
fi
STAGE2_CKPT="$STAGE2_DIR/epoch_$([[ "$STAGE2_KD" == "1" ]] && echo 5 || echo 12).pth"

if [[ -f "$STAGE2_CKPT" ]]; then
    log "Stage 2 (KD=$STAGE2_KD) already complete ($STAGE2_CKPT exists), skipping."
else
    log "Stage 2 (KD=$STAGE2_KD): launching, load_from=$STAGE2_LOAD_FROM ..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28822 \
        tools/train.py "$STAGE2_CFG" \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic \
        --cfg-options load_from="$STAGE2_LOAD_FROM" "${ANN_OVERRIDES[@]}"
    log "Stage 2 (KD=$STAGE2_KD): done."
fi

# --- Hold-out planning L2 evaluation ---------------------------------------
EVAL_OUT="$STAGE2_DIR/holdout_l2.txt"
if [[ -f "$EVAL_OUT" ]]; then
    log "Eval output already exists ($EVAL_OUT), skipping. Delete it to rerun."
else
    log "Running hold-out L2 evaluation..."
    python tools/eval_holdout_l2.py \
        "$STAGE2_EVAL_CFG" \
        "$STAGE2_CKPT" \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl" \
        --frame-offsets 0 \
        | tee "$EVAL_OUT"
    log "Eval: done, results in $EVAL_OUT."
fi

# --- Test-set inference / submission ----------------------------------------
SUBMIT_OUT="$STAGE2_DIR/submission.json"
if [[ -f "$SUBMIT_OUT" ]]; then
    log "Submission already exists ($SUBMIT_OUT), skipping."
else
    log "Running test-set inference (1125 clips)..."
    python tools/etri_test_submit.py \
        "$STAGE2_EVAL_CFG" \
        "$STAGE2_CKPT" \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_test.pkl" \
        --frame-offsets 0 \
        --out "$SUBMIT_OUT"
    log "Submission: done, written to $SUBMIT_OUT."
fi

# --- FLOPs measurement -------------------------------------------------------
if grep -q '"__flops__"' "$SUBMIT_OUT" 2>/dev/null; then
    log "FLOPs already recorded in $SUBMIT_OUT, skipping."
else
    log "Measuring FLOPs (single warm frame)..."
    python tools/measure_flops.py \
        "$STAGE2_EVAL_CFG" \
        --ann-file "$DATA_ROOT/vad_etri_infos_temporal_test.pkl" \
        --out "$SUBMIT_OUT"
    log "FLOPs: done, merged into $SUBMIT_OUT."
fi

log "Pipeline complete (STAGE1_KD=$STAGE1_KD, STAGE2_KD=$STAGE2_KD)."
log "  stage1 ckpt:  $STAGE1_CKPT"
log "  stage2 ckpt:  $STAGE2_CKPT"
log "  holdout eval: $EVAL_OUT"
log "  submission:   $SUBMIT_OUT"
