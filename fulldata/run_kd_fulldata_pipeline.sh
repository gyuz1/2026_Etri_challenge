#!/usr/bin/env bash
# EvoDriveVLA-teacher distillation on top of the fulldata (376-scene, no
# train/val split) stage2 checkpoint. Sibling of run_full_pipeline_kd.sh
# (the 301/75-split KD track) and run_fulldata_pipeline.sh (the plain
# fulldata track, no KD) -- does NOT touch either script's files or
# work_dirs, safe to prepare/launch without disturbing a run already in
# progress under those.
#
# Only STAGE2_KD exists here (no STAGE1_KD toggle): stage1 is always reused
# as-is from work_dirs/stage1_etri_fulldata/epoch_48.pth (produced by
# run_fulldata_pipeline.sh) -- this script does not retrain it, to avoid
# duplicating that ~40h job by accident. No hold-out L2 step either: unlike
# run_full_pipeline_kd.sh, there is no val split left once every scene is
# in the fulldata training set, so that step is dropped entirely rather
# than adapted.
#
#   docker exec -it ad2026 bash
#   cd /workspace/VAD
#   nohup bash fulldata/run_kd_fulldata_pipeline.sh \
#       > work_dirs/kd_fulldata_pipeline.log 2>&1 &
#
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped.
set -euo pipefail
cd /workspace/VAD

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

STAGE1_CKPT=work_dirs/stage1_etri_fulldata/epoch_48.pth
STAGE2_CKPT=work_dirs/stage2_etri_fulldata/epoch_12.pth
TEACHER_CACHE=work_dirs/teacher_cache/etri_train_teacher_cache_fulldata.json

if [[ ! -f "$STAGE1_CKPT" ]]; then
    log "ERROR: $STAGE1_CKPT not found. Run fulldata/run_fulldata_pipeline.sh first."
    exit 1
fi
if [[ ! -f "$STAGE2_CKPT" ]]; then
    log "ERROR: $STAGE2_CKPT not found. This script warm-starts from the"
    log "already-converged (non-KD) fulldata stage2 checkpoint -- run"
    log "fulldata/run_fulldata_pipeline.sh first (through its stage2 step)."
    exit 1
fi
if [[ ! -f "$TEACHER_CACHE" ]]; then
    log "ERROR: $TEACHER_CACHE not found."
    log "Run evodrive_etri_prep/generate_teacher_cache.py first, pointed at"
    log "the FULLDATA train ann-file (vad_etri_infos_fulldata_train.pkl) --"
    log "the 301/75-split teacher cache is not a substitute, see"
    log "fulldata/configs/VADLAW_etri_tiny_cached_kd_fulldata.py's docstring."
    exit 1
fi

# --- Stage 2 (KD): short fine-tune of the fulldata stage2 checkpoint ------
STAGE2_KD_DIR=work_dirs/stage2_etri_fulldata_kd
STAGE2_KD_CKPT="$STAGE2_KD_DIR/epoch_5.pth"

if [[ -f "$STAGE2_KD_CKPT" ]]; then
    log "Stage 2 (KD) already complete ($STAGE2_KD_CKPT exists), skipping."
else
    log "Stage 2 (KD): launching (5 epochs, lr=1e-5, fulldata)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28813 \
        tools/train.py fulldata/configs/VADLAW_etri_tiny_cached_kd_fulldata.py \
        --launcher pytorch --work-dir "$STAGE2_KD_DIR" \
        --no-validate --deterministic
    log "Stage 2 (KD): done."
fi

# --- Test-set inference / submission ----------------------------------------
SUBMIT_OUT=work_dirs/stage2_etri_fulldata_kd_submission.json
TEST_ANN=data/etri/annotations/vad_etri_infos_fulldata_test.pkl

if [[ -f "$SUBMIT_OUT" ]]; then
    log "Submission output already exists ($SUBMIT_OUT), skipping. Delete it to rerun."
else
    log "Running test-set inference (1125 clips, live fast-eval pipeline)..."
    python tools/etri_test_submit.py \
        projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "$STAGE2_KD_CKPT" \
        --ann-file "$TEST_ANN" \
        --out "$SUBMIT_OUT"
    log "Test inference: done, submission in $SUBMIT_OUT."
fi

# --- FLOPs measurement ------------------------------------------------------
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

# --- T_infer measurement -----------------------------------------------------
# Model-forward-only, per the 2026-08-25 organizer Q&A -- see
# measure_t_infer_fwd_only.py's docstring. Same NVMe-copy caveat as
# fulldata/run_fulldata_pipeline.sh: point NVME_TEST_DIR at real fast
# local storage or this number is meaningless.
TINFER_OUT=work_dirs/stage2_etri_fulldata_kd_t_infer.log
NVME_TEST_DIR=/data_fast/test

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
    ( cd /data_fast && python -u /workspace/VAD/tools/measure_t_infer_fwd_only.py \
        /workspace/VAD/projects/configs/VAD/VADLAW_etri_tiny_fast_eval.py \
        "/workspace/VAD/$STAGE2_KD_CKPT" \
        --ann-file "/workspace/VAD/$TEST_ANN" \
        --frame-offsets=-30,-25,-20,-15,-10,-5,0 \
        --fp16 --warmup-clips 5 --num-clips 1120 ) | tee "$TINFER_OUT"
    log "T_infer measurement: done, results in $TINFER_OUT."
fi

log "Pipeline complete."
log "  stage2 KD ckpt: $STAGE2_KD_CKPT"
log "  submission:     $SUBMIT_OUT"
log "  t_infer log:    $TINFER_OUT"
