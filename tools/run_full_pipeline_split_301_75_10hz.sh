#!/usr/bin/env bash
# 10Hz variant of run_full_pipeline_split_301_75.sh -- same 301/75 scene
# split, but ego-motion (can_bus/ego_lcf_feat/STOP-derivation/PRISM long_fut)
# computed from dense 10Hz-spaced pose instead of TRAJ_STEP(0.5s)-spaced,
# per the organizers' 2026-08-25 Q&A confirming pose used for this calc
# doesn't need a matching fed image. Only data.{train,val,test}.ann_file
# changes vs the 2Hz configs -- model architecture is identical, so this
# overrides ann_file via --cfg-options instead of new config files.
#
# Geometry cache (image crops) is reused as-is from the 2Hz run's
# work_dirs/etri_geometry_cache_v1 / _val_v1 -- it's a pure function of
# images + calibration, never ego-motion (same reuse precedent as
# fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata_10hz.py). The
# configs' own LoadETRIGeometryCache.expected_ann_file stays pointed at the
# ORIGINAL 2Hz pkl path (matches what the cache manifest was actually built
# against) -- do not change that, only the ann_file overrides below.
#
# Supersedes the 2Hz split_301_75 run, which was stopped at stage1 epoch 29
# once the decision to switch to 10Hz was made (checkpoint kept at
# work_dirs/stage1_etri_split_301_75/epoch_29.pth as a reference, not
# reused here -- shapes are compatible but starting fresh on the 10Hz pkl
# is simpler than reasoning about partial-epoch warm-start effects).
#
# Adds a 7th stage vs the 2Hz script: the 6-condition target_point-shortcut
# / BEV-refine diagnostic suite (tools/eval_holdout_l2_shortcut_ablation.py),
# split across both GPUs (3 pairs of 2 conditions each). Expected to show
# near-zero target_point sensitivity now that generation never sees it --
# this is a compliance confirmation, not just a hope that the redesign
# worked.
#
# Meant to be run inside the ad2026-derived gyuz_split container, from
# /workspace/VAD:
#   docker exec gyuz_split bash -c 'cd /workspace/VAD && nohup bash \
#       tools/run_full_pipeline_split_301_75_10hz.sh \
#       > work_dirs/full_pipeline_split_301_75_10hz.log 2>&1 &'
#
# Each stage blocks until it finishes (or fails) before the next one starts.
# Safe to re-run after an interruption -- already-finished stages are
# detected by their output files and skipped.
set -euo pipefail
cd /workspace/VAD

DATA_ROOT=data/etri/.causal_regen_split_301_75_10hz
STAGE1_DIR=work_dirs/stage1_etri_split_301_75_10hz
STAGE1_CKPT="$STAGE1_DIR/epoch_48.pth"
MERGED_CKPT="$STAGE1_DIR/stage2_init_merged.pth"
STAGE2_DIR=work_dirs/stage2_etri_split_301_75_10hz
STAGE2_CKPT="$STAGE2_DIR/epoch_12.pth"
EVAL_OUT=work_dirs/stage2_etri_split_301_75_10hz_holdout_l2.txt
SUBMIT_OUT=work_dirs/stage2_etri_split_301_75_10hz_submission.json
TINFER_OUT=work_dirs/stage2_etri_split_301_75_10hz_t_infer.log
ABLATION_OUT=work_dirs/stage2_etri_split_301_75_10hz_shortcut_ablation
NVME_TEST_DIR=/data_fast/test

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ANN_OVERRIDES=(
    data.train.ann_file="$DATA_ROOT/vad_etri_infos_temporal_train_split.pkl"
    data.val.ann_file="$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl"
    data.test.ann_file="$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl"
)

# --- Stage 1: VAD-only, ETRI domain adaptation, 48 epochs, 10Hz motion -----
if [[ -f "$STAGE1_CKPT" ]]; then
    log "Stage 1 already complete ($STAGE1_CKPT exists), skipping."
else
    log "Stage 1: launching (48 epochs, 10Hz ego-motion)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28811 \
        tools/train.py projects/configs/VAD/VAD_etri_tiny_stage1_cached.py \
        --launcher pytorch --work-dir "$STAGE1_DIR" \
        --no-validate --deterministic \
        --cfg-options "${ANN_OVERRIDES[@]}"
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

# --- Stage 2: VADLAW, waypoint + world model, 12 epochs, PRISM enabled ----
if [[ -f "$STAGE2_CKPT" ]]; then
    log "Stage 2 already complete ($STAGE2_CKPT exists), skipping."
else
    log "Stage 2: launching (12 epochs, 10Hz ego-motion, PRISM enabled)..."
    python -m torch.distributed.launch --nproc_per_node=2 --master_port=28812 \
        tools/train.py projects/configs/VAD/VADLAW_etri_tiny_cached.py \
        --launcher pytorch --work-dir "$STAGE2_DIR" \
        --no-validate --deterministic \
        --cfg-options load_from="$MERGED_CKPT" "${ANN_OVERRIDES[@]}"
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

# --- FLOPs measurement -------------------------------------------------------
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
if [[ -f "$TINFER_OUT" ]]; then
    log "T_infer measurement already exists ($TINFER_OUT), skipping. Delete it to rerun."
else
    log "Measuring T_infer (1125 clips, NVMe test copy)..."
    if [[ ! -d "$NVME_TEST_DIR" ]]; then
        log "ERROR: $NVME_TEST_DIR not found -- copy dataset/test there first " \
            "(mkdir -p /data_fast && cp -r data/test /data_fast/test)."
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

# --- 6-condition target_point-shortcut / BEV-refine diagnostic suite -------
# Split across both GPUs, 2 conditions at a time. Post-redesign, conditions
# 3/4 (zero/shuffle target_point) are expected to show ~0 change vs
# baseline -- that's the point: confirming target_point no longer has a
# generation-path shortcut, not measuring how much it used to matter.
if [[ -f "$ABLATION_OUT/summary.txt" ]]; then
    log "Shortcut ablation suite already complete, skipping."
else
    log "Running 6-condition shortcut ablation suite (2 GPUs)..."
    mkdir -p "$ABLATION_OUT"

    run_cond() {
        local name=$1 device=$2
        shift 2
        python3 tools/eval_holdout_l2_shortcut_ablation.py \
            projects/configs/VAD/VADLAW_etri_tiny_cached_eval.py \
            "$STAGE2_CKPT" \
            --ann-file "$DATA_ROOT/vad_etri_infos_temporal_val_split.pkl" \
            --device "$device" "$@" > "$ABLATION_OUT/${name}.log" 2>&1
    }

    run_cond 01_baseline 0 & run_cond 02_zero_images 1 --zero-images &
    wait
    run_cond 03_zero_target_point 0 --zero-target-point & \
        run_cond 04_shuffle_target_point 1 --shuffle-target-point &
    wait
    run_cond 05_disable_bev_refine 0 --disable-bev-refine & \
        run_cond 06_zero_bev_sampled 1 --zero-bev-sampled &
    wait

    SUMMARY="$ABLATION_OUT/summary.txt"
    echo "condition,L2_avg(1s/2s/3s),L2_far(last2wp)" > "$SUMMARY"
    for f in 01_baseline 02_zero_images 03_zero_target_point \
             04_shuffle_target_point 05_disable_bev_refine 06_zero_bev_sampled; do
        L2AVG=$(grep -oP 'Final Planning L2 avg.*: \K[0-9.]+' "$ABLATION_OUT/${f}.log" || echo "N/A")
        L2FAR=$(grep -oP 'Far-only L2.*: \K[0-9.]+' "$ABLATION_OUT/${f}.log" || echo "N/A")
        echo "${f},${L2AVG},${L2FAR}" >> "$SUMMARY"
    done
    log "Shortcut ablation suite: done."
    cat "$SUMMARY"
fi

log "Pipeline complete."
