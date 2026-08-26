#!/usr/bin/env bash
# Runs all 6 target_point-shortcut / BEV-refine diagnostic conditions
# against one checkpoint in sequence, via eval_holdout_l2_shortcut_ablation.py:
#   1. baseline (no corruption)
#   2. --zero-images
#   3. --zero-target-point
#   4. --shuffle-target-point
#   5. --disable-bev-refine
#   6. --zero-bev-sampled
#
# Usage:
#   bash tools/run_shortcut_ablation_suite.sh CONFIG CHECKPOINT ANN_FILE [OUT_DIR] [DEVICE]
#
# ANN_FILE should be the val_split pkl (hold-out, has gt_ego_fut_trajs).
# OUT_DIR defaults to work_dirs/shortcut_ablation_suite. Logs are written
# there as 01_baseline.log .. 06_zero_bev_sampled.log, plus a summary.txt
# with each run's "Final Planning L2 avg" / "Far-only L2" lines pulled out.

set -e

CONFIG=$1
CHECKPOINT=$2
ANN_FILE=$3
OUT_DIR=${4:-work_dirs/shortcut_ablation_suite}
DEVICE=${5:-0}

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$ANN_FILE" ]; then
    echo "Usage: bash tools/run_shortcut_ablation_suite.sh CONFIG CHECKPOINT ANN_FILE [OUT_DIR] [DEVICE]"
    exit 1
fi

mkdir -p "$OUT_DIR"

run_one() {
    NAME=$1
    shift
    LOG="$OUT_DIR/${NAME}.log"
    echo "=== [$NAME] ==="
    python3 tools/eval_holdout_l2_shortcut_ablation.py \
        "$CONFIG" "$CHECKPOINT" --ann-file "$ANN_FILE" --device "$DEVICE" "$@" \
        2>&1 | tee "$LOG"
}

run_one 01_baseline
run_one 02_zero_images --zero-images
run_one 03_zero_target_point --zero-target-point
run_one 04_shuffle_target_point --shuffle-target-point
run_one 05_disable_bev_refine --disable-bev-refine
run_one 06_zero_bev_sampled --zero-bev-sampled

SUMMARY="$OUT_DIR/summary.txt"
echo "condition,L2_avg(1s/2s/3s),L2_far(last2wp)" > "$SUMMARY"
for f in 01_baseline 02_zero_images 03_zero_target_point 04_shuffle_target_point 05_disable_bev_refine 06_zero_bev_sampled; do
    L2AVG=$(grep -oP 'Final Planning L2 avg.*: \K[0-9.]+' "$OUT_DIR/${f}.log" || echo "N/A")
    L2FAR=$(grep -oP 'Far-only L2.*: \K[0-9.]+' "$OUT_DIR/${f}.log" || echo "N/A")
    echo "${f},${L2AVG},${L2FAR}" >> "$SUMMARY"
done

echo
echo "=== Summary ($SUMMARY) ==="
cat "$SUMMARY"
