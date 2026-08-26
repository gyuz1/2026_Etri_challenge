#!/usr/bin/env bash
# Measures T_infer (model.forward()-only, per the 2026-08-25 organizer Q&A)
# for the actual submission inference path: 1-frame, no prev_bev
# (--frame-offsets 0), matching what etri_test_submit.py uses.
#
# Usage:
#   bash tools/run_tinfer_measurement.sh CONFIG CHECKPOINT ANN_FILE [NUM_CLIPS] [DEVICE]
#
# ANN_FILE should be the TEST set's temporal-info pkl (the one
# etri_test_submit.py reads), not the val_split pkl -- T_infer must be
# measured on the same clips/format used for the real submission.

set -e

CONFIG=$1
CHECKPOINT=$2
ANN_FILE=$3
NUM_CLIPS=${4:-100}
DEVICE=${5:-0}

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$ANN_FILE" ]; then
    echo "Usage: bash tools/run_tinfer_measurement.sh CONFIG CHECKPOINT ANN_FILE [NUM_CLIPS] [DEVICE]"
    exit 1
fi

python3 tools/measure_t_infer_fwd_only.py \
    "$CONFIG" "$CHECKPOINT" \
    --ann-file "$ANN_FILE" \
    --frame-offsets 0 \
    --num-clips "$NUM_CLIPS" \
    --warmup-clips 5 \
    --device "$DEVICE"
