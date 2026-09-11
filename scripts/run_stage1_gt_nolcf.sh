#!/usr/bin/env bash
# stage1 GT-only (Qwen KD 끔) — 기존 KD stage1 과의 A/B.
# 차이는 두 줄뿐: kd_weight 0.2->0, loss_plan_reg 0.0->1.0.
# 근거는 config docstring 참조.
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=3090
CONFIG=projects/configs/VAD/VAD_etri_tiny_stage1_cached_gt_nolcf.py
WORK_DIR=work_dirs/stage1_etri_split_301_75_10hz_gt_nolcf

echo "=== stage1 GT-only (Qwen KD off) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$CONFIG" "stage1 GT config"

launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28985
verify_start $MACHINE "$WORK_DIR"
