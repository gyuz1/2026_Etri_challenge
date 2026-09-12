#!/usr/bin/env bash
# A안 teacher — 지금까지 측정된 최선 구성 전부 적용.
cd "$(dirname "$0")/.."
source scripts/_common.sh
MACHINE=${MACHINE:-3090}
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py
WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon/stage2_init_merged.pth
echo "=== A teacher BEST ($MACHINE) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "ego_lcf-ON stage1 병합 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28988
verify_start $MACHINE "$WORK_DIR"
