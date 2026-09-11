#!/usr/bin/env bash
# A안 teacher v3 — 3프레임 motion descriptor (가속도 관측 가능). 3090.
cd "$(dirname "$0")/.."
source scripts/_common.sh
MACHINE=3090
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_3f.py
WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_3f
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon/stage2_init_merged.pth
echo "=== A teacher v3 (3프레임, 가속도 관측) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "ego_lcf-ON stage1 병합 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28986
verify_start $MACHINE "$WORK_DIR"
