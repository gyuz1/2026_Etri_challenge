#!/usr/bin/env bash
# B안 teacher — ego_lcf 를 raw 8칸으로 넣는다 (구조 변경 없음). A5000.
# 제출 불가. distillation 소스 전용.
# 산출물: work_dirs/stage2_kd_lcfon_diag/epoch_12.pth  → run_B_student.sh 가 사용
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=a5000
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfon_diag.py
WORK_DIR=work_dirs/stage2_kd_lcfon_diag
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfon8.pth

echo "=== B안 teacher (ego_lcf raw 8칸) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "초기 체크포인트 (8칸 zero-pad)"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28971
verify_start $MACHINE "$WORK_DIR"
