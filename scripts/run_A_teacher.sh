#!/usr/bin/env bash
# A안 teacher — ego_lcf 를 학습된 64차원 임베딩으로 넣는다. 3090.
# 제출 불가 (ego_lcf 를 planner 에 직접 사용). distillation 소스 전용.
# 산출물: work_dirs/stage2_kd_lcfemb_teacher/epoch_12.pth  → run_A_student.sh 가 사용
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=3090
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb_teacher.py
WORK_DIR=work_dirs/stage2_kd_lcfemb_teacher
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfemb64.pth

echo "=== A안 teacher (ego_lcf → 64d 임베딩) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "초기 체크포인트 (64칸 zero-pad)"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28981
verify_start $MACHINE "$WORK_DIR"
