#!/usr/bin/env bash
# A안 student + 새 descriptor 스택 — 기존 teacher v1(0.2328) 로 통제된 A/B.
# v1 student 0.4218 대비 descriptor 관련 설정만 다르다.
cd "$(dirname "$0")/.."
source scripts/_common.sh
MACHINE=a5000
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_split_distill_newstack.py
WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill_newstack
TEACHER=work_dirs/stage2_kd_lcfemb_teacher/epoch_12.pth
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfemb64.pth
echo "=== A student (새 스택: 3프레임 + grid8 + 가속도 + 미래속도) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$TEACHER" "A teacher v1 체크포인트"
require_file $MACHINE "$INIT" "student 초기 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28987
verify_start $MACHINE "$WORK_DIR"
