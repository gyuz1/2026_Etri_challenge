#!/usr/bin/env bash
# B안 student — 규정 준수. 디코더 첫 Linear 직후 은닉값 512 를 teacher 와 맞춘다.
# 디코더 입력은 순수 vision 512 그대로. 추정 채널을 만들지 않는다.
# 선행: run_B_teacher.sh 완료 (epoch_12.pth 필요)
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=a5000
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_feature_distill.py
WORK_DIR=work_dirs/stage2_kd_nolcf_feature_distill
TEACHER=work_dirs/stage2_kd_lcfon_diag/epoch_12.pth
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged.pth

echo "=== B안 student (융합 은닉값 증류) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$TEACHER" "B안 teacher 체크포인트"
require_file $MACHINE "$INIT" "초기 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28972
verify_start $MACHINE "$WORK_DIR"
