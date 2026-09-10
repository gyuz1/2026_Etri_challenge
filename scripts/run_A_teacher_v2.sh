#!/usr/bin/env bash
# A안 teacher v2 — ego 상태를 8차원 임베딩(내부 64)으로. 규정 위반, 제출 불가.
# 선행: run_B_stage1_lcfon.sh 완료 + merge_stage1_world_model.py 로 stage2_init_merged.pth 생성
#
# v1(64차원)과의 차이: 출력을 8로 줄여 ego_feats 를 520 으로 맞춘다.
# 그래야 ego_lcf-ON stage1 이 48epoch 학습시킨 decoder 를 zero-pad 없이 상속받는다.
# 내부 폭은 64 로 유지하므로 비선형 표현력은 그대로.
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=3090
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher.py
WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon/stage2_init_merged.pth

echo "=== A안 teacher v2 (ego_lcf -> 8차원 임베딩, 내부 64) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "ego_lcf-ON stage1 병합 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28984
verify_start $MACHINE "$WORK_DIR"
