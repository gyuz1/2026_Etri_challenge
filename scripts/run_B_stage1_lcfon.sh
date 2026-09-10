#!/usr/bin/env bash
# B안 teacher 계보 수정 — stage1부터 ego_lcf ON + Qwen KD (48 epoch, ~2일)
#
# 왜: 현재 B안 teacher(0.2542m)는 512폭 KD decoder에 8칸을 zero-pad로 붙여
# stage2 12 epoch 동안만 학습시킨다. 0.2166 기록을 낸 옛 계보는 stage1부터
# 520폭이었다. 이 스크립트는 그 성질만 현재 아키텍처에서 재현한다.
# 자세한 근거는 config docstring과 SHARED_CONTEXT.md 참조.
#
# 완료 후: tools/merge_stage1_world_model.py 로 stage2_init_merged.pth 를 만들고
# (zero-pad surgery 불필요 — 이미 520폭), VADLAW_etri_tiny_kd_lcfon_diag.py 의
# load_from 을 그쪽으로 바꿔 stage2 를 다시 돌린다.
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=a5000
CONFIG=projects/configs/VAD/VAD_etri_tiny_stage1_cached_kd_lcfon.py
WORK_DIR=work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon

echo "=== B안 stage1 재학습 (ego_lcf ON + KD, 48 epoch) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$CONFIG" "stage1 config"
# KD teacher 캐시가 없으면 loss_plan_kd 가 조용히 죽는다. stage1 에서
# ego_fut_decoder 를 학습시키는 유일한 신호이므로 없으면 48시간이 통째로 낭비된다.
require_file $MACHINE "work_dirs/teacher_cache/etri_train_teacher_cache.json" \
    "Qwen teacher 궤적 캐시 (loss_plan_kd 의 유일한 감독 신호)"

launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28983
verify_start $MACHINE "$WORK_DIR"
