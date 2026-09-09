#!/usr/bin/env bash
# A안 student — 규정 준수. 512(장면)/64(자기운동) 를 각각 teacher 와 맞춘다.
# 64칸은 vision 에서 추정한 값이며 modality dropout 으로 과의존을 막는다.
# 선행: run_A_teacher.sh 완료 (epoch_12.pth 필요)
cd "$(dirname "$0")/.."
source scripts/_common.sh

MACHINE=3090
CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_split_distill.py
WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill
TEACHER=work_dirs/stage2_kd_lcfemb_teacher/epoch_12.pth
# teacher 와 같은 도너를 쓴다. 둘 다 decoder 첫 Linear 가 576->512 이고
# 도너의 마지막 64칸은 0 으로 패딩되어 있다 (확인함) — 별도 surgery 불필요.
INIT=work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfemb64.pth

echo "=== A안 student (분리 증류 + modality dropout) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$TEACHER" "A안 teacher 체크포인트"
require_file $MACHINE "$INIT" "초기 체크포인트 (64칸 zero-pad)"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" 28982
verify_start $MACHINE "$WORK_DIR"
