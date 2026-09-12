#!/usr/bin/env bash
# stage2 최선 구성. 사용법: ./scripts/run_stage2_best.sh <teacher|student>
#
#   teacher -> A5000. ego_lcf 직접 입력. 제출 불가. student 의 증류 대상.
#   student -> 3090.  제출 모델. teacher 가 끝나야 시작할 수 있다.
#
# 두 도너 모두 stage1 이 끝난 뒤 merge 로 만들어진다 (README 의 순서 참조).
cd "$(dirname "$0")/.."
source scripts/_common.sh

ROLE="${1:-}"
case "$ROLE" in
  teacher)
    MACHINE=a5000 ; PORT=28988
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py
    WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
    INIT=work_dirs/stage1_best_lcfon/stage2_init_merged.pth
    ;;
  student)
    MACHINE=3090 ; PORT=28989
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py
    WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill8_3f_fut_g8
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  *) echo "사용법: $0 <teacher|student>" >&2; exit 1 ;;
esac

echo "=== stage2 BEST ($ROLE, $MACHINE) ==="

# 긴 학습 전 감사. 이 저장소에서 조용히 틀린 사례가 전부 여기서 걸린다.
echo "--- 감사 ---"
in_container $MACHINE "cd /workspace/VAD && python tools/audit_pipeline.py $CONFIG" \
  || { echo "감사 실패 -- 학습을 시작하지 않는다" >&2; exit 1; }
in_container $MACHINE "cd /workspace/VAD && python tools/check_accel_block_live.py $CONFIG" \
  || { echo "가속도 경로 검사 실패 -- 학습을 시작하지 않는다" >&2; exit 1; }

require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "stage1 병합 도너"
if [ "$ROLE" = student ]; then
  require_file $MACHINE "work_dirs/stage2_kd_lcfemb8_teacher_best/epoch_12.pth" \
    "증류 teacher 체크포인트 (teacher 를 먼저 돌릴 것)"
fi
launch_train $MACHINE "$CONFIG" "$WORK_DIR" $PORT
verify_start $MACHINE "$WORK_DIR"
