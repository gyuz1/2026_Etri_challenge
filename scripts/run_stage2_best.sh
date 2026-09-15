#!/usr/bin/env bash
# stage2 최선 구성. 사용법: ./scripts/run_stage2_best.sh <student-clean|nodistill-clean|goalgrid-clean>
#
#   student-clean   -> A5000. 증류 student (제출 모델 후보). teacher epoch_12 필요.
#   nodistill-clean -> 3090.  증류 없는 대조군 (제출 가능).
#   goalgrid-clean  -> 3090.  TP 목표 격자 5x5 (nodistill-clean 과 격자만 다름).
#   기본 머신 변경: MACHINE_OVERRIDE=3090 ./scripts/run_stage2_best.sh student-clean
cd "$(dirname "$0")/.."
source scripts/_common.sh

ROLE="${1:-}"
case "$ROLE" in
  # 모든 역할은 audit 3c(학습/추론 불일치 설정 금지)를 통과해야 시작된다.
  # 2026-09-15 이전 역할(teacher/student/nodistill/goalgrid 등)은 dropout 등이
  # 켜진 config 라 삭제했다. 결과물은 work_dirs 에 남아 있고 eval_l2.sh 로 평가된다.
  student-clean)
    # 증류 student. teacher(stage2_kd_lcfemb8_teacher_best/epoch_12)가 그 머신에 있어야 한다.
    MACHINE=a5000 ; PORT=28993
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_clean_student.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py
    WORK_DIR=work_dirs/stage2_clean_student
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  nodistill-clean)
    MACHINE=3090 ; PORT=28994
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_clean_nodistill.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py
    WORK_DIR=work_dirs/stage2_clean_nodistill
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  goalgrid-clean)
    # nodistill-clean + goal_grid 5x5 (TP 라벨, 커맨드별 선택). config docstring 참조.
    MACHINE=3090 ; PORT=28995
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_clean_goalgrid.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalgrid.py
    WORK_DIR=work_dirs/stage2_clean_goalgrid
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  *) echo "사용법: $0 <student-clean|nodistill-clean|goalgrid-clean>" >&2; exit 1 ;;
esac

# 기본 머신을 바꿀 때: MACHINE_OVERRIDE=a5000 ./scripts/run_stage2_best.sh student
# (도너·teacher 체크포인트가 그 머신에 있어야 한다 -- require_file 이 확인한다)
MACHINE="${MACHINE_OVERRIDE:-$MACHINE}"

echo "=== stage2 BEST ($ROLE, $MACHINE) ==="

# 긴 학습 전 감사. 이 저장소에서 조용히 틀린 사례가 전부 여기서 걸린다.
echo "--- 감사 ---"
in_container $MACHINE "cd /workspace/VAD && python tools/audit_pipeline.py $CONFIG --eval-config $EVAL_CONFIG" \
  || { echo "감사 실패 -- 학습을 시작하지 않는다" >&2; exit 1; }
in_container $MACHINE "cd /workspace/VAD && python tools/check_accel_block_live.py $CONFIG" \
  || { echo "가속도 경로 검사 실패 -- 학습을 시작하지 않는다" >&2; exit 1; }

if grep -q "goal_grid_size" "$CONFIG"; then
  # 실제 학습 batch 로 TP 라벨·손실·초기 출력·추론 게이트를 확인. GridMask 가 cuda:0 에
  # mask 를 만들므로 GPU 하나만 보이게 한다.
  in_container $MACHINE "cd /workspace/VAD && CUDA_VISIBLE_DEVICES=0 python tools/check_goal_grid_live.py $CONFIG --n 6 --device 0" \
    || { echo "goal grid 실측 검사 실패 -- 학습을 시작하지 않는다" >&2; exit 1; }
fi
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "stage1 병합 도너"
if [ "$ROLE" = student-clean ]; then
  require_file $MACHINE "work_dirs/stage2_kd_lcfemb8_teacher_best/epoch_12.pth" \
    "증류 teacher 체크포인트 (teacher 를 먼저 돌릴 것)"
fi
launch_train $MACHINE "$CONFIG" "$WORK_DIR" $PORT
verify_start $MACHINE "$WORK_DIR"
