#!/usr/bin/env bash
# stage2 최선 구성. 사용법: ./scripts/run_stage2_best.sh <teacher|student|nodistill|goalgrid>
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
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_best.py
    WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
    INIT=work_dirs/stage1_best_lcfon/stage2_init_merged.pth
    ;;
  student)
    MACHINE=3090 ; PORT=28989
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py
    WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill8_3f_fut_g8
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  teacher-norefine)
    # A5000 의 refine-ON teacher 와 통제된 ablation. 차이는 bev_residual_refine
    # 하나뿐이고, student 는 이긴 쪽에서 증류받으면 되므로 어느 쪽도 안 버려진다.
    MACHINE=3090 ; PORT=28986
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_norefine.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_norefine.py
    WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_norefine
    INIT=work_dirs/stage1_best_lcfon/stage2_init_merged.pth
    ;;
  nodistill)
    # teacher 를 기다리지 않는 제출 가능 모델. teacher 가 도는 동안 3090 이
    # 노는 걸 막고, stage1 재구축이 단독으로 얼마를 벌었는지 재는 유일한 측정이다.
    MACHINE=3090 ; PORT=28987
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_nodistill_best.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_best.py
    WORK_DIR=work_dirs/stage2_nodistill_best
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  nodistill-nodrop-ft)
    # nodistill epoch 12 을 dropout 전부 끄고 2 epoch 이어 학습. dropout 이 만든
    # train/eval 속도 편향(+5%) 수정 검증. config docstring 참조.
    MACHINE=3090 ; PORT=28991
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_nodistill_nodrop_ft.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_best.py
    WORK_DIR=work_dirs/stage2_nodistill_nodrop_ft
    INIT=work_dirs/stage2_nodistill_best/epoch_12.pth
    ;;
  goalgrid-nodrop)
    MACHINE=3090 ; PORT=28992
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_nodistill_goalgrid_nodrop.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_goalgrid.py
    WORK_DIR=work_dirs/stage2_nodistill_goalgrid_nodrop
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  student-nodrop)
    MACHINE=a5000 ; PORT=28993
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8_nodrop.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py
    WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill8_3f_fut_g8_nodrop
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  goalgrid)
    # nodistill 과 goal_grid_size=(7,3) 하나만 다르다. TP 는 목표 칸 라벨로만 쓰고
    # 추론에선 goal_cls_head 가 칸을 고른다 (config docstring 참조).
    MACHINE=3090 ; PORT=28990
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_nodistill_goalgrid.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_goalgrid.py
    WORK_DIR=work_dirs/stage2_nodistill_goalgrid
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  *) echo "사용법: $0 <teacher|student|student-nodrop|nodistill|nodistill-nodrop-ft|teacher-norefine|goalgrid|goalgrid-nodrop>" >&2; exit 1 ;;
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

require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "stage1 병합 도너"
if [ "$ROLE" = student ] || [ "$ROLE" = student-nodrop ]; then
  require_file $MACHINE "work_dirs/stage2_kd_lcfemb8_teacher_best/epoch_12.pth" \
    "증류 teacher 체크포인트 (teacher 를 먼저 돌릴 것)"
fi
launch_train $MACHINE "$CONFIG" "$WORK_DIR" $PORT
verify_start $MACHINE "$WORK_DIR"
