#!/usr/bin/env bash
# Usage: ./scripts/run_stage2_best.sh <student-clean|nodistill-clean|goalpred-clean|cellplanner>
#
#   student-clean   -> A5000. Distilled student; needs the teacher's epoch_12.
#   nodistill-clean -> 3090.  Control without distillation (submittable).
#   goalpred-clean  -> 3090.  Plans toward a predicted 5s goal; differs from
#                             nodistill-clean only in goal_pred.
#   cellplanner     -> 3090.  Command cell planner, target point selects only.
#   MACHINE_OVERRIDE=<3090|a5000> picks a different machine.
cd "$(dirname "$0")/.."
source scripts/_common.sh

ROLE="${1:-}"
case "$ROLE" in
  # Every role must pass audit 3c (no train/inference mismatch settings). Roles
  # from before 2026-09-15 were deleted because their configs left dropout and
  # friends on; their checkpoints remain in work_dirs for eval_l2.sh.
  student-clean)
    # Distilled student; the teacher checkpoint must be on that machine.
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
  goalpred-clean)
    # nodistill-clean + predicted 5s goal. The target point is a label only.
    MACHINE=3090 ; PORT=28995
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_clean_goalpred.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalpred.py
    WORK_DIR=work_dirs/stage2_clean_goalpred
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  cellplanner)
    # nodistill-clean with the planner output replaced by command cell heads.
    MACHINE=3090 ; PORT=28998
    CONFIG=projects/configs/VAD/VADLAW_etri_tiny_clean_cellplanner.py
    EVAL_CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_cellplanner.py
    WORK_DIR=work_dirs/stage2_cellplanner_v1
    INIT=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth
    ;;
  *) echo "usage: $0 <student-clean|nodistill-clean|goalpred-clean|cellplanner>" >&2; exit 1 ;;
esac

# MACHINE_OVERRIDE picks a different machine; require_file then checks that the
# donor and teacher checkpoints are actually there.
MACHINE="${MACHINE_OVERRIDE:-$MACHINE}"

echo "=== stage2 BEST ($ROLE, $MACHINE) ==="

# Pre-training audit: every silent failure this repo has had is caught here.
echo "--- audit ---"
in_container $MACHINE "cd /workspace/VAD && python tools/audit_pipeline.py $CONFIG --eval-config $EVAL_CONFIG" \
  || { echo "audit failed -- not starting training" >&2; exit 1; }
in_container $MACHINE "cd /workspace/VAD && python tools/check_accel_block_live.py $CONFIG" \
  || { echo "acceleration-path check failed -- not starting training" >&2; exit 1; }

if grep -q "goal_pred=True" "$CONFIG"; then
  # Checks the target-point labels, losses, initial output and inference gate on
  # real training batches. GridMask builds its mask on cuda:0, so expose one GPU.
  in_container $MACHINE "cd /workspace/VAD && CUDA_VISIBLE_DEVICES=0 python tools/check_goal_pred_live.py $CONFIG --n 6 --device 0" \
    || { echo "goal_pred live check failed -- not starting training" >&2; exit 1; }
fi
if grep -q "cell_planner=True" "$CONFIG"; then
  # Selection rule, init from the donor, target-point invariance and backward
  # on real batches.
  in_container $MACHINE "cd /workspace/VAD && CUDA_VISIBLE_DEVICES=0 python tools/check_cell_planner_live.py $CONFIG --n 4 --device 0" \
    || { echo "cell planner live check failed -- not starting training" >&2; exit 1; }
fi
require_gpu_free $MACHINE
require_file $MACHINE "$INIT" "the merged stage-1 donor"
if [ "$ROLE" = student-clean ]; then
  require_file $MACHINE "work_dirs/stage2_kd_lcfemb8_teacher_best/epoch_12.pth" \
    "the distillation teacher checkpoint (train the teacher first)"
fi
launch_train $MACHINE "$CONFIG" "$WORK_DIR" $PORT
verify_start $MACHINE "$WORK_DIR"
