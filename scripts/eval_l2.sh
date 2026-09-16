#!/usr/bin/env bash
# Usage: [EVAL_GPU=0] ./scripts/eval_l2.sh <student-clean|nodistill-clean|goalpred-clean|goalpred-cand|teacher|student|nodistill|nodrop|A:student> [epoch] [flags...]
#
#   Measures hold-out L2 + T_infer. epoch defaults to 12. Evaluation runs on the
#   3090 by default; EVAL_MACHINE=a5000 sends it to the A5000 (both machines now
#   have the val images). A checkpoint must exist on whichever machine runs it.
#   Flags are passed straight to eval_holdout_l2_and_tinfer.py and appear in the log name.
#   --test-commands : test-time conditions. Drops val's STOP command (a label
#                     derived from the training pkl) and picks STOP from the
#                     model's own speed estimate. This is the submission number.
#   --zero-ego-lcf  : control that zeroes the privileged input, to verify a
#                     teacher really uses ego_lcf (if it does not, the number
#                     barely moves).
#
# The frame count is read from the config. This is a correctness requirement,
# not a convenience: scoring an aux_bev_motion_frames=3 model in a 2-frame
# window leaves prev_bev2 None throughout, so the acceleration block is zero and
# the model sees different inputs than training gave it. L2 gets quietly worse
# and nothing in the log says why.
cd "$(dirname "$0")/.."
source scripts/_common.sh

TARGET="${1:-}"; EPOCH="${2:-12}"; shift 2 2>/dev/null || shift $#
EXTRA="$*"
GPU="${EVAL_GPU:-0}"
case "$TARGET" in
  student-clean)   WORK_DIR=work_dirs/stage2_clean_student
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py ;;
  goalpred-clean)  WORK_DIR=work_dirs/stage2_clean_goalpred
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalpred.py ;;
  goalpred-cand)   WORK_DIR=work_dirs/stage2_clean_goalpred
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalpred_cand.py ;;
  nodistill-clean) WORK_DIR=work_dirs/stage2_clean_nodistill
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py ;;
  # Trained before 2026-09-15 (with dropout and other mismatch settings on);
  # kept for comparison and fallback.
  teacher)   WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_best.py ;;
  student)   WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill8_3f_fut_g8
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py ;;
  nodistill) WORK_DIR=work_dirs/stage2_nodistill_best
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_best.py ;;
  nodrop)    WORK_DIR=work_dirs/stage2_nodistill_nodrop_ft
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_best.py ;;
  A:student) WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill.py ;;
  *) echo "usage: $0 <student-clean|nodistill-clean|goalpred-clean|goalpred-cand|teacher|student|nodistill|nodrop|A:student> [epoch] [flags...]" >&2
     exit 1 ;;
esac

# 2026-09-16: the A5000 now has the val scene images too, so it can evaluate.
# EVAL_MACHINE=a5000 sends the job there (the checkpoint must be on that machine).
MACHINE="${EVAL_MACHINE:-3090}"
CKPT="$WORK_DIR/epoch_${EPOCH}.pth"

# Read the frame count the config needs and size the window from it. An explicit
# FRAME_OFFSETS wins, but a window shorter than the config needs is refused.
# The configs are identical on both machines, so always read on the 3090: over
# ssh -> docker exec the A5000 strips one layer of quoting and python -c breaks
# (the same class of bug as verify_start had).
FRAMES=$(in_container 3090 "cd /workspace/VAD && python -c \"
import mmcv; print(mmcv.Config.fromfile('$CONFIG').model.pts_bbox_head.get('aux_bev_motion_frames') or 2)
\" 2>/dev/null" | tr -d '\r' | tail -1)
case "$FRAMES" in
  2) DEFAULT_OFFSETS=0,-5 ;;
  3) DEFAULT_OFFSETS=0,-5,-10 ;;
  *) echo "could not read aux_bev_motion_frames: '$FRAMES'" >&2; exit 1 ;;
esac
OFFSETS="${FRAME_OFFSETS:-$DEFAULT_OFFSETS}"
N_OFF=$(awk -F, '{print NF}' <<<"$OFFSETS")
if [ "$N_OFF" -lt "$FRAMES" ]; then
  echo "refused: the config wants ${FRAMES} frames but the window has ${N_OFF} ($OFFSETS)." >&2
  echo "  the acceleration block would be scored as zero. Give at least ${FRAMES} frames." >&2
  exit 1
fi

# One log per flag set. This used to append _zerolcf whatever the flag was, so a
# --test-commands result was about to be saved under the control's name.
TAG=$(echo "$EXTRA" | sed 's/--//g; s/[^A-Za-z0-9.-]\+/_/g; s/^_//; s/_$//')
OUT="$WORK_DIR/eval_l2_ep${EPOCH}_${N_OFF}frame${TAG:+_$TAG}.log"

echo "=== $TARGET eval (epoch $EPOCH, ${N_OFF} frames $OFFSETS) ==="
require_file $MACHINE "$CKPT" "the checkpoint to evaluate"

in_container $MACHINE "
cd /workspace/VAD
python tools/eval_holdout_l2_and_tinfer.py $CONFIG $CKPT \
    --ann-file $VAL_ANN --frame-offsets $OFFSETS --fp16 \
    --bev-only-history --device $GPU $EXTRA \
    > $OUT 2>&1
grep -E 'L2@|Final Planning|LANE_KEEP|LANE_CHANGE|TURN_|U_TURN|STOP|T_mean|T_median|penalty' $OUT | head -20
"
echo
echo "full log: $WORK_DIR/$(basename $OUT)  ($MACHINE)"
echo "reference: compliant baseline 0.4885 / clean control 0.3339 (current best)"
echo "score = L2 x (1 + max(0,T-100)/200). T is T_median/T_mean above (forward sum per clip)"
echo "note: T_infer measured alongside another eval on the same machine is inflated by CPU contention"
