#!/usr/bin/env bash
# 사용법: ./scripts/eval_l2.sh <teacher|student|A:teacher|A:student|B:teacher> [epoch] [--zero-ego-lcf]
#
#   hold-out L2 + T_infer 를 측정한다. epoch 기본값 12.
#   --zero-ego-lcf : 특권 입력을 0으로 만든 대조군. teacher 가 ego_lcf 를 실제로
#                    쓰는지 검증할 때 쓴다 (안 쓰면 수치가 거의 안 변한다).
#
# 프레임 수는 config 에서 읽는다. 이건 편의 기능이 아니라 정확성 요건이다 --
# aux_bev_motion_frames=3 모델을 2프레임 창으로 평가하면 prev_bev2 가 끝까지
# None 이라 가속도 블록이 0 인 채 채점되고, 학습이 채운 것과 다른 입력을 준다.
# 그러면 L2 는 조용히 나빠지고 원인은 로그 어디에도 안 남는다.
cd "$(dirname "$0")/.."
source scripts/_common.sh

TARGET="${1:-}"; EPOCH="${2:-12}"; EXTRA="${3:-}"
case "$TARGET" in
  teacher)   MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_best.py ;;
  student)   MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill8_3f_fut_g8
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py ;;
  # 과거 계보 (폴백·기록용)
  A:teacher) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_lcfemb_teacher
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb_teacher.py ;;
  A:student) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill.py ;;
  B:teacher) MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_lcfon_diag
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfon_diag.py ;;
  *) echo "사용법: $0 <teacher|student|A:teacher|A:student|B:teacher> [epoch] [--zero-ego-lcf]" >&2
     exit 1 ;;
esac

CKPT="$WORK_DIR/epoch_${EPOCH}.pth"

# config 가 요구하는 프레임 수를 읽어 창 길이를 정한다. FRAME_OFFSETS 를 직접
# 지정하면 그게 우선하지만, 프레임 수가 모자라면 거부한다.
FRAMES=$(in_container $MACHINE "cd /workspace/VAD && python -c \"
import mmcv; print(mmcv.Config.fromfile('$CONFIG').model.pts_bbox_head.get('aux_bev_motion_frames') or 2)
\" 2>/dev/null" | tr -d '\r' | tail -1)
case "$FRAMES" in
  2) DEFAULT_OFFSETS=0,-5 ;;
  3) DEFAULT_OFFSETS=0,-5,-10 ;;
  *) echo "aux_bev_motion_frames 를 읽지 못했다: '$FRAMES'" >&2; exit 1 ;;
esac
OFFSETS="${FRAME_OFFSETS:-$DEFAULT_OFFSETS}"
N_OFF=$(awk -F, '{print NF}' <<<"$OFFSETS")
if [ "$N_OFF" -lt "$FRAMES" ]; then
  echo "거부: config 는 ${FRAMES}프레임인데 창이 ${N_OFF}프레임이다 ($OFFSETS)." >&2
  echo "  가속도 블록이 0 인 채 채점된다. 최소 ${FRAMES}프레임을 지정할 것." >&2
  exit 1
fi

OUT="$WORK_DIR/eval_l2_${N_OFF}frame${EXTRA:+_zerolcf}.log"

echo "=== $TARGET eval (epoch $EPOCH, ${N_OFF}프레임 $OFFSETS) ==="
require_file $MACHINE "$CKPT" "평가할 체크포인트"

in_container $MACHINE "
cd /workspace/VAD
python tools/eval_holdout_l2_and_tinfer.py $CONFIG $CKPT \
    --ann-file $VAL_ANN --frame-offsets $OFFSETS --fp16 \
    --bev-only-history --device 0 $EXTRA \
    > $OUT 2>&1
grep -E 'L2@|Final Planning|LANE_KEEP|LANE_CHANGE|TURN_|U_TURN|STOP|T_mean|T_median|penalty' $OUT | head -20
"
echo
echo "전체 로그: $WORK_DIR/$(basename $OUT)  ($MACHINE)"
echo "비교 기준: compliant 베이스라인 0.4885 / A student v1 0.4218 (현 최고)"
echo "T_infer: 3프레임 117.3ms -> 페널티 x1.087. 점수 = L2 x (1 + max(0,T-100)/200)"
