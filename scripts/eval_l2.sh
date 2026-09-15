#!/usr/bin/env bash
# 사용법: [EVAL_GPU=0] ./scripts/eval_l2.sh <student-clean|nodistill-clean|goalgrid-clean|teacher|student|nodistill|nodrop|A:student> [epoch] [플래그...]
#
#   hold-out L2 + T_infer 를 측정한다. epoch 기본값 12. 평가는 전부 3090 에서 한다
#   (A5000 컨테이너엔 원본 데이터셋이 없다 -- AGENTS.md). A5000 에서 학습한
#   체크포인트는 먼저 3090 의 같은 경로로 복사할 것.
#   플래그는 eval_holdout_l2_and_tinfer.py 로 그대로 넘어가고 로그 이름에 붙는다.
#   --test-commands : 테스트 조건. val 의 STOP 명령(학습 pkl 파생 라벨)을 지우고
#                     모델 자신의 속도 추정으로 STOP 을 고른다. 제출 수치는 이것.
#   --zero-ego-lcf  : 특권 입력을 0으로 만든 대조군. teacher 가 ego_lcf 를 실제로
#                     쓰는지 검증할 때 쓴다 (안 쓰면 수치가 거의 안 변한다).
#
# 프레임 수는 config 에서 읽는다. 이건 편의 기능이 아니라 정확성 요건이다 --
# aux_bev_motion_frames=3 모델을 2프레임 창으로 평가하면 prev_bev2 가 끝까지
# None 이라 가속도 블록이 0 인 채 채점되고, 학습이 채운 것과 다른 입력을 준다.
# 그러면 L2 는 조용히 나빠지고 원인은 로그 어디에도 안 남는다.
cd "$(dirname "$0")/.."
source scripts/_common.sh

TARGET="${1:-}"; EPOCH="${2:-12}"; shift 2 2>/dev/null || shift $#
EXTRA="$*"
GPU="${EVAL_GPU:-0}"
case "$TARGET" in
  student-clean)   WORK_DIR=work_dirs/stage2_clean_student
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py ;;
  goalgrid-clean)  WORK_DIR=work_dirs/stage2_clean_goalgrid
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalgrid.py ;;
  nodistill-clean) WORK_DIR=work_dirs/stage2_clean_nodistill
                   CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py ;;
  # 2026-09-15 이전 학습물 (dropout 등 불일치 설정으로 학습됨, 비교·폴백용)
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
  *) echo "사용법: $0 <student-clean|nodistill-clean|goalgrid-clean|teacher|student|nodistill|nodrop|A:student> [epoch] [플래그...]" >&2
     exit 1 ;;
esac

MACHINE=3090   # 평가는 3090 에서만 (A5000 컨테이너엔 원본 데이터셋이 없다)
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

# 플래그마다 로그를 따로 둔다. 예전엔 플래그가 뭐든 _zerolcf 가 붙어서
# --test-commands 결과가 대조군 이름으로 저장될 뻔했다.
TAG=$(echo "$EXTRA" | sed 's/--//g; s/[^A-Za-z0-9.-]\+/_/g; s/^_//; s/_$//')
OUT="$WORK_DIR/eval_l2_ep${EPOCH}_${N_OFF}frame${TAG:+_$TAG}.log"

echo "=== $TARGET eval (epoch $EPOCH, ${N_OFF}프레임 $OFFSETS) ==="
require_file $MACHINE "$CKPT" "평가할 체크포인트"

in_container $MACHINE "
cd /workspace/VAD
python tools/eval_holdout_l2_and_tinfer.py $CONFIG $CKPT \
    --ann-file $VAL_ANN --frame-offsets $OFFSETS --fp16 \
    --bev-only-history --device $GPU $EXTRA \
    > $OUT 2>&1
grep -E 'L2@|Final Planning|LANE_KEEP|LANE_CHANGE|TURN_|U_TURN|STOP|T_mean|T_median|penalty' $OUT | head -20
"
echo
echo "전체 로그: $WORK_DIR/$(basename $OUT)  ($MACHINE)"
echo "비교 기준: compliant 베이스라인 0.4885 / A student v1 0.4218 (현 최고)"
echo "점수 = L2 x (1 + max(0,T-100)/200). T 는 위 로그의 T_median/T_mean (클립당 forward 합)"
echo "주의: 같은 머신에서 다른 평가와 동시에 돌린 T_infer 는 CPU 경합으로 부풀 수 있다"
