#!/usr/bin/env bash
# 사용법: ./scripts/eval_l2.sh <A|B> <teacher|student> [epoch] [--zero-ego-lcf]
#   2-frame hold-out L2 + T_infer 를 측정한다. epoch 기본값 12.
#   --zero-ego-lcf : 특권 입력을 0으로 만든 대조군. teacher 가 ego_lcf 를 실제로
#                    쓰는지 검증할 때 사용 (안 쓰면 수치가 거의 안 변한다).
cd "$(dirname "$0")/.."
source scripts/_common.sh

SCHEME="${1:-}"; ROLE="${2:-}"; EPOCH="${3:-12}"; EXTRA="${4:-}"
case "$SCHEME:$ROLE" in
  A:teacher) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_lcfemb_teacher
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb_teacher.py ;;
  A:student) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_split_distill.py ;;
  B:teacher) MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_lcfon_diag
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfon_diag.py ;;
  B:student) MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_nolcf_feature_distill
             CONFIG=projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nolcf_distill.py ;;
  *) echo "사용법: $0 <A|B> <teacher|student> [epoch] [--zero-ego-lcf]" >&2; exit 1 ;;
esac

CKPT="$WORK_DIR/epoch_${EPOCH}.pth"
OUT="$WORK_DIR/eval_l2_2frame${EXTRA:+_zerolcf}.log"

echo "=== $SCHEME안 $ROLE eval (epoch $EPOCH) ==="
require_file $MACHINE "$CKPT" "평가할 체크포인트"

in_container $MACHINE "
cd /workspace/VAD
python tools/eval_holdout_l2_and_tinfer.py $CONFIG $CKPT \
    --ann-file $VAL_ANN --frame-offsets 0,-5 --fp16 --device 0 $EXTRA \
    > $OUT 2>&1
grep -E 'L2@|Final Planning|LANE_KEEP|LANE_CHANGE|TURN_|U_TURN|STOP|T_mean|penalty' $OUT | head -20
"
echo
echo "전체 로그: $WORK_DIR/$(basename $OUT)  ($MACHINE)"
echo "비교 기준: compliant 베이스라인 0.4885m / LANE_KEEP 0.5079 (오차의 85%)"
