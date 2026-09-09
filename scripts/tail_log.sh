#!/usr/bin/env bash
# 사용법: ./scripts/tail_log.sh <A|B> <teacher|student> [--raw]
#   --raw 없으면 핵심 필드만 추려서 보여준다.
cd "$(dirname "$0")/.."
source scripts/_common.sh

SCHEME="${1:-}"; ROLE="${2:-}"; MODE="${3:-}"
case "$SCHEME:$ROLE" in
  A:teacher) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_lcfemb_teacher ;;
  A:student) MACHINE=3090;  WORK_DIR=work_dirs/stage2_kd_nolcf_split_distill ;;
  B:teacher) MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_lcfon_diag ;;
  B:student) MACHINE=a5000; WORK_DIR=work_dirs/stage2_kd_nolcf_feature_distill ;;
  *) echo "사용법: $0 <A|B> <teacher|student> [--raw]" >&2; exit 1 ;;
esac

echo "=== $SCHEME안 $ROLE  ($MACHINE : $WORK_DIR) ==="
if [ "$MODE" = "--raw" ]; then
  in_container $MACHINE "tail -f $WORK_DIR/train.log"
else
  in_container $MACHINE "tail -n 400 -f $WORK_DIR/train.log | grep --line-buffered -oE 'Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\]|eta: [^,]+|loss_plan_reg: [0-9.]+|loss_feature_distill: [0-9.]+|loss_scene_distill: [0-9.]+|loss_motion_distill: [0-9.]+|Traceback'"
fi
