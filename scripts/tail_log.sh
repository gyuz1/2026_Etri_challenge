#!/usr/bin/env bash
# Usage: ./scripts/tail_log.sh <control|goalpred|student> [--raw]
#   Without --raw only the key fields are shown.
cd "$(dirname "$0")/.."
source scripts/_common.sh

RUN="${1:-}"; MODE="${2:-}"
case "$RUN" in
  control)  MACHINE=a5000; WORK_DIR=work_dirs/stage2_clean_nodistill ;;
  goalpred) MACHINE=3090;  WORK_DIR=work_dirs/stage2_clean_goalpred ;;
  student)  MACHINE=a5000; WORK_DIR=work_dirs/stage2_clean_student ;;
  *) echo "usage: $0 <control|goalpred|student> [--raw]" >&2; exit 1 ;;
esac

echo "=== $RUN  ($MACHINE : $WORK_DIR) ==="
if [ "$MODE" = "--raw" ]; then
  in_container $MACHINE "tail -f $WORK_DIR/train.log"
else
  in_container $MACHINE "tail -n 400 -f $WORK_DIR/train.log | grep --line-buffered -oE 'Epoch \[[0-9]+\]\[[0-9]+/[0-9]+\]|eta: [^,]+|loss_plan_reg: [0-9.]+|loss_goal_cls: [0-9.]+|loss_goal_off: [0-9.]+|Traceback'"
fi
