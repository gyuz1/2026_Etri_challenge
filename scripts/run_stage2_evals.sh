#!/usr/bin/env bash
# stage2 nodistill + teacher 평가를 3090 두 GPU 에서 동시에 돌린다.
#   GPU0: nodistill --test-commands -> nodistill (val 명령 그대로)
#   GPU1: teacher --test-commands   -> teacher --test-commands --zero-ego-lcf
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
(
  log "[GPU0] nodistill --test-commands"
  EVAL_GPU=0 ./scripts/eval_l2.sh nodistill 12 --test-commands || log "[GPU0] 실패"
  log "[GPU0] nodistill (val 명령 그대로)"
  EVAL_GPU=0 ./scripts/eval_l2.sh nodistill 12 || log "[GPU0] 실패"
) &
(
  log "[GPU1] teacher --test-commands"
  EVAL_GPU=1 ./scripts/eval_l2.sh teacher 12 --test-commands || log "[GPU1] 실패"
  log "[GPU1] teacher --test-commands --zero-ego-lcf (대조군)"
  EVAL_GPU=1 ./scripts/eval_l2.sh teacher 12 --test-commands --zero-ego-lcf || log "[GPU1] 실패"
) &
wait
log "평가 전부 끝"
