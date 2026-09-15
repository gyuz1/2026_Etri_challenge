#!/usr/bin/env bash
# 3090 평가(run_stage2_evals.sh)가 끝나면 곧장 goal-grid 학습을 올린다.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "3090 평가 종료 대기"
while pgrep -f 'scripts/run_stage2_evals[.]sh' >/dev/null \
   || docker exec gyuz_split_3090 bash -c "ps -eo args | grep -q '[e]val_holdout_l2'"; do
  sleep 30
done
log "평가 끝 -- goal-grid 학습 시작"
./scripts/run_stage2_best.sh goalgrid && log "시작 확인" || log "시작 실패 -- 위 출력 확인"
