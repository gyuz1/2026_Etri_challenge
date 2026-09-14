#!/usr/bin/env bash
# A5000 stage1 이 끝나면 도너 두 개를 만들고 A5000 에 teacher 만 올린다.
# 3090 은 5프레임 실험 라인에 쓰므로 nodistill 을 띄우지 않는다.
# (nolcf 도너는 여기서 같이 만들어 둔다 -- 나중 distilled student 가 쓴다)
cd "$(dirname "$0")/.."
source scripts/_common.sh
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

log "A5000 stage1 완료 대기"
while true; do
  b=$(ssh -p 22022 vcl@10.10.52.49 "docker exec gyuz_split2 bash -lc \
      \"test -f /workspace/VAD/work_dirs/stage1_best_lcfon/epoch_48.pth && echo 1 || echo 0\"" 2>/dev/null | tr -d '\r')
  [ "$b" = 1 ] && break
  sleep 120
done
log "epoch_48 확인. 기록 완료 대기"
sleep 120

log "=== 도너 생성 + 감사 (양 계보) ==="
if ! ./scripts/make_stage2_donors.sh 48; then
  log "도너 준비 실패 -- teacher 를 시작하지 않는다"; exit 1
fi

log "=== A5000 teacher 시작 ==="
./scripts/run_stage2_best.sh teacher || { log "teacher 시작 실패"; exit 1; }
log "완료. 3090 은 5프레임 라인용으로 비워둔다"
