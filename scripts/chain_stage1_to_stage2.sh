#!/usr/bin/env bash
# stage1 이 끝나는 즉시 도너를 만들고 stage2 를 양 머신에 올린다.
#
#   A5000 -> teacher    (증류 대상. 제출 불가)
#   3090  -> nodistill  (teacher 를 기다리지 않는 제출 가능 모델)
#
# distilled student 는 teacher 의 epoch_12.pth 가 있어야 모델 생성이 되므로
# 여기서 띄울 수 없다. teacher 가 끝난 뒤 ./scripts/run_stage2_best.sh student.
#
# 도너 준비가 감사에 걸리면 학습을 시작하지 않고 멈춘다.
cd "$(dirname "$0")/.."
source scripts/_common.sh

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

log "stage1 완료 대기 (epoch_48 두 개)"
while true; do
  a=$(docker exec gyuz_split_3090 bash -lc \
      "test -f /workspace/VAD/work_dirs/stage1_best_nolcf/epoch_48.pth && echo 1 || echo 0" 2>/dev/null | tr -d '\r')
  b=$(ssh -p 22022 vcl@10.10.52.49 "docker exec gyuz_split2 bash -lc \
      \"test -f /workspace/VAD/work_dirs/stage1_best_lcfon/epoch_48.pth && echo 1 || echo 0\"" 2>/dev/null | tr -d '\r')
  [ "$a" = 1 ] && [ "$b" = 1 ] && break
  sleep 300
done
log "epoch_48 확인. 체크포인트 기록이 끝나도록 잠시 대기"
sleep 120

log "=== 도너 생성 + 감사 ==="
if ! ./scripts/make_stage2_donors.sh 48; then
  log "도너 준비 실패 -- stage2 를 시작하지 않는다"
  exit 1
fi

log "=== A5000 teacher 시작 ==="
MACHINE=a5000 ./scripts/run_stage2_best.sh teacher || { log "teacher 시작 실패"; exit 1; }

log "=== 3090 nodistill student 시작 ==="
./scripts/run_stage2_best.sh nodistill || { log "nodistill 시작 실패"; exit 1; }

log "완료. teacher 가 끝나면 ./scripts/run_stage2_best.sh student"
