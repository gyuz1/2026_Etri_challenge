#!/usr/bin/env bash
# v1 refine 재평가가 끝나면 A5000 에 teacher 를 올린다.
# teacher 는 이제 nodistill/증류 student 와 설정이 대칭이다 (refine off,
# aux_long_horizon residual on) -- ego_feats 를 빚는 손실이 양쪽에서 같아야
# 증류가 서로 반대로 당기지 않는다.
cd "$(dirname "$0")/.."
source scripts/_common.sh
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

log "v1 재평가 완료 대기"
while ssh -p 22022 vcl@10.10.52.49 \
    "docker exec gyuz_split2 pgrep -f '[e]val_holdout'" >/dev/null 2>&1; do
  sleep 60
done
log "평가 종료 확인"
for T in on off; do
  R=$(ssh -p 22022 vcl@10.10.52.49 "docker exec gyuz_split2 grep -a 'Final Planning L2' /workspace/VAD/work_dirs/_v1_refine_$T.log" 2>/dev/null | grep -oE '[0-9.]+')
  log "  v1 refine=$T : ${R:-없음}"
done

log "=== A5000 teacher 시작 (refine off, long-horizon residual on) ==="
./scripts/run_stage2_best.sh teacher || { log "teacher 시작 실패"; exit 1; }
log "완료"
