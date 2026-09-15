#!/usr/bin/env bash
# stage2 두 학습이 끝나는 대로 곧장 추론하고, 빈 GPU 에 다음 학습을 올린다.
#
#   A5000 teacher 끝  -> epoch_12 를 3090 으로 복사(md5 대조)
#                     -> A5000 에 distilled student 시작 (감사 자동)
#   3090 nodistill 끝 -> GPU0: nodistill  --test-commands  -> nodistill (val 명령 그대로)
#                        GPU1: teacher    --test-commands  -> teacher --test-commands --zero-ego-lcf
#                     -> nodistill epoch_12 가속도 블록 검사
#
# 평가는 전부 3090 에서 한다 (A5000 컨테이너엔 원본 데이터셋이 없다).
# 사용법: nohup ./scripts/chain_stage2_finish.sh > work_dirs/chain_stage2_finish.log 2>&1 &
cd "$(dirname "$0")/.."
source scripts/_common.sh
set +e   # 한 단계 실패가 나머지 평가를 막지 않게. 실패는 전부 로그에 남긴다.

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
A5000_ROOT=/media/vcl/SSD-DATA/gyuz/LAW_split
T_DIR=work_dirs/stage2_kd_lcfemb8_teacher_best
N_DIR=work_dirs/stage2_nodistill_best

train_running() {
  local n
  n=$(in_container "$1" "ps -eo args | grep -c '[t]ools/train.py' || true" 2>/dev/null | tail -1)
  n=${n//[!0-9]/}
  [ "${n:-0}" -gt 0 ]
}

teacher_phase() {
  log "[teacher] A5000 학습 종료 대기"
  until in_container a5000 "test -f /workspace/VAD/$T_DIR/epoch_12.pth" 2>/dev/null \
        && ! train_running a5000; do
    sleep 120
  done
  sleep 60   # EMA 사본·latest 링크 기록이 끝나도록
  log "[teacher] epoch_12 -> 3090 복사"
  mkdir -p "$T_DIR"
  scp -q -P 22022 "vcl@10.10.52.49:$A5000_ROOT/$T_DIR/epoch_12.pth" "$T_DIR/epoch_12.pth.part"
  local src dst
  src=$($A5000_SSH "md5sum $A5000_ROOT/$T_DIR/epoch_12.pth" | awk '{print $1}')
  dst=$(md5sum "$T_DIR/epoch_12.pth.part" | awk '{print $1}')
  if [ -n "$src" ] && [ "$src" = "$dst" ]; then
    mv "$T_DIR/epoch_12.pth.part" "$T_DIR/epoch_12.pth"
    scp -q -P 22022 "vcl@10.10.52.49:$A5000_ROOT/$T_DIR/train.log" "$T_DIR/train.log.a5000" || true
    log "[teacher] 복사 확인 md5=$src"
  else
    log "[teacher] 복사 md5 불일치 ($src vs $dst) -- teacher 평가 불가"
  fi

  log "[teacher] A5000 에 distilled student 시작"
  MACHINE_OVERRIDE=a5000 ./scripts/run_stage2_best.sh student \
    || log "[teacher] student 시작 실패 -- 위 감사 출력 확인"
}

nodistill_phase() {
  log "[3090] nodistill 학습 종료 대기"
  until [ -f "$N_DIR/epoch_12.pth" ] && ! train_running 3090; do
    sleep 120
  done
  sleep 60

  (
    log "[GPU0] nodistill --test-commands"
    EVAL_GPU=0 ./scripts/eval_l2.sh nodistill 12 --test-commands || log "[GPU0] 실패"
    log "[GPU0] nodistill (val 명령 그대로, 이전 수치와 비교용)"
    EVAL_GPU=0 ./scripts/eval_l2.sh nodistill 12 || log "[GPU0] 실패"
  ) &
  local p0=$!

  (
    log "[GPU1] teacher 체크포인트 대기"
    until [ -f "$T_DIR/epoch_12.pth" ]; do sleep 60; done
    log "[GPU1] teacher --test-commands"
    EVAL_GPU=1 ./scripts/eval_l2.sh teacher 12 --test-commands || log "[GPU1] 실패"
    log "[GPU1] teacher --test-commands --zero-ego-lcf (대조군: 크게 나빠져야 정상)"
    EVAL_GPU=1 ./scripts/eval_l2.sh teacher 12 --test-commands --zero-ego-lcf || log "[GPU1] 실패"
  ) &
  local p1=$!

  wait $p0 $p1
  log "[3090] nodistill epoch_12 가속도 블록 검사"
  in_container 3090 "cd /workspace/VAD && CUDA_VISIBLE_DEVICES= python tools/check_accel_block_trained.py \
    $N_DIR/epoch_12.pth --config projects/configs/VAD/VADLAW_etri_tiny_nodistill_best.py" \
    || log "[3090] 가속도 블록 검사 실패"
  log "[3090] 평가 전부 끝"
}

log "chain 시작"
teacher_phase &
TP=$!
nodistill_phase
wait $TP
log "chain 끝"
