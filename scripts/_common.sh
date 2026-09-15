#!/usr/bin/env bash
# 공통 정의. 각 run_*.sh 가 source 한다.
set -euo pipefail

ANN_DIR="data/etri/.causal_regen_split_301_75_10hz"
TRAIN_ANN="$ANN_DIR/vad_etri_infos_temporal_train_split.pkl"
VAL_ANN="$ANN_DIR/vad_etri_infos_temporal_val_split.pkl"

# 머신별 접속 방법. 3090은 로컬 docker, A5000은 ssh 경유.
A5000_SSH="ssh -p 22022 vcl@10.10.52.49"
C_3090="gyuz_split_3090"
C_A5000="gyuz_split2"

# in_container <machine> <command...>
#   machine: 3090 | a5000
in_container() {
  local machine="$1"; shift
  case "$machine" in
    3090)  docker exec "$C_3090" bash -c "$*" ;;
    a5000) $A5000_SSH "docker exec $C_A5000 bash -c \"$*\"" ;;
    *) echo "unknown machine: $machine" >&2; return 1 ;;
  esac
}

# in_container_detached <machine> <command...>
in_container_detached() {
  local machine="$1"; shift
  case "$machine" in
    3090)  docker exec -d "$C_3090" bash -c "$*" ;;
    a5000) $A5000_SSH "docker exec -d $C_A5000 bash -c \"$*\"" ;;
    *) echo "unknown machine: $machine" >&2; return 1 ;;
  esac
}

# require_file <machine> <path> <설명>
#   없으면 즉시 중단. teacher 없이 student 를 돌려 조용히 틀린 결과가 나오는 걸 막는다.
require_file() {
  local machine="$1" path="$2" what="$3"
  if ! in_container "$machine" "test -f '$path'" 2>/dev/null; then
    echo "중단: $what 가 없습니다" >&2
    echo "  기대 경로: $path  ($machine)" >&2
    echo "  선행 단계를 먼저 끝내세요 (scripts/README.md 참조)" >&2
    return 1
  fi
  echo "확인: $what"
}

# require_gpu_free <machine>
#   이미 학습이 돌고 있으면 중단. 실수로 두 개를 겹쳐 돌려 OOM 나는 걸 막는다.
require_gpu_free() {
  local machine="$1"
  local n
  # grep -c 는 0건일 때 exit 1 을 낸다. `|| echo 0` 을 그대로 두면 출력이
  # "0" 과 "0" 두 줄이 되어 `[ "0\n0" -gt 0 ]` 이 "integer expression
  # expected" 로 죽는다. grep 쪽에서 실패를 흡수하고 마지막 한 줄만 쓴다.
  n=$(in_container "$machine" "ps -eo args | grep -c '[t]ools/train.py' || true" 2>/dev/null | tail -1)
  n=${n//[!0-9]/}
  if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
    echo "중단: $machine 에서 이미 학습이 돌고 있습니다 (train.py $n개)" >&2
    echo "  확인: ./scripts/tail_log.sh 로 무엇이 도는지 보세요" >&2
    return 1
  fi
  echo "확인: $machine GPU 비어있음"
}

# launch_train <machine> <config> <work_dir> <port> [extra cfg-options...]
launch_train() {
  local machine="$1" config="$2" work_dir="$3" port="$4"; shift 4
  local extra="${*:-}"
  echo "시작: $config"
  echo "  머신    : $machine"
  echo "  work_dir: $work_dir"
  # Marker so verify_start reads only THIS launch. Without it, a Traceback
  # left in the previous log -- pkill writes one -- can still be visible when
  # verify_start's poll races the rm -rf, and a healthy start gets reported
  # as a crash. That happened once and cost a needless round of debugging.
  local marker="=== LAUNCH $(date -u +%Y%m%d_%H%M%S) ==="
  LAST_LAUNCH_MARKER="$marker"
  in_container_detached "$machine" "
cd /workspace/VAD
export WANDB_INIT_TIMEOUT=600 WANDB__SERVICE_WAIT=600 WANDB_HTTP_TIMEOUT=120
rm -rf $work_dir; mkdir -p $work_dir
echo '$marker' > $work_dir/train.log
nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port \
    tools/train.py $config \
    --launcher pytorch --work-dir $work_dir \
    --no-validate --deterministic \
    --cfg-options data.workers_per_gpu=2 \
        data.train.ann_file=$TRAIN_ANN \
        data.val.ann_file=$VAL_ANN \
        data.test.ann_file=$VAL_ANN $extra \
    >> $work_dir/train.log 2>&1 &
disown
"
  echo "  로그    : ./scripts/tail_log.sh 로 확인"
}

# verify_start <machine> <work_dir>
#   첫 iteration 이 찍히거나 크래시할 때까지 기다린 뒤 결과를 보여준다.
#
#   로그는 머신에서 원문 그대로 가져오고 판정은 호스트에서 한다. 예전엔 셸 함수를
#   in_container 문자열 안에 정의했는데, A5000 은 ssh -> docker exec 로 따옴표가
#   한 겹 더 벗겨져 `$p` 가 사라지고 함수 정의가 깨졌다. 그 결과 coreutils `cut`
#   이 불려 매번 에러를 내며 until 루프가 영원히 돌았고(09-12 부터 6개 누적),
#   정상 시작한 학습이 "시작 실패"로 보고됐다.
verify_start() {
  local machine="$1" work_dir="$2"
  local marker="${LAST_LAUNCH_MARKER:-=== LAUNCH}"
  local tmp; tmp=$(mktemp)
  echo
  echo "첫 iteration 대기 중 (크래시하면 즉시 표시)..."
  local tries=0
  while :; do
    in_container "$machine" "cat $work_dir/train.log" 2>/dev/null \
      | awk -v m="$marker" 'index($0, m) {on=1} on' > "$tmp"
    grep -qE 'Epoch \[1\]\[100/|Traceback|Error' "$tmp" && break
    tries=$((tries + 1))
    if [ "$tries" -gt 180 ]; then   # 30분
      echo "30분 안에 첫 100 iteration 도 크래시도 안 보인다 -- 직접 확인할 것"
      rm -f "$tmp"; return 1
    fi
    sleep 10
  done
  echo '--- 체크포인트 로딩 ---'
  grep -E 'load checkpoint from local path: work_dirs' "$tmp" | tail -1
  grep -oE 'size mismatch for [a-z_.0-9]+' "$tmp" | head -5
  echo '--- 첫 iteration ---'
  grep -oE 'Epoch \[1\]\[100/[0-9]+\].*eta: [^,]+' "$tmp" | head -1
  local nt; nt=$(grep -c Traceback "$tmp" || true)
  echo "--- Traceback 수: $nt"
  grep -oE 'loss_plan_reg: [0-9.]+|loss_feature_distill: [0-9.]+|loss_scene_distill: [0-9.]+|loss_status_distill: [0-9.]+|loss_aux_bev_motion: [0-9.]+|loss_aux_bev_future_motion: [0-9.]+|loss_goal_cls: [0-9.]+' "$tmp" | tail -7
  rm -f "$tmp"
  [ "${nt:-0}" -eq 0 ]
}
