#!/usr/bin/env bash
# Shared definitions. Every run_*.sh sources this.
set -euo pipefail

ANN_DIR="data/etri/.causal_regen_split_301_75_10hz"
TRAIN_ANN="$ANN_DIR/vad_etri_infos_temporal_train_split.pkl"
VAL_ANN="$ANN_DIR/vad_etri_infos_temporal_val_split.pkl"

# How to reach each machine: the 3090 is local docker, the A5000 goes over ssh.
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

# require_file <machine> <path> <description>
#   Stops immediately if missing, so a student never runs without its teacher
#   and quietly produces wrong results.
require_file() {
  local machine="$1" path="$2" what="$3"
  if ! in_container "$machine" "test -f '$path'" 2>/dev/null; then
    echo "STOP: $what is missing" >&2
    echo "  expected at: $path  ($machine)" >&2
    echo "  finish the preceding step first" >&2
    return 1
  fi
  echo "ok: $what"
}

# require_gpu_free <machine>
#   Stops if a training run is already going, so two runs never collide and OOM.
require_gpu_free() {
  local machine="$1"
  local n
  # grep -c exits 1 when it finds nothing. Leaving `|| echo 0` there makes the
  # output two lines ("0" and "0"), and `[ "0\n0" -gt 0 ]` then dies with
  # "integer expression expected". Absorb the failure inside grep and keep only
  # the last line.
  n=$(in_container "$machine" "ps -eo args | grep -c '[t]ools/train.py' || true" 2>/dev/null | tail -1)
  n=${n//[!0-9]/}
  if [ "${n:-0}" -gt 0 ] 2>/dev/null; then
    echo "STOP: training is already running on $machine ($n train.py processes)" >&2
    echo "  check what is running with ./scripts/tail_log.sh" >&2
    return 1
  fi
  echo "ok: $machine GPUs are free"
}

# launch_train <machine> <config> <work_dir> <port> [extra cfg-options...]
launch_train() {
  local machine="$1" config="$2" work_dir="$3" port="$4"; shift 4
  local extra="${*:-}"
  echo "starting: $config"
  echo "  machine : $machine"
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
  echo "  log     : see ./scripts/tail_log.sh"
}

# verify_start <machine> <work_dir>
#   Waits until the first iteration is logged or the run crashes, then reports.
#
#   The log is fetched verbatim from the machine and judged on the host. This
#   used to define a shell function inside the in_container string, but on the
#   A5000 (ssh -> docker exec) one layer of quoting is stripped, `$p` vanished
#   and the definition broke. coreutils `cut` was called instead, erroring every
#   time while the until loop span forever (six piled up from 09-12 on), and a
#   healthy launch was reported as a failure.
verify_start() {
  local machine="$1" work_dir="$2"
  local marker="${LAST_LAUNCH_MARKER:-=== LAUNCH}"
  local tmp; tmp=$(mktemp)
  echo
  echo "waiting for the first iteration (a crash is shown immediately)..."
  local tries=0
  while :; do
    in_container "$machine" "cat $work_dir/train.log" 2>/dev/null \
      | awk -v m="$marker" 'index($0, m) {on=1} on' > "$tmp"
    grep -qE 'Epoch \[1\]\[100/|Traceback|Error' "$tmp" && break
    tries=$((tries + 1))
    if [ "$tries" -gt 180 ]; then   # 30 minutes
      echo "30 minutes with neither iteration 100 nor a crash -- check by hand"
      rm -f "$tmp"; return 1
    fi
    sleep 10
  done
  echo '--- checkpoint loading ---'
  grep -E 'load checkpoint from local path: work_dirs' "$tmp" | tail -1 || true
  # Zero hits is the healthy case. Under set -euo pipefail grep's "not found"
  # (exit 1) would kill the script, hence || true -- on 2026-09-15 two healthy
  # launches were reported as rc=1 because of this.
  grep -oE 'size mismatch for [a-z_.0-9]+' "$tmp" | head -5 || true
  echo '--- first iteration ---'
  grep -oE 'Epoch \[1\]\[100/[0-9]+\].*eta: [^,]+' "$tmp" | head -1 || true
  local nt; nt=$(grep -c Traceback "$tmp" || true)
  echo "--- Traceback count: $nt"
  grep -oE 'loss_plan_reg: [0-9.]+|loss_feature_distill: [0-9.]+|loss_scene_distill: [0-9.]+|loss_status_distill: [0-9.]+|loss_aux_bev_motion: [0-9.]+|loss_aux_bev_future_motion: [0-9.]+|loss_goal_cls: [0-9.]+|loss_goal_off: [0-9.]+|loss_goal_follow: [0-9.]+' "$tmp" | tail -9 || true
  rm -f "$tmp"
  [ "${nt:-0}" -eq 0 ]
}
