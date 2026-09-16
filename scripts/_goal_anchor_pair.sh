#!/usr/bin/env bash
# Helpers for the isolated goal-anchor experiment. Never starts a job on source.
set -euo pipefail
PAIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAIR_REMOTE_HOST="vcl@10.10.52.49"
PAIR_REMOTE_PORT=22022
PAIR_REMOTE_ROOT=/media/vcl/SSD-DATA/gyuz/LAW_split
PAIR_TRAIN_ANN=data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_train_split.pkl
PAIR_VAL_ANN=data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl
PAIR_DONOR=work_dirs/stage1_best_nolcf/stage2_init_merged_lcfemb8.pth

pair_role() {
  case "$1" in
    lat1) PAIR_MACHINE=a5000; PAIR_CONTAINER=gyuz_split2; PAIR_PORT=28996 ;;
    adaptive) PAIR_MACHINE=3090; PAIR_CONTAINER=gyuz_split_3090; PAIR_PORT=28997 ;;
    *) echo "unknown role: $1 (lat1|adaptive)" >&2; return 2 ;;
  esac
  PAIR_CONFIG="projects/configs/VAD/VADLAW_etri_tiny_clean_goalanchors_$1.py"
  PAIR_EVAL_CONFIG="projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean_goalanchors_$1.py"
  PAIR_WORK_DIR="work_dirs/stage2_goalanchors_${1}_v1"
}

pair_ssh() { ssh -p "$PAIR_REMOTE_PORT" "$PAIR_REMOTE_HOST" "$@"; }

# Quote each argv element once for the remote login shell, not nested bash strings.
pair_exec() {
  local machine="$1"; shift
  if [ "$machine" = 3090 ]; then
    docker exec -i gyuz_split_3090 "$@"
  else
    local command
    printf -v command '%q ' docker exec -i gyuz_split2 "$@"
    pair_ssh "$command"
  fi
}

pair_exec_detached() {
  local machine="$1"; shift
  if [ "$machine" = 3090 ]; then
    docker exec -d gyuz_split_3090 "$@"
  else
    local command
    printf -v command '%q ' docker exec -d gyuz_split2 "$@"
    pair_ssh "$command"
  fi
}

pair_require_idle() {
  pair_exec "$1" bash -c '
set -euo pipefail
jobs=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$jobs" ]; then
  printf "REFUSED: GPU compute processes are active:\n%s\n" "$jobs" >&2
  exit 1
fi
if pgrep -af "[t]ools/train[.]py"; then
  echo "REFUSED: a training process exists, even though no GPU process was reported." >&2
  exit 1
fi
' </dev/null
}

pair_source_files() {
  (cd "$PAIR_ROOT" && find projects/mmdet3d_plugin projects/configs tools scripts -type f |
    LC_ALL=C sort | while IFS= read -r path; do
      case "$path" in *.py|*.sh) printf '%s\n' "$path" ;; esac
    done)
}

pair_manifest() {
  local path
  while IFS= read -r path; do (cd "$PAIR_ROOT" && sha256sum "$path"); done < <(pair_source_files)
}

pair_verify_source() {
  echo "Checking all model/config/tool/script hashes against the local package ($1)."
  pair_manifest | pair_exec "$1" bash -c 'cd /workspace/VAD && sha256sum --check --strict'
}

pair_plan() {
  pair_role "$1"
  printf 'role=%s machine=%s work_dir=%s\ntrain=%s\neval=%s\ndonor=%s\n' \
    "$1" "$PAIR_MACHINE" "$PAIR_WORK_DIR" "$PAIR_CONFIG" "$PAIR_EVAL_CONFIG" "$PAIR_DONOR"
}
