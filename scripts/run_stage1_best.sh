#!/usr/bin/env bash
# stage1 최선 구성. 사용법: ./scripts/run_stage1_best.sh <nolcf|lcfon>
#   nolcf -> student 도너 (512폭),  3090
#   lcfon -> teacher 도너 (520폭),  A5000
cd "$(dirname "$0")/.."
source scripts/_common.sh

VARIANT="${1:-}"
case "$VARIANT" in
  nolcf) MACHINE=3090  ; PORT=28990 ;;
  lcfon) MACHINE=a5000 ; PORT=28991 ;;
  *) echo "사용법: $0 <nolcf|lcfon>" >&2; exit 1 ;;
esac
CONFIG=projects/configs/VAD/VAD_etri_tiny_stage1_best_${VARIANT}.py
WORK_DIR=work_dirs/stage1_best_${VARIANT}

echo "=== stage1 BEST ($VARIANT, $MACHINE) ==="
require_gpu_free $MACHINE
require_file $MACHINE "$CONFIG" "stage1 config"
require_file $MACHINE "ckpts/law_pretrained_nus.pth" "nuScenes warm-start 체크포인트"
launch_train $MACHINE "$CONFIG" "$WORK_DIR" $PORT
verify_start $MACHINE "$WORK_DIR"
