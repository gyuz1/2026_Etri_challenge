#!/usr/bin/env bash
# Holdout evaluation queue for the A5000 container, 3-frame and 2-frame per run.
# Two GPU lanes in parallel for L2, then clean T_infer runs one at a time.
# Run inside gyuz_split2:  bash scripts/overnight_eval_a5000.sh
# Existing logs are never overwritten.
set -u
cd /workspace/VAD
VAL=data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl
CFG=projects/configs/VAD
OFFSETS=(--frame-offsets 0,-5,-10 0,-5)

run() {  # run <gpu> <log> <config> <checkpoint> [flags...]
  local gpu=$1 log=$2 cfg=$3 ckpt=$4; shift 4
  if [ -e "$log" ]; then echo "skip (exists): $log"; return; fi
  if [ ! -f "$ckpt" ]; then echo "missing checkpoint: $ckpt"; return; fi
  echo "=== $(date '+%F %T') start gpu$gpu $log"
  CUDA_VISIBLE_DEVICES=$gpu python tools/eval_holdout_l2_and_tinfer.py "$cfg" "$ckpt" \
    --ann-file "$VAL" "${OFFSETS[@]}" --fp16 --bev-only-history --device 0 "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T') done rc=$? gpu$gpu $log"
}

LAT1=work_dirs/stage2_goalanchors_lat1_v1
ADAPT=work_dirs/stage2_goalanchors_adaptive_v1
CTRL=work_dirs/stage2_clean_nodistill

lane0() {
  run 0 $LAT1/eval_ep12_3f2f_tpfree.log $CFG/VADLAW_etri_tiny_fast_eval_clean_goalanchors_lat1.py \
    $LAT1/epoch_12.pth --test-commands
  run 0 $LAT1/eval_ep12_3f2f_tpselect.log $CFG/VADLAW_etri_tiny_fast_eval_clean_goalanchors_lat1.py \
    $LAT1/epoch_12.pth --test-commands --select-goal-by-tp
}

lane1() {
  run 1 $CTRL/eval_ep12_3f2f_speedstop.log $CFG/VADLAW_etri_tiny_fast_eval_clean.py \
    $CTRL/epoch_12.pth --test-commands
  run 1 $CTRL/eval_ep12_3f2f_tpstop.log $CFG/VADLAW_etri_tiny_fast_eval_clean.py \
    $CTRL/epoch_12.pth --stop-by-tp
  until [ -f $ADAPT/epoch_12.pth ]; do sleep 60; done
  run 1 $ADAPT/eval_ep12_3f2f_tpfree.log $CFG/VADLAW_etri_tiny_fast_eval_clean_goalanchors_adaptive.py \
    $ADAPT/epoch_12.pth --test-commands
  run 1 $ADAPT/eval_ep12_3f2f_tpselect.log $CFG/VADLAW_etri_tiny_fast_eval_clean_goalanchors_adaptive.py \
    $ADAPT/epoch_12.pth --test-commands --select-goal-by-tp
}

lane0 &
lane1 &
wait

# T_infer with the other GPU idle (the L2 lanes above share the machine).
run 0 $CTRL/timing_ep12_3f2f.log $CFG/VADLAW_etri_tiny_fast_eval_clean.py \
  $CTRL/epoch_12.pth --test-commands --stride 50 --warmup-windows 20
run 0 $LAT1/timing_ep12_3f2f.log $CFG/VADLAW_etri_tiny_fast_eval_clean_goalanchors_lat1.py \
  $LAT1/epoch_12.pth --test-commands --select-goal-by-tp --stride 50 --warmup-windows 20
echo "=== $(date '+%F %T') queue finished"
