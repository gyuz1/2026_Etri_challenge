#!/usr/bin/env bash
# Builds the two stage-2 donors after stage 1. Usage: ./scripts/make_stage2_donors.sh [epoch]
#
#   A5000: stage1_best_lcfon/epoch_N.pth -> stage2_init_merged.pth         (520 wide, teacher)
#   3090 : stage1_best_nolcf/epoch_N.pth -> stage2_init_merged_lcfemb8.pth (520 wide, student)
#
# The student needs an extra 8-column zero pad after the merge; the teacher's
# stage 1 already trained the 8 ego_lcf columns, so it is 520 wide already.
#
# Both steps were dry-run on existing checkpoints (2026-09-12): the merge takes
# 41 world-model keys from nuScenes and the rest from stage 1 with nothing
# missing, and the surgery widens (512,512) -> (512,520) leaving the 512 scene
# columns bit-identical and the 8 new ones exactly zero.
cd "$(dirname "$0")/.."
source scripts/_common.sh

EPOCH="${1:-48}"
FAIL=0

prep() {
  local machine="$1" wd="$2" out="$3" pad="$4"
  local ck="work_dirs/$wd/epoch_${EPOCH}.pth"
  echo "=== $machine : $wd (epoch $EPOCH) ==="
  require_file $machine "$ck" "the stage-1 checkpoint" || { FAIL=1; return 1; }
  in_container $machine "
cd /workspace/VAD
set -e
python tools/merge_stage1_world_model.py \
    --stage1 $ck \
    --world-model-source ckpts/law_pretrained_nus.pth \
    --output work_dirs/$wd/_merged_raw.pth
if [ '$pad' = 'yes' ]; then
  python tools/surgical_ego_fut_decoder_transfer.py \
      --stage1-on work_dirs/$wd/_merged_raw.pth \
      --target    work_dirs/$wd/_merged_raw.pth \
      --ego-lcf-n 0 --pad-input-cols 8 \
      --output work_dirs/$wd/$out
  rm -f work_dirs/$wd/_merged_raw.pth
else
  mv work_dirs/$wd/_merged_raw.pth work_dirs/$wd/$out
fi
ls -la work_dirs/$wd/$out
" || { FAIL=1; return 1; }
}

prep a5000 stage1_best_lcfon stage2_init_merged.pth         no
prep 3090  stage1_best_nolcf stage2_init_merged_lcfemb8.pth yes

echo
echo "=== auditing the donors against the stage-2 model ==="
in_container 3090 "cd /workspace/VAD && python tools/audit_pipeline.py \
    projects/configs/VAD/VADLAW_etri_tiny_clean_nodistill.py \
    --eval-config projects/configs/VAD/VADLAW_etri_tiny_fast_eval_clean.py" || FAIL=1

[ $FAIL -eq 0 ] && echo "done -- stage 2 can start" || { echo "failed -- fix the items above first" >&2; exit 1; }
