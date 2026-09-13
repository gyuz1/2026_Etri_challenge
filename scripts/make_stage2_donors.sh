#!/usr/bin/env bash
# stage1 이 끝난 뒤 stage2 도너 두 개를 만든다. 사용법: ./scripts/make_stage2_donors.sh [epoch]
#
#   A5000: stage1_best_lcfon/epoch_N.pth  -> stage2_init_merged.pth          (520폭, teacher)
#   3090 : stage1_best_nolcf/epoch_N.pth  -> stage2_init_merged_lcfemb8.pth  (520폭, student)
#
# student 쪽은 merge 뒤에 8칸 zero-pad 가 한 번 더 필요하다. teacher 계보는 stage1 이
# 이미 ego_lcf 8열을 학습했으므로 폭이 520 이라 수술이 필요 없다.
#
# 두 단계 모두 2026-09-12 에 기존 체크포인트로 예행 검증했다:
#   merge   -> world model 41키는 nuScenes, 나머지는 stage1, 누락 0 (정확한 합집합)
#   surgery -> (512,512)->(512,520), scene 512열 bit-identical, 추가 8열 정확히 0
cd "$(dirname "$0")/.."
source scripts/_common.sh

EPOCH="${1:-48}"
FAIL=0

prep() {
  local machine="$1" wd="$2" out="$3" pad="$4"
  local ck="work_dirs/$wd/epoch_${EPOCH}.pth"
  echo "=== $machine : $wd (epoch $EPOCH) ==="
  require_file $machine "$ck" "stage1 체크포인트" || { FAIL=1; return 1; }
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
echo "=== 도너가 stage2 모델에 맞는지 감사 ==="
in_container a5000 "cd /workspace/VAD && python tools/audit_pipeline.py \
    projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py \
    --eval-config projects/configs/VAD/VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_best.py" || FAIL=1
in_container 3090 "cd /workspace/VAD && python tools/audit_pipeline.py \
    projects/configs/VAD/VADLAW_etri_tiny_nodistill_best.py \
    --eval-config projects/configs/VAD/VADLAW_etri_tiny_fast_eval_nodistill_best.py" || FAIL=1

[ $FAIL -eq 0 ] && echo "완료 -- stage2 를 시작해도 된다" || { echo "실패 -- 위 항목 해결 전 진행 금지" >&2; exit 1; }
