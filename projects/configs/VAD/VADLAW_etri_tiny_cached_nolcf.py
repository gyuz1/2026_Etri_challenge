"""Stage-2 VADLAW, ego_lcf ablation: current-frame ego status
(velocity/accel/yaw-rate) removed from the planner's direct input.

Organizer Q&A (2026-09-01, ETRI_오영민, verbatim):
"판정 기준은 과거 궤적이나 status가 planner에 직접 또는 단순 임베딩으로
입력되어 최종궤적을 출력하는 여부에 결정됩니다. 배포 베이스라인 기준 과거
traj과 status를 planner에 입력으로 쓰는 구조를 지원하나 기본적으로 사용하지
않습니다. 이를 활성화 시 ... 영상이 결과에 실질적으로 기여하지 않는 것으로
판단합니다."

The distributed baseline ships with ego_lcf_feat_idx=None / ego_his_encoder=None
in every one of its 8 configs (verified against the original
ETRI_E2E_Driving_Challenge.tar.gz) -- our VADLAW_etri_tiny.py had
ego_lcf_feat_idx=[0..7] / use_ego_lcf_status=True already set before this
LAW_split experiment folder existed (predates this repo's first commit), to
match the LAW-recipe checkpoint's own training config. This file undoes that
so our submission matches the compliant default.

load_from here points at a *different* merged checkpoint
(stage2_init_merged_nolcf.pth) than the base config's default: it's built
from LAW_p-based/work_dirs/vad_tiny_law_stage2/epoch_12.pth (the LAW variant
that was itself trained with use_ego_lcf_status=False), not
ckpts/law_pretrained_nus.pth (use_ego_lcf_status=True) -- so ego_fut_decoder's
non-final layers still transfer (input dim now matches, 512 not 520) instead
of being reinitialized from scratch like a naive "just flip the flag" run
would cause.
"""

_base_ = ['./VADLAW_etri_tiny_cached.py']

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_nolcf.pth'

model = dict(
    use_ego_lcf_status=False,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=None,
    ))
