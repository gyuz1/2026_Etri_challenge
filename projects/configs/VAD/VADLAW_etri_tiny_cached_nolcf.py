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
(stage2_init_merged_nolcf.pth) than the base config's default. Built via:

    tools/merge_stage1_world_model.py
        --stage1 stage1_etri_split_301_75_10hz/epoch_48.pth
        --world-model-source ckpts/law_pretrained_nus_nolcf.pth
        --override-prefixes-from-world-model pts_bbox_head.ego_fut_decoder.

Stage1 was kept unchanged (use_ego_lcf_status=True there, per its own
config) rather than retrained, so its ego_fut_decoder is shaped for the
520-wide lcf-ON input -- incompatible with this 512-wide lcf-OFF head.
merge_stage1_world_model.py's plain union (base = stage1, + bev_world_model.*
from world-model-source) therefore left ego_fut_decoder shape-mismatched
and silently skipped by mmcv's load_checkpoint, training it from complete
random init -- discovered by loading this checkpoint outside the normal
launch path and actually reading the "size mismatch" log lines, which the
first version of this file's docstring wrongly assumed away. The
--override-prefixes-from-world-model flag (added once this was found) takes
ego_fut_decoder specifically from the nolcf LAW checkpoint instead: its
first two layers are 512-wide (matches exactly) even though it's nuScenes-
trained rather than ETRI-domain-adapted, and its final layer stays
randomly initialized regardless (36 = nuScenes's ego_fut_mode=3 vs our 84 =
ego_fut_mode=7, a real vocab-size difference no checkpoint here resolves).

The original (buggy) merge is kept at
stage2_init_merged_nolcf_BUGGY_520dim_ego_fut_decoder.pth for reference,
not for use.
"""

_base_ = ['./VADLAW_etri_tiny_cached.py']

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_nolcf.pth'

model = dict(
    use_ego_lcf_status=False,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=None,
    ))
