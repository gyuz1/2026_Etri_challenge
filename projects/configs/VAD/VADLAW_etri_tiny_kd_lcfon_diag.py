"""DIAGNOSTIC ONLY -- NOT SUBMITTABLE. ego_lcf is fed straight into
ego_fut_decoder here, which the 2026-08-31 organizer ruling prohibits.

Purpose: establish the ceiling of the KD stage1 lineage. That lineage
(stage1 trained with ego_lcf_feat_idx=None from the start, so its decoder
is natively 512-wide and ETRI-domain-adapted) is the best compliant
result so far at 0.4885 on the 2-frame hold-out. The only ego_lcf-ON
number on record, 0.2166, came from a different lineage and an older
codebase, so it cannot be differenced against 0.4885 to say what
compliance actually costs. This run supplies the matched comparison:
same stage1, same stage2 recipe, ego_lcf the only difference.

Whatever this scores must never be submitted -- it exists to size the
gap, not to close it.

The KD stage1 donor never had ego_lcf columns (it was trained without
them), so its 512-wide layer-0 weight is padded with 8 zero input
columns rather than sliced; ego_fut_dec_hidden_dim stays at the donor's
512 while the input widens to 520.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion_prismlcf.py']

model = dict(
    prev_bev_dropout=0.0,
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        ego_fut_dec_hidden_dim=512,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged_lcfon8.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_kd_LCFON_DIAG (ceiling of the KD lineage -- '
                      'NOT compliant, not submittable)'))),
    ])
