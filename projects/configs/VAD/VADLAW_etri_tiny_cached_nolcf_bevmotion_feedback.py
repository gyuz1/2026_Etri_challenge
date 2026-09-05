"""Stage-2 VADLAW, ego_lcf OFF, same technique stack as
VADLAW_etri_tiny_cached_nolcf_bevmotion.py -- but aux_bev_motion's own
prediction is fed forward into ego_feats (aux_bev_motion_feedback=True),
not just used as a loss target.

Compliance basis: 2026-09-04 organizer Q&A (ETRI_오영민) -- "네트워크가
영상으로부터 직접 추론한 결과를 planner에 사용하는 것은 해당 값이 영상에서
파생된 정보이므로 허용됩니다" (a network's own value inferred FROM VIDEO is
allowed as planner input, since it's vision-derived). Explicitly
distinguished the same week from feeding privileged data (or a value gated
on it) directly into generation, which was rejected for another team's
goal-as-attention-query design even with the value path zeroed.

aux_bev_motion_head's prediction is a function of bev_embed alone -- it
never reads ego_lcf_target -- so it runs identically at real test-time
inference where no privileged ego_lcf data exists at all. See
VAD_head.py's aux_bev_motion_feedback constructor comment.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_feedback=True,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf (bevmotion stack, aux_bev_motion '
                      'FEEDBACK into ego_feats)'))),
    ])
