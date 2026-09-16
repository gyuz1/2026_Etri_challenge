"""Predicted 5s goal on per-command anchors (one lateral centre per forward bin). Submittable.

VADLAW_etri_tiny_clean_nodistill.py + goal_pred with _goal_anchors_pair_lat1.py.
The lat1/adaptive pair shares forward bins, goal_scale, donor, seed and
schedule; only the lateral anchors differ. The target point is a training
label and selects which candidate loss_goal_select supervises; it never
enters generation. Anchors: tools/make_goal_anchor_pair.py (train only).
"""

_base_ = ['./VADLAW_etri_tiny_clean_nodistill.py',
          './_goal_anchors_pair_lat1.py']

model = dict(
    pts_bbox_head=dict(
        goal_pred=True,
        goal_anchors={{_base_.goal_anchors}},
        goal_scale=(115.0, 25.0),
        goal_long_ts=10,
        goal_cls_weight=0.5,
        goal_off_weight=0.5,
        goal_follow_weight=0.1,
        goal_select_weight=1.0,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_goalanchors_lat1_v1 (TP-select loss)')),
    ])
