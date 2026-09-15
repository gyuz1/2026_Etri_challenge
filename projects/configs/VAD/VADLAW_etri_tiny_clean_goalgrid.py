"""Target-point goal grid on the clean no-distillation student. Submittable.

VADLAW_etri_tiny_clean_nodistill.py + goal_grid_size=(5, 5), nothing else, so
the pair (stage2_clean_nodistill vs stage2_clean_goalgrid) measures the goal
grid alone. [사용자 2026-09-15 설계 그림]

GRID
5s destination (target point) binned over forward -5..110m x lateral
-25..25m, 5x5 (23m x 10m cells). Outside the range: 0.06% of val, 0.51% of
train, clamped to the border cell. Planner head 7x6x2 -> 7x(5x5)x6x2.

HOW IT IS TRAINED
  - goal_cls_head predicts a cell distribution PER COMMAND MODE (7x25).
  - ego_fut_preds = each mode's 25 trajectories mixed by its own
    probabilities, in training and at inference alike, so the planning loss
    trains the exact submitted output and reaches the classifier.
  - loss_goal_cls: CE of the commanded mode's distribution against the cell
    the ground-truth target point falls in.
  - loss_goal_cell_traj: L1 of that cell's own trajectory (commanded mode)
    against the GT 3s trajectory, which keeps cells distinct.

COMPLIANCE
The target point is read only under self.training, as a label. Inference
never reads it; each mode's classifier chooses. Checked by audit_pipeline.py.

Offline proxy (tools/goal_grid_value.py, linear model, not the network): a
perfectly known 5x5 cell cut L2 0.2234 -> 0.2207. The network may use it
differently; this run measures that.
"""

_base_ = ['./VADLAW_etri_tiny_clean_nodistill.py']

model = dict(
    pts_bbox_head=dict(
        goal_grid_size=(5, 5),
        goal_grid_range=(-5.0, 110.0, -25.0, 25.0),
        goal_grid_weight=0.5,
        # 0.1, not 1.0: measured on real batches at initialization,
        # loss_goal_cell_traj reads ~0.14 while loss_plan_reg reads ~0.014
        # (VAD's planning L1 averages over all 7 modes). At 1.0 the per-cell
        # term would outweigh the loss on the submitted output 10x.
        goal_cell_traj_weight=0.1,
        goal_grid_select='soft',
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_clean_goalgrid 5x5 (TP label, per-command select)')),
    ])
