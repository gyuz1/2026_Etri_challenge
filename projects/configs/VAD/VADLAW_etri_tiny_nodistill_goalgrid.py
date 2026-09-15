"""No-distillation student + goal-grid planning. Submittable.

Identical to VADLAW_etri_tiny_nodistill_best.py except goal_grid_size, so the
pair measures the goal grid alone.

WHAT IT DOES
ego_fut_decoder emits one trajectory per (command, goal cell) over a coarse
grid of 5s destinations -- forward -5..110m in 7 bins (16.4m), lateral
-25..25m in 3 bins (16.7m), goal_grid_range in VAD_head. goal_cls_head picks
the cell from ego_feats. Collapsed inside the head, so ego_fut_preds keeps
[B, 7, 6, 2].

HOW GROUND TRUTH IS USED [사용자 2026-09-15: GT 는 커멘드처럼, 위배 안 되는 선에서]
Training: the target point names which cell's trajectory is supervised and is
the goal_cls_head label -- the same role the GT command plays in choosing the
supervised mode. Inference: the target point is never read (gate:
self.training, checked by audit_pipeline.py); goal_cls_head alone chooses,
by soft probability weighting. Q&A A1 forbids selecting among generated
trajectories with information the network was not given.

This grid is a LABEL space only. The BEV the encoder sees is unchanged at
-30..30m x -15..15m, so cells beyond +30m are chosen from ego state and
scene cues inside the BEV, not from seeing that far.

Measured on the val split: a constant-acceleration extrapolation of the
current state alone lands in the right 7x3 cell 80.4%, adjacent 99.9%.
"""

_base_ = ['./VADLAW_etri_tiny_nodistill_best.py']

model = dict(
    pts_bbox_head=dict(
        goal_grid_size=(7, 3),
        goal_grid_range=(-5.0, 110.0, -25.0, 25.0),
        goal_grid_weight=0.5,
        goal_grid_select='soft',
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name='stage2_STUDENT_no-distill + goal grid 7x3 (-5..110 x -25..25)')),
    ])
