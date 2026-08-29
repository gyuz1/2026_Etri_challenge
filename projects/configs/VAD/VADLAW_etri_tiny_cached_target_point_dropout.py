"""Stage-2 VADLAW, target_point-dropout variant: short fine-tune to correct
the ego_target_point shortcut confirmed via ablation (2026-08-25,
tools/eval_holdout_l2_shortcut_ablation.py against the existing 301/75-
split stage2 checkpoint):

    baseline (real images + real target_point):  L2 avg 0.114239 m
    images zeroed, target_point real:             L2 avg 0.120686 m ( 1.06x)
    target_point zeroed, images real:              L2 avg 5.454110 m (47.75x)

I.e. the model was essentially point-to-point navigating off the 5s goal
point and barely using vision -- removing target_point collapsed L2 47x
while removing images barely moved it. VAD_head.py's new
`target_point_dropout` (default 0, so a no-op everywhere else) withholds
ego_target_point for a fraction of TRAINING samples only (never at
inference -- see VAD_head.py forward()'s `self.training` guard), forcing
the planning head to stay accurate from vision alone often enough that it
can't fully outsource trajectory shape to the goal point.

Warm-started from the already-converged (non-dropout) stage2 checkpoint,
same short-fine-tune pattern as VADLAW_etri_tiny_cached_kd.py -- this is a
correction to an existing failure mode, not a from-scratch re-learn, so a
short run at a reduced LR is enough. Only stage2 needs to change: stage1's
config has all loss_plan_* weights at 0.0
(VAD_etri_tiny_stage1.py), so target_point_encoder never received
gradient there regardless of this checkpoint's stage1 origin -- confirmed
before writing this config, not assumed.

STATUS: code + config only, not yet launched. Verify the actual dropout
rate (0.5 below is a starting guess, not tuned) and re-run
eval_holdout_l2_shortcut_ablation.py against the resulting checkpoint to
confirm the fix actually reduced the 47x collapse before trusting it.
"""

_base_ = ['./VADLAW_etri_tiny_cached.py']

load_from = 'work_dirs/stage2_etri_split_301_75/epoch_12.pth'
resume_from = None

total_epochs = 5
optimizer = dict(lr=1e-5)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
evaluation = dict(interval=total_epochs + 1, metric='bbox', map_metric='chamfer')

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                               name='stage2_target_point_dropout')),
    ])

model = dict(
    pts_bbox_head=dict(
        target_point_dropout=0.5,
    ))
