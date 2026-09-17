"""Batch ablation of VADLAW_etri_tiny_clean_cellplanner.py: 2 samples per GPU
(total 4 on 2 GPUs) instead of 1 (total 2). Everything else is identical.

Kept equal per sample: learning rate (x2), warmup length (iterations / 2) and
the EMA averaging window (momentum x2). Compare against stage2_cellplanner_v1
by epoch, not iteration.
"""

_base_ = ['./VADLAW_etri_tiny_clean_cellplanner.py']

data = dict(samples_per_gpu=2, workers_per_gpu=4)

optimizer = dict(lr=1e-4)

lr_config = dict(warmup_iters=250)

custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(type='EMAHook', momentum=0.0004, priority='HIGH'),
]

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_cellplanner_bs2x2_v1 (batch ablation)')),
    ])
