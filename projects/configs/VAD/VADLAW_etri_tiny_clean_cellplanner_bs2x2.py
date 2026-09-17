"""Batch ablation of VADLAW_etri_tiny_clean_cellplanner.py: 2 samples per GPU
(total 4 on 2 GPUs) instead of 1 (total 2). Nothing else changes -- same lr,
warmup, EMA, schedule and data -- so the only variable is the batch size.
Compare against stage2_cellplanner_v1 by epoch (half the iterations per epoch).
"""

_base_ = ['./VADLAW_etri_tiny_clean_cellplanner.py']

data = dict(samples_per_gpu=2)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_cellplanner_bs2x2_v1 (batch ablation)')),
    ])
