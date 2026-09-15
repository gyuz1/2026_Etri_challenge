"""No-distillation student with every train/inference mismatch removed.

VADLAW_etri_tiny_nodistill_best.py plus the same four switches as
VADLAW_etri_tiny_clean_student.py (read that docstring for why). Paired with
the clean student it measures distillation alone; paired with
stage2_nodistill_best it measures the mismatch removal. Submittable.
"""

_base_ = ['./VADLAW_etri_tiny_nodistill_best.py']

model = dict(
    disable_dropout=True,
    prev_bev_dropout=0.0,
    pts_bbox_head=dict(
        ego_status_est_dropout=0.0,
        prism_latent_supervision=False,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_nodistill_clean (no train/eval mismatch)')),
    ])
