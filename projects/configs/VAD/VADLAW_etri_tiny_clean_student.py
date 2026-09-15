"""Distilled student with every train/inference mismatch removed. Submittable.

VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py plus four switches. Each
one made training see something inference never does:

  disable_dropout=True           64 nn.Dropout (p=0.1). MEASURED 2026-09-15:
                                 with dropout off (inference) the vision speed
                                 reads +0.52 m/s (+5%) and the plan +0.29 m/s;
                                 all dropout on cut L2@3s 0.732 -> 0.553.
  prev_bev_dropout=0.0           was 0.5: half of training steps had no history
                                 BEV, while every scored test frame has one.
  ego_status_est_dropout=0.0     was 0.3: the planner's status slot was zeroed
                                 on 30% of steps and never at inference.
  prism_latent_supervision=False the planner was trained on a latent drawn from
                                 a posterior that sees the GT 0-5s future, and
                                 run at inference on the prior mean.

The last three are unmeasured individually; they are removed on the same
principle, which tools/audit_pipeline.py now enforces (3c). The frozen teacher
runs in eval mode, so its features are deterministic too.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py']

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
                              name='stage2_STUDENT_clean (distill, no train/eval mismatch)')),
    ])
