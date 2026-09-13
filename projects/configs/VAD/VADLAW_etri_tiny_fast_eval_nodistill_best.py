"""Eval config for VADLAW_etri_tiny_nodistill_best.py.

Distillation is train-only, so the network this builds is identical to the
distilled student's eval network. It exists as its own file only so the
train/eval pairing stays one-to-one and audit_pipeline.py's parity check has
something to compare against.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8_3f_fut_g8.py']

model = dict(
    feature_distill_teacher_cfg=None,
    feature_distill_teacher_ckpt=None,
    feature_distill_mode='fused',
    scene_distill_weight=0.0,
    status_distill_weight=0.0,
    feature_distill_weight=0.0,
)
