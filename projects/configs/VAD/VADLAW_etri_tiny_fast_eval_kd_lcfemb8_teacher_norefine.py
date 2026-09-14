"""Eval config for VADLAW_etri_tiny_kd_lcfemb8_teacher_norefine.py."""

_base_ = ['./VADLAW_etri_tiny_fast_eval_kd_lcfemb8_teacher_best.py']

model = dict(pts_bbox_head=dict(bev_residual_refine=False))
