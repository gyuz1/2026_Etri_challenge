"""Distilled student, trained with every dropout OFF. Submittable.

VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py + disable_dropout=True.
The frozen teacher already runs in eval mode (dropout off), so this also puts
student and teacher features in the same regime. Why: see
VADLAW_etri_tiny_nodistill_nodrop_ft.py (measured 2026-09-15).
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py']

model = dict(disable_dropout=True)
