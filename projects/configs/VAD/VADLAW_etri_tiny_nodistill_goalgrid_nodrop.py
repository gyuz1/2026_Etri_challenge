"""Goal-grid student (no distillation), trained with every dropout OFF.

VADLAW_etri_tiny_nodistill_goalgrid.py + disable_dropout=True. Submittable.
Why dropout off: see VADLAW_etri_tiny_nodistill_nodrop_ft.py -- with dropout
on in training the speed read-out and the planned speed come out ~5% fast at
inference (measured 2026-09-15).
"""

_base_ = ['./VADLAW_etri_tiny_nodistill_goalgrid.py']

model = dict(disable_dropout=True)
