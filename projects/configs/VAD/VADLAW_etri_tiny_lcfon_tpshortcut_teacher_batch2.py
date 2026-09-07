"""Same recipe as VADLAW_etri_tiny_lcfon_tpshortcut_teacher.py, run at
samples_per_gpu=2 (global batch 4 instead of 2) on A5000, in parallel with
the batch=1 copy already training on the local 3090.

Purpose: not a correctness diagnostic this time (that question was already
settled by VADLAW_etri_tiny_kd_nolcf_distill_batch2.py's clean 200-iter
run) -- this is a second, independently-seeded copy of the strongest
non-compliant teacher recipe, run faster (~18h vs ~24h) on spare A5000
capacity once its current job frees up. Whichever of the two (3090 batch=1
or this) finishes with the better L2 becomes the actual distillation
source; the other is a discardable extra data point, not wasted since
A5000 would otherwise sit idle waiting for the 3090 run anyway.

lr scaled linearly with the resulting global batch, same as the distill
batch=2 diagnostic: base 5e-5 (global batch=2) -> 1e-4 (global batch=4).
"""

_base_ = ['./VADLAW_etri_tiny_lcfon_tpshortcut_teacher.py']

data = dict(samples_per_gpu=2)

optimizer = dict(lr=1e-4)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_TEACHER_NONCOMPLIANT batch=2/GPU on A5000 '
                      '(lr=1e-4, parallel copy of the 3090 batch=1 run)'))),
    ])
