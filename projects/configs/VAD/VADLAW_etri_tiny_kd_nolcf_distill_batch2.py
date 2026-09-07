"""Same recipe as VADLAW_etri_tiny_kd_nolcf_distill.py, testing whether
samples_per_gpu can go from 1 to 2 (global batch 2 -> 4). The base config's
own comment records batch=4 (samples_per_gpu=2 on this project's earlier
2-GPU DDP setups is global 4, or per this repo's actual convention,
samples_per_gpu itself IS the per-GPU value -- global = samples_per_gpu *
num_gpus) OOM'ing before; this run's own live measurement showed only
12.4/24GB used, so a smaller step up (1 -> 2 per GPU, not straight to 4)
is being tried instead of assuming the old OOM still applies to the
current, heavier technique stack.

lr scaled linearly with the resulting global batch: base 5e-5 is
calibrated for global batch=2 (samples_per_gpu=1 x 2 GPUs), so global
batch=4 (samples_per_gpu=2 x 2 GPUs) gets 1e-4.

Diagnostic only -- if this OOMs or produces batch>1-specific errors
(plausible given how much of this codebase's custom module code assumes
batch=1: several such bugs were found and fixed this session), the
answer is no and VADLAW_etri_tiny_kd_nolcf_distill.py resumes from its
saved epoch_7.pth at the original batch=1.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_distill.py']

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
                name=('stage2_kd_nolcf DISTILL batch=2/GPU diagnostic '
                      '(lr=1e-4)'))),
    ])
