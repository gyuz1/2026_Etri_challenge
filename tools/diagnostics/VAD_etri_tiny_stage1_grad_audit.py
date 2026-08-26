_base_ = [
    '../../projects/configs/VAD/VAD_etri_tiny_stage1.py',
]

# The wrapper builds the untouched production dataset and exposes only enough
# samples for 300 global DDP iterations (600 samples with two ranks).
custom_imports = dict(
    imports=['tools.diagnostics.vad_grad_update_audit'],
    allow_failed_imports=False)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='DiagnosticSubsetDataset',
        base_config=(
            '/workspace/VAD/projects/configs/VAD/'
            'VAD_etri_tiny_stage1.py'),
        max_samples=600))

# One epoch over the 600-sample wrapper is exactly 300 iterations on two GPUs.
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
workflow = [('train', 1)]

# Load model weights only.  Optimizer state creation is deliberately observed
# afresh so it directly identifies which parameters receive a gradient.
load_from = '/workspace/VAD/work_dirs/stage1_etri/epoch_4.pth'
resume_from = None

work_dir = '/tmp/vad_grad_update_audit'
checkpoint_config = None
evaluation = dict(interval=2)
log_config = dict(interval=25, hooks=[dict(type='TextLoggerHook')])

custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(
        type='GradientUpdateAuditHook',
        output_dir='/tmp/vad_grad_update_audit',
        write_interval=50,
        expected_iters=300,
        inspect_iters=(1, 2, 10, 50, 100, 300),
        priority='LOWEST'),
]
