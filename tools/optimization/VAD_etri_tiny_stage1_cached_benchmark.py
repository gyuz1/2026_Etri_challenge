"""Finite 2-GPU throughput benchmark; never use as a production config."""

_base_ = [
    './VAD_etri_tiny_stage1_cached_resume48_b2_runtime.py'
]

custom_imports = dict(
    imports=[
        'tools.cache.etri_geometry_cache',
        'tools.optimization.vad_throughput_benchmark',
    ],
    allow_failed_imports=False)

# Match the prepared 48-epoch resume checkpoint. The custom runner exits after
# the finite measurement window, so max_epochs controls scheduler state only.
runner = dict(
    type='VADThroughputBenchmarkRunner',
    max_epochs=48,
    benchmark_warmup_iters=50,
    benchmark_measure_iters=300,
    expected_world_size=2,
    expected_start_epoch=4,
    expected_samples_per_gpu=2,
    expected_workers_per_gpu=2)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_dataloader=dict(persistent_workers=True))

find_unused_parameters = False

# The benchmark reads only the explicitly prepared resume checkpoint and must
# not write any checkpoint itself.
checkpoint_config = None
evaluation = dict(interval=999999)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
