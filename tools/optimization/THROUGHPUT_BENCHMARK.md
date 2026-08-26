# Cached VAD throughput sweep

Run only after
`/workspace/VAD/work_dirs/etri_geometry_cache_v1/cache_manifest.json` reports
344 requested and completed scenes. The launcher refuses an incomplete cache,
a prepared checkpoint whose source hash/batch/sampler length do not match, a
non-2-GPU launch, or a timing window other than 50 warmup plus 300 measured
iterations.

First make a new 48-epoch resume checkpoint. This command is intentionally not
run by the sweep; it refuses to overwrite either input or output:

```bash
python tools/optimization/prepare_resume_checkpoint.py \
  work_dirs/stage1_etri/epoch_4.pth \
  --output work_dirs/optimization/epoch_4_benchmark_resume48_b2_v2.pth \
  --max-epochs 48 \
  --group-sizes 19608 \
  --world-size 2 \
  --samples-per-gpu 2 \
  --runtime-config \
    tools/optimization/VAD_etri_tiny_stage1_cached_resume48_b2_runtime.py \
  --verify-source-sha256 \
  --write
```

Then, from `/workspace/VAD` inside `ad2026`:

```bash
python tools/optimization/run_vad_throughput_sweep.py \
  --checkpoint work_dirs/optimization/epoch_4_benchmark_resume48_b2_v2.pth \
  --gpus 0,1 \
  --batches 2 \
  --workers 2,4 \
  --cache-root /workspace/VAD/work_dirs/etri_geometry_cache_v1
```

This first sweep is intentionally the current-config HDD baseline: the cache
above resides on the same rotational HGST disk as the raw dataset.  Storage
identity is recorded in preflight.  If data wait or GPU starvation remains,
copy the same cache bytes to NVMe and rerun the identical sweep to isolate the
storage bottleneck.

The source `work_dirs/stage1_etri/epoch_4.pth` and the separately prepared
checkpoint are hashed and re-hashed after every case; the sweep makes no extra
checkpoint copy. The prepared checkpoint embeds the original epoch-4 SHA-256,
batch size, sampler group sizes, and iterations per epoch. Before launching it
strictly checks all 344 cache entries,
free disk, idle selected GPUs, and the absence of another train/DDP process.
The benchmark config disables checkpoint writes and validation. Results live
under a timestamped directory in `work_dirs/throughput_sweep/`:

The b2 runtime config and prepared checkpoint are benchmark-only artifacts.
After selecting a batch size, create a separate production config and prepare
a new checkpoint with that final batch's sampler length.

- `sweep_summary.json`: comparison of all cases;
- `b2_w*/throughput_summary.json`: 300-iteration aggregate samples/s,
  CPU-observed data/step timings, rank-local peak CUDA allocator memory, and
  non-finite loss records;
- `b2_w*/gpu_monitor.json`: one-second `nvidia-smi` utilization and VRAM
  samples;
- `b2_w*/launcher.log`: the complete DDP log.

The status is `safe` at 3.70 samples/s or faster, `hard_only` from 3.329 to
3.70 samples/s, and `fail` below 3.329 samples/s.

The A/B sweep intentionally uses the same deterministic cuDNN setting for all
worker cases. After choosing the loader setting, a separate non-deterministic
`cudnn_benchmark`/TF32 production-speed experiment may be useful; do not mix
that result into this controlled worker comparison.

To inspect commands without touching the cache or checkpoint:

```bash
python tools/optimization/run_vad_throughput_sweep.py \
  --checkpoint placeholder.pth \
  --dry-run
```
