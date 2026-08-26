# Preparing the epoch-4 resume checkpoint

`prepare_resume_checkpoint.py` adapts a copy-on-write MMCV checkpoint after
the final batch size has been selected. Its default mode is a read-only dry
run. `--write` requires a distinct, non-existing output path; the tool writes
through a temporary file, atomically renames it, reloads it for structural
verification, and checks that the source inode, size, and mtime did not change.
Use `--verify-source-sha256` when a slower byte-for-byte source check is wanted.
Writing also requires `--runtime-config`. The config is fully resolved,
validated against the epoch/batch/LR arguments, assigned the new GPU count,
and embedded in checkpoint metadata.

For MMCV 1.4's epoch cosine scheduler, the first resumed LR is

```text
factor = min_lr_ratio + (1 - min_lr_ratio)
         * (1 + cos(pi * checkpoint_epoch / new_max_epochs)) / 2
new_initial_lr[group] = saved_lr[group] / factor
```

The saved optimizer-group `lr` is deliberately preserved. Only
`initial_lr` is changed, independently for every group, so the first resumed
`before_train_epoch` calculation has no LR jump and the backbone's 0.1 LR
multiplier is retained. This does not apply linear LR scaling for a larger
batch; that would be a separate optimization decision.

The ETRI train sampler has one flag group of 19,608 samples. Its per-rank
DataLoader length is `ceil(19608 / (world_size * samples_per_gpu))`, and the
corrected checkpoint iteration is `checkpoint_epoch * iters_per_epoch`.

Example dry run for two GPUs and batch/GPU 4:

```bash
python tools/optimization/prepare_resume_checkpoint.py \
  work_dirs/stage1_etri/epoch_4.pth \
  --max-epochs 48 --group-sizes 19608 \
  --world-size 2 --samples-per-gpu 4
```

The final 48-epoch runtime config must explicitly agree on all of these:

```text
total_epochs = 48
runner.max_epochs = 48
model.pts_bbox_head.tot_epoch = 48
evaluation.interval = 49
checkpoint_config.max_keep_ckpts = 48
```

The nested head value needs an explicit override because the base config has
already resolved `tot_epoch=8` before a derived config is merged.

Embedding the resolved runtime config matters for later restarts: MMCV 1.4's
`resume()` replaces the runner metadata with checkpoint metadata, and that
metadata wins over the checkpoint hook's fresh config when the next epoch is
saved. Without replacement, epoch 5 and later would misleadingly retain the
old 8-epoch config and could apply the wrong world-size iteration correction.

The epoch-4 optimizer has 587 parameter groups but only 386 Adam state
entries. The tool consequently reports 201 parameter IDs without state: 119
are frozen tensors, while the gradient audit identified the remaining 82 as
trainable tensors whose gradients had been detached by the autocast-cache
issue. Those 82 resume without Adam moments and acquire state lazily on their
first fixed optimizer step. The other trained tensors retain their moments.
The utility never fabricates, drops, or resets optimizer state.
