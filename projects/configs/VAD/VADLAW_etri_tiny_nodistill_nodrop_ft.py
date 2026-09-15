"""nodistill epoch 12, fine-tuned with every dropout OFF. Submittable.

WHY
Measured 2026-09-15: stage2_nodistill_best/epoch_12 scores 0.5018 under test
commands, worse than its own stage-1 donor (0.4807). The cause is a train/eval
feature shift from dropout, not the data or the pipeline:

  val stream, 200 windows        speed-est bias   plan 0.5s bias   L2@3s
  inference as usual (drop off)      +0.52            +0.29        0.732
  BEV-encoder dropout only on        +0.05            +0.29        0.742
  every nn.Dropout on                +0.05            +0.07        0.553

  ruled out, same +0.52: fp16, bev-only history, EMA vs raw weights, cache vs
  JPEG images, frame spacing (index-5 is exactly 5 frames / 500ms).
  TRAINING forward path on train samples: train mode +0.007, eval mode +0.49.

The speed read-out and the planner's speed were both fit on dropout-perturbed
features; with dropout off they read ~5% fast. Running dropout at inference
would make the output random. Instead: continue training with dropout off
so everything recalibrates to the deterministic features inference uses.

WHAT CHANGES
disable_dropout=True (all 64 nn.Dropout -> p=0), load_from nodistill epoch 12,
lr 5e-5 -> 1e-5, 2 epochs. Everything else identical to nodistill_best,
so the pair isolates the dropout fix. Eval config: the nodistill one --
dropout is inactive at eval either way.
"""

_base_ = ['./VADLAW_etri_tiny_nodistill_best.py']

model = dict(disable_dropout=True)

load_from = 'work_dirs/stage2_nodistill_best/epoch_12.pth'

optimizer = dict(lr=1e-5)
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=200,
                 warmup_ratio=1.0 / 3, min_lr_ratio=0.1)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name='stage2_nodistill ep12 + dropout-off fine-tune (2ep, lr 1e-5)')),
    ])
