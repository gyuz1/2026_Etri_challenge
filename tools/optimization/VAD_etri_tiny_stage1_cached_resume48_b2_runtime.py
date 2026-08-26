"""Temporary b2 runtime config used only to prepare/execute the b2 sweep.

Batch size is not yet a production decision.  After the sweep, create the
final production config and a newly prepared checkpoint for the chosen batch.
"""

_base_ = [
    '../../projects/configs/VAD/VAD_etri_tiny_stage1_cached.py'
]

# These values must stay in sync so MMCV's resumed cosine schedule has no LR
# jump and future benchmark metadata cannot recover the original eight-epoch
# schedule.
total_epochs = 48
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
model = dict(pts_bbox_head=dict(tot_epoch=total_epochs))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train_dataloader=dict(persistent_workers=True))

evaluation = dict(interval=total_epochs + 1)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

find_unused_parameters = False

