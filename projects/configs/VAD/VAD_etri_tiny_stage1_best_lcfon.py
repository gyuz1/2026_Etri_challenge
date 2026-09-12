"""Stage 1, best known configuration, ego_lcf ON -- the TEACHER's donor.

Identical to VAD_etri_tiny_stage1_best_nolcf.py except ego_lcf_feat_idx
returns to [0..7]. Read that file for why each setting is what it is.

WHY A SEPARATE LINEAGE IS NEEDED AT ALL
The teacher takes ego status into ego_feats, so its ego_fut_decoder is
embed_dims*2 + 8 = 520 wide, while the compliant student's is 512. A donor
trained at one width cannot transfer into the other. Training the decoder
with those 8 columns present from stage 1 epoch 1 is the point: when the
columns were instead zero-padded on at stage 2, they reached only 0.147x
the scene columns' magnitude in 12 epochs, against 1.564x by epoch 4 when
stage 1 trained them -- and the 0.2166 pre-ban record came from a lineage
whose decoder had ego columns from the start.

ego_fut_dec_hidden_dim=512 MUST be set here, and is the one thing that is
not simply inherited. It defaults to ego_fut_dec_in_dim, which turning
ego_lcf on just widened to 520 -- but every stage-2 consumer builds the
decoder at hidden 512, because a teacher's ego_plan_hidden has to be the
same width as the compliant student's. Leaving it at the default makes all
three decoder layers mismatch, and mmcv loads with strict=False, so nothing
raises: 48 epochs of training would be silently dropped at stage 2. This
already happened once on this exact config family and was caught at epoch 1.

NOT SUBMITTABLE. Everything downstream of this donor reads real ego status
directly and exists only to produce a distillation target.
"""

_base_ = ['./VAD_etri_tiny_stage1_best_nolcf.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        ego_fut_dec_hidden_dim=512,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage1_BEST_lcfon (GT planning, 3-frame, grid8, '
                      'accel targets, normalized, future speed)'))),
    ])
