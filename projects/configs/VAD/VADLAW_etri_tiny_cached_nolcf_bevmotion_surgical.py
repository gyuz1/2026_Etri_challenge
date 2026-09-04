"""Stage-2 VADLAW, ego_lcf OFF, same technique stack as
VADLAW_etri_tiny_cached_nolcf_bevmotion.py -- but ego_fut_decoder is
surgically initialized from the ego_lcf-ON stage1's own (48-epoch,
ETRI-domain-adapted) decoder instead of the off-domain nuScenes one.

Bridge experiment: the plain nolcf_bevmotion run's ego_fut_decoder came
from law_pretrained_nus_nolcf.pth (via merge_stage1_world_model.py's
--override-prefixes-from-world-model, needed because stage1's 520-wide
decoder didn't fit stage2's 512-wide slot) -- meaning it had never seen
a single ETRI frame before stage2's 12 epochs. This config instead uses
tools/surgical_ego_fut_decoder_transfer.py to slice stage1's own
520-wide decoder down to a 512-wide-INPUT one (dropping exactly the 8
ego_lcf input columns, an exact operation -- not an approximation --
since a Linear layer's output is a sum over input columns and removing
one term from a sum is exact) while keeping the hidden width at 520 so
the two deeper layers transfer completely unchanged. Result:
ego_fut_decoder starts from something that HAS seen 48 epochs of ETRI
data, just never through the ego_lcf-input path.

Requires ego_fut_dec_hidden_dim=520 to match the donor's (stage1) hidden
width -- see VAD_head.py's ego_fut_dec_hidden_dim constructor comment.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        ego_fut_dec_hidden_dim=520,
    ))

load_from = 'work_dirs/stage1_etri_split_301_75_10hz/stage2_init_merged_surgical_decoder.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf (bevmotion stack, SURGICAL decoder '
                      'transfer from ego_lcf-ON stage1, hidden_dim=520)'))),
    ])
