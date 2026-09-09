"""Fast-eval config for VADLAW_etri_tiny_kd_nolcf_distill.py (and its
batch=2 diagnostic variant, same checkpoint architecture).

privileged_head (privileged_distill) and prism_posterior_lcf_idx are both
train-only branches -- gated on self.training / discarded at inference,
same as documented in VADLAW_etri_tiny_cached_nolcf_bevmotion_prismlcf.py.
Neither touches ego_fut_decoder's input width or any other inference-path
shape, so this is architecturally identical to
VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py -- no new model= overrides
needed, this file exists only so the eval config's name matches the
train config's name per convention.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        # prism_posterior_net is built unconditionally in _init_layers()
        # regardless of self.training, so its shape (and therefore whether
        # the checkpoint's weights for it load at all) depends on this
        # matching the training config -- even though the module itself
        # never runs at inference. Without this, load_state_dict silently
        # size-mismatches on prism_posterior_net.0.weight.
        prism_posterior_lcf_idx=(0, 1, 4, 7),
    ))
