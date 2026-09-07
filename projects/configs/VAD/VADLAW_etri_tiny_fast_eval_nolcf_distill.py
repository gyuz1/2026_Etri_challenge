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
