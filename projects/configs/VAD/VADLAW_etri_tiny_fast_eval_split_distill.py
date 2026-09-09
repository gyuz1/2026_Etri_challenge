"""Fast-eval config for VADLAW_etri_tiny_kd_nolcf_split_distill.py --
the Scheme-A student. Compliant: no ego_lcf, no target point.

Unlike the Scheme-B student, this one is NOT architecturally identical to
the plain nolcf eval. Its 64-d vision-estimated status slot is part of the
INFERENCE graph, so three settings must be repeated here or the evaluated
model is a different network from the trained one:

- ego_status_est_dim=64: builds ego_status_est_net and widens
  ego_fut_dec_in_dim to 576. Without it the decoder is 512-wide and every
  ego_fut_decoder weight size-mismatches on load (strict=False, so it would
  NOT raise -- it would silently evaluate a randomly initialized planner).
- aux_bev_motion_temporal=True: the estimator reads
  cat([current, current - previous]) regional BEV descriptors, so
  aux_bev_in_dim is 2*32*4^2=1024, not embed_dims=256. This also changes
  what the estimator sees at inference, not just a training loss.
- ego_fut_dec_hidden_dim=512: in_dim is now 576, so the default would build
  a 576-wide hidden layer against a 512-wide checkpoint.

ego_status_est_dropout is not repeated: it is gated on self.training, so
the slot is always live at eval -- which is the intended deployment.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        # See VADLAW_etri_tiny_fast_eval_nolcf_distill.py: built
        # unconditionally, so its width must match or the checkpoint's
        # weights for it silently fail to load.
        prism_posterior_lcf_idx=(0, 1, 4, 7),
        # The eval chain leaves aux_bev_motion OFF because for every earlier
        # config it was a train-only loss whose head simply goes unused at
        # inference. Not here: ego_status_est_net reads the descriptor that
        # block builds, so this becomes an inference-path requirement.
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 4, 7),
        aux_bev_motion_temporal=True,
        ego_status_est_dim=64,
        ego_fut_dec_hidden_dim=512,
    ))
