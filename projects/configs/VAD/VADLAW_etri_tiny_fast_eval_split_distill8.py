"""Fast-eval config for VADLAW_etri_tiny_kd_nolcf_split_distill8.py --
the Scheme-A student v2. Compliant: no ego_lcf, no target point.

Same rules as VADLAW_etri_tiny_fast_eval_split_distill.py (read that file
for why each line is required), with two settings updated to match v2:

- ego_status_est_dim=8 instead of 64. This sets ego_fut_dec_in_dim to
  512+8=520, so getting it wrong size-mismatches every ego_fut_decoder
  weight -- and mmcv loads with strict=False, so it would silently evaluate
  a randomly initialized planner rather than raise.
- aux_bev_motion_idx now has 6 entries (acceleration added), which changes
  aux_bev_motion_head's output width. That head is built here because
  ego_status_est_net reads the descriptor its block constructs, so the
  shape has to match the checkpoint.

aux_bev_motion_norm is deliberately NOT repeated: it only scales a training
loss and builds no parameters, so it cannot affect what loads or what runs
at inference.

Verify with tools/diff_eval_config.py before trusting any number from here.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        prism_posterior_lcf_idx=(0, 1, 4, 7),
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_temporal=True,
        ego_status_est_dim=8,
        ego_fut_dec_hidden_dim=512,
    ))
