"""Fast-eval config for VADLAW_etri_tiny_kd_nolcf_split_distill8_3f.py.

Same as VADLAW_etri_tiny_fast_eval_split_distill8.py plus
aux_bev_motion_frames=3, which is NOT optional here: it widens
aux_bev_motion_head's input from 2*desc_dim to 3*desc_dim. Getting it wrong
size-mismatches that head, and since ego_status_est_net reads the same
descriptor, the status slot feeding the decoder would be wrong too --
silently, because mmcv loads with strict=False.

Run this with --frame-offsets 0,-5,-10 and --bev-only-history. Two frames
would leave the acceleration block permanently zeroed (prev_bev2 never
populates), which evaluates a different model from the one trained.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_split_distill8.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion_frames=3,
    ))
