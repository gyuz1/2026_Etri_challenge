"""Stage-2 VADLAW, ego_lcf OFF -- same as VADLAW_etri_tiny_cached_nolcf_bevmotion.py
plus one new addition: prism_posterior_lcf_idx.

PRISM's posterior network currently sees only the privileged 0-5s GT
future trajectory. It's a train-only branch -- discarded at inference,
where only the vision-only prior is used (VAD_head.py's
prism_latent_supervision constructor comment) -- so it's exactly the
kind of place the organizers' "indirect use that improves shared
features" allowance covers, same compliance shape as every other ego_lcf
use in this file. This lets the posterior ALSO condition on current
ego_lcf status (velocity/accel/yaw-rate via idx (0,1,4,7), matching
aux_bev_motion_idx), on top of the future trajectory it already sees --
a stronger privileged signal for the KL term to pull the vision-only
prior toward, without adding anything to the inference-time input.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_bevmotion.py']

model = dict(
    pts_bbox_head=dict(
        prism_posterior_lcf_idx=(0, 1, 4, 7),
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf (prism_s2+lcf_posterior, '
                      'prev_bev_dropout, echo_cycle, cascade_refine_x3, '
                      'ema, col_dir_bound_loss, aux_bev_motion)'))),
    ])
