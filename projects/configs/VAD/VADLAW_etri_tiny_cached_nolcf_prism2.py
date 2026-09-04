"""Stage-2 VADLAW, ego_lcf OFF -- same recipe as
VADLAW_etri_tiny_cached_nolcf_aux.py MINUS the two aux_* heads
(aux_ego_motion, aux_long_horizon).

Reason for dropping them: both aux_long_horizon and PRISM's posterior net
read/shape the same ego_feats tensor toward encoding the same underlying
signal (the 5s GT future) -- PRISM via a KL-regularized latent that gets
injected into the decoder input (feats_i = ego_feats + prism_z_proj(z)),
aux_long_horizon via a direct L1 regression that only backpropagates into
ego_feats with no injection path of its own. PRISM's pathway actually
reaches the decoded trajectory; aux_long_horizon's is a pure regularizer
demanding a different (deterministic, unbounded) encoding of the same
information on the same tensor. Measured symptom: grad_norm-nan rate in
the first ~5k iterations of nolcf_aux's actual DDP+fp16 training run was
~47%, versus ~32% for both the ego_lcf-ON run and the plain nolcf run
(neither of which have aux_long_horizon) at the same point -- PRISM alone
is not the source (its own loss_prism_kl curve is smooth in every run),
so aux_long_horizon is the leading suspect. Not conclusively confirmed (a
quick isolated nan-rate check without matching the real Fp16OptimizerHook
dynamic loss scaling came back 0% for both settings, i.e. inconclusive,
not exonerating) -- this config is the actual controlled test.

aux_ego_motion dropped too, for a clean before/after: keeping it alone
would leave two simultaneous changes (dropping long_horizon AND keeping
ego_motion) muddying whether any L2 change traces to the PRISM conflict
specifically or to aux supervision in general.

Kept: prev_bev_dropout, echo_cycle_weight, cascaded bev_refine (3 stages)
+ EMA, VAD's loss_plan_col/dir/bound (re-enabled, lane_bound_cls_idx bug
fixed) -- see VADLAW_etri_tiny_cached_nolcf_aux.py for each one's own
rationale, unchanged here. Also inherits the ego_fut_decoder merge fix
(--override-prefixes-from-world-model) via the same
stage2_init_merged_nolcf.pth load_from.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_aux.py']

model = dict(
    pts_bbox_head=dict(
        aux_ego_motion=False,
        aux_long_horizon=False,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_nolcf (prism_s2, prev_bev_dropout, '
                      'echo_cycle, cascade_refine_x3, ema, '
                      'col_dir_bound_loss)'))),
    ])
