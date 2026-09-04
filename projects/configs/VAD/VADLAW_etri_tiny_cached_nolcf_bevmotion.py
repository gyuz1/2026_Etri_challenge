"""Stage-2 VADLAW, ego_lcf OFF -- same as VADLAW_etri_tiny_cached_nolcf_prism2.py
(aux_ego_motion/aux_long_horizon dropped, PRISM S=2 + prev_bev_dropout +
echo_cycle + cascaded refine + EMA + col/dir/bound loss kept) plus one new
addition: aux_bev_motion.

aux_bev_motion regresses current ego status directly from bev_embed (the
BEV encoder's raw output -- the single shared tensor detection, map, agent
and planning heads all branch off), rather than from ego_feats (a late,
planning-only feature aux_ego_motion used). Gradient from this loss reaches
the BEV encoder itself, which is a closer match to the organizers'
"improves common features across multiple tasks" allowance than
supervising a planning-specific feature. Same compliance shape as every
other aux_* head: ego_lcf_target is consumed only as a regression TARGET
on a branch with no path back into any decoder output -- see VAD_head.py's
aux_bev_motion constructor comment.
"""

_base_ = ['./VADLAW_etri_tiny_cached_nolcf_prism2.py']

model = dict(
    pts_bbox_head=dict(
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 4, 7),
        aux_bev_motion_weight=0.5,
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
                      'col_dir_bound_loss, aux_bev_motion)'))),
    ])
