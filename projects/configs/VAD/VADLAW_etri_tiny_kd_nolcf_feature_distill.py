"""Feature-level distillation into the compliant planner, from a frozen
ego_lcf-ON teacher trained on the SAME KD stage1 lineage
(VADLAW_etri_tiny_kd_lcfon_diag.py).

WHAT IS MATCHED
ego_fut_decoder's first Linear+ReLU output ('ego_plan_hidden'), not
ego_feats. The teacher builds ego_feats as
cat([agent_query, map_query, raw ego_lcf]) -- ego status sits in its own
trailing columns that never passed through the agent/map cross-attention,
and its decoder reads speed straight out of them, so nothing pressures its
leading 2*D to encode motion at all. Matching that would teach the student
the opposite of what aux_bev_motion asks of it. One Linear later, the
ego_lcf columns have been mixed into every hidden unit, and a student
with no such columns has to reconstruct their contribution from vision --
which is the transfer worth having. Both hidden layers are 512-wide (the
teacher sets ego_fut_dec_hidden_dim=512; the student leaves it unset and
its ego_fut_dec_in_dim is 512), so the match needs no slicing.

WHY A TP-FREE TEACHER
The other teacher on hand (VADLAW_etri_tiny_lcfon_tpshortcut_teacher.py)
also turns on target_point_shortcut, which adds a goal residual across the
full width of ego_feats and conditions the ego agent/map attention on the
goal. That contaminates everything downstream with information the student
categorically cannot recover -- target_point is exogenous routing data, not
something inferable from pixels. A cosine loss against it would plateau
high and, at any meaningful weight, fight the trajectory loss rather than
help it. Removing target_point costs the teacher raw L2 but makes what
remains actually transferable. Same KD stage1 lineage as the student is a
second reason to prefer this teacher: closer weight space, more
comparable activations.

WHY WEIGHT 0.3
The cosine term does not decay to zero the way the task losses do (the
student can never fully match a teacher that saw real ego status), while
loss_plan_reg reaches ~0.006 late in training. At weight 1.0 the
distillation term would dominate the gradient exactly when trajectory
accuracy matters most.

WHY aux_bev_motion_temporal
The inherited default estimates ego motion from a single frame's
GLOBAL-MEAN-pooled bev_embed, which is provably shift-invariant -- it
cannot see displacement, so it can only learn appearance correlates of
speed, which can_bus already supplies directly. The temporal variant
contrasts regionally-pooled descriptors of the current and previous BEV
instead, asking bev_embed for temporal consistency, which can_bus does
not supply. Distinct from the aux_bev_motion_feedback experiment that
measured worse (0.5635 -> 0.6419): that one fed the estimate into the
decoder as an input channel; this stays a loss-only target.

privileged_distill (trajectory-level, in-graph privileged head) stays off:
measured 0.4897m against a 0.4885m pre-distill baseline. Its privileged
head did fit better than the deployed decoder during training (0.0084 vs
0.0130 at epoch 6), so the advantage existed -- it just did not survive
being squeezed through 12 output numbers. That is the specific failure
this config's richer 512-d target is meant to address.
"""

_base_ = ['./VADLAW_etri_tiny_kd_nolcf_distill.py']

model = dict(
    pts_bbox_head=dict(
        privileged_distill=False,
        aux_bev_motion_temporal=True,
    ),
    feature_distill_teacher_cfg=(
        'projects/configs/VAD/VADLAW_etri_tiny_kd_lcfon_diag.py'),
    feature_distill_teacher_ckpt=(
        'work_dirs/stage2_kd_lcfon_diag/epoch_12.pth'),
    feature_distill_weight=0.3,
)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage2_kd_nolcf FEATURE_DISTILL (post-fusion hidden '
                      '<- frozen ego_lcf-ON TP-free teacher, w=0.3, '
                      '+aux_bev_motion_temporal)'))),
    ])
