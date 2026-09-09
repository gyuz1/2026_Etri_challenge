"""Fast-eval config for VADLAW_etri_tiny_kd_lcfon_diag.py -- the ego_lcf-ON,
target-point-FREE teacher that feature distillation reads from.

Diagnostic only, never submittable: ego_lcf feeds the decoder directly.

Built from the plain fast_eval base rather than the *_nolcf chain, which
forces ego_lcf OFF. Every setting below either changes the inference graph
or changes a module's shape, so a mismatch would either silently evaluate a
different model or size-mismatch on load:

- use_ego_lcf_status / ego_lcf_feat_idx: ego status must actually reach the
  decoder, exactly as in training. Omitting these evaluates a model that
  never sees the input it was trained on.
- ego_fut_dec_hidden_dim=512: the checkpoint's first Linear is (520 -> 512).
  Left unset it would default to in_dim (520), so the whole decoder loads at
  the wrong width.
- prism_posterior_lcf_idx=(0,1,4,7): the posterior never runs at inference,
  but the module is still BUILT at this width, and the checkpoint carries
  weights for it. Getting this wrong is exactly the failure that produced a
  size mismatch on prism_posterior_net.0.weight once already.
- bev_refine_steps=3 / bev_residual_refine: 3-stage cascaded BEV refine runs
  during inference. Fewer stages means the extra learned corrections never
  fire.
- aux_bev_motion=True with idx (0,1,4,7): the head is built (and carries
  checkpoint weights) even though its output is a train-only loss target
  here, since aux_bev_motion_feedback is off.

target_point_shortcut stays off, matching the teacher: this teacher
deliberately does not use the goal, so that what it learns is transferable
to a student that can never see one.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval.py']

model = dict(
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        ego_fut_dec_hidden_dim=512,
        prism_posterior_lcf_idx=(0, 1, 4, 7),
        bev_refine_steps=3,
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 4, 7),
    ))
