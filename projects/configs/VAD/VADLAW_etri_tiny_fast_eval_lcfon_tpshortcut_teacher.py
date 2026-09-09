"""Fast-eval config for VADLAW_etri_tiny_lcfon_tpshortcut_teacher.py.

Diagnostic-only, non-compliant -- ego_lcf feeds the decoder directly and
target_point conditions generation via attention + residual. Never
submit; this exists purely to measure the teacher's own L2/T_infer as a
reference point and to sanity-check the feature-distillation source.

Built from the plain fast_eval base (not the *_nolcf chain, which forces
ego_lcf OFF) since this teacher needs it ON. Mirrors every
architecture-affecting setting from the teacher's own resolved train
config so checkpoint tensor shapes line up:
- use_ego_lcf_status / ego_lcf_feat_idx: real ego_lcf as decoder input.
- target_point_shortcut(_mode): TP conditions ego_agent/map attention and
  is added as a residual into ego_feats.
- prism_posterior_lcf_idx: train-only (posterior never runs at
  inference), but the module is still built at this width, so the
  checkpoint's prism_posterior_net weights need it to match or
  load_state_dict size-mismatches on a tensor eval never uses.
- bev_refine_steps=3: inherited from the bevmotion lineage's cascaded
  BEV refine -- an inference-time architectural choice, unlike the
  train-only settings above.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval.py']

model = dict(
    use_ego_lcf_status=True,
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7],
        prism_posterior_lcf_idx=(0, 1, 2, 3, 4, 7),
        target_point_shortcut=True,
        target_point_shortcut_mode='both',
        bev_refine_steps=3,
    ))
