# LAW_split

301/75-scene split experiment folder for the ETRI 2026 E2E Driving Challenge
(VAD-tiny + LAW). Used to validate the `ego_target_point` shortcut fix
(target_point attention conditioning + corruption + BEV residual
refinement) on the fast hold-out split before porting to the full 376-scene
run.

See `projects/configs/VAD/VADLAW_etri_tiny_targetpoint_attn.py` for the
experiment config and `projects/mmdet3d_plugin/VAD/VAD_head.py` (the
`target_point_mode`, `target_point_noise_std`, and `bev_residual_refine`
docstrings) for the implementation and rationale.

`data/`, `ckpts/`, and `work_dirs/` are gitignored -- this repo holds code
only. Expects the same directory layout as the other ETRI_E2E_Driving_
Challenge checkouts (dataset under `data/train`, `data/test`,
`data/etri/...`, pretrained backbone under `ckpts/`).
