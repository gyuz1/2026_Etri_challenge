"""Command cell planner, target point for selection only. Submittable.

VADLAW_etri_tiny_clean_nodistill.py with the planner output replaced:
  - moving commands: one head per command shared by its cells, input =
    ego_feats (520) + fixed cell positional code (256), output [B, Nf, Nl, 6, 2]
  - U_TURN, STOP: plain heads, output [B, 6, 2]
  - selection (not learned, same rule in training/eval/submission):
    |target point| < 1 m or STOP label -> STOP; U_TURN -> its head;
    otherwise the cell of the command containing the target point
  - the waypoint loss supervises the selected trajectory only
The target point never enters a feature. Cell edges: within-cell 3 s
trajectory spread minimised on train (>= 50 frames, >= 3 scenes, >= 4 m wide
per cell), 2026-09-17; see SHARED_CONTEXT.md.
"""

_base_ = ['./VADLAW_etri_tiny_clean_nodistill.py']

model = dict(
    pts_bbox_head=dict(
        cell_planner=True,
        cell_layouts=[
            # LANE_KEEP 15x1
            ([1, 12, 22, 31, 38, 45, 51, 56, 61, 66, 72, 82, 94, 101, 106, 117],
             [-20, 17]),
            # LANE_CHANGE_L 11x1
            ([1, 20, 25, 36, 44, 51, 58, 66, 78, 94, 103, 116], [-16, 12]),
            # LANE_CHANGE_R 14x1
            ([1, 23, 30, 35, 41, 45, 49, 53, 59, 64, 74, 81, 89, 100, 115],
             [-12, 18]),
            # TURN_LEFT 4x2
            ([1, 16, 21, 30, 48], [-2, 10, 27]),
            # TURN_RIGHT 6x2
            ([1, 12, 16, 20, 26, 33, 43], [-26, -11.5, 2]),
            None,  # U_TURN
            None,  # STOP
        ],
        cell_pe_dim=256,
        cell_pe_wavelengths=(4.0, 400.0),
        cell_stop_tp_thresh=1.0,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_cellplanner_v1 (command cells, TP selection)')),
    ])
