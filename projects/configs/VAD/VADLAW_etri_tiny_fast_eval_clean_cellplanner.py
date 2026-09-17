"""Eval config for VADLAW_etri_tiny_clean_cellplanner.py.

Cell layouts, positional code and head widths define the model and must match
training (the checkpoint's stored layout is checked at load). Scoring needs
--select-cell-by-tp: the model emits every cell's trajectory and the caller
selects with cell_planner_utils.route_trajectory.
"""

_base_ = ['./VADLAW_etri_tiny_fast_eval_clean.py']

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
