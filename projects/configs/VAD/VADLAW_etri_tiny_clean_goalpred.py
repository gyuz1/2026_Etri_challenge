"""Planning toward a PREDICTED 5s goal, target point as a label. Submittable.

VADLAW_etri_tiny_clean_nodistill.py + goal_pred, nothing else, so
stage2_clean_nodistill vs stage2_clean_goalpred measures the goal alone.
Design and training losses: see the goal_pred comment in VAD_head.__init__.

GOAL
Forward-distance bins every 10m over -5..115m (12 bins) with a continuous
offset regressed inside the chosen bin, lateral regressed in units of 25m.
Lateral is not binned: in the linear proxy the TP's value was mostly forward
(x only 0.1533 vs y only 0.1964 against 0.2234), and the command already
fixes most of the lateral choice. 115m (not 110) keeps the 0.51% of train
targets beyond 110m inside a bin; the border bin still absorbs anything
further with an unclamped offset.

Why not the 5x5 grid it replaces (never trained): a perfectly known 5x5 cell
kept 0.2207 of the 0.1191 the exact TP reached in the proxy; mixing 25
trajectories by probability averages accelerate/brake hypotheses; and 25
separate output copies split the data. Here one goal is chosen per command
mode and one shared decoder follows it.
"""

_base_ = ['./VADLAW_etri_tiny_clean_nodistill.py']

model = dict(
    pts_bbox_head=dict(
        goal_pred=True,
        goal_bin_edges=(-5.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
                        75.0, 85.0, 95.0, 105.0, 115.0),
        goal_lat_scale=25.0,
        goal_long_ts=10,
        goal_cls_weight=0.5,
        goal_off_weight=0.5,
        goal_follow_weight=0.1,
    ))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                              name='stage2_clean_goalpred (predicted 5s goal, TP label)')),
    ])
