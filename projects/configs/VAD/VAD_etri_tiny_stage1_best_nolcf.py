"""Stage 1, best known configuration, ego_lcf OFF -- the STUDENT's donor.

Stage 1 is where the BEV encoder is actually trained: 48 epochs against
stage 2's 12. Every motion-representation improvement therefore belongs
here first, and until now none of them were -- they had all been applied
only to stage 2, i.e. to the 12-epoch quarter of the problem. This config
puts them where they have four times the effect.

WHAT CHANGES VS VAD_etri_tiny_stage1_cached_kd_nolcf.py

  aux_bev_motion_frames  2 -> 3
      Two BEV frames determine a velocity and nothing more, which is why
      ax/ay were never in the regression targets. The third frame makes the
      second difference available, and that IS the acceleration
      (verified in closed form: d1 - d2 == a*dt^2 under constant accel).
      Kinematic oracles put this at the larger half of everything ego state
      can buy: 0.5965 with perfect velocity, 0.2708 with velocity AND
      acceleration (tools/kinematic_oracle_ceiling.py).

  aux_bev_motion_grid    4 -> 8
      At grid 4 a descriptor cell covers 15.0m x 7.5m while the ego moves
      ~5m between frames at 10 m/s -- the displacement averages away inside
      one cell before the difference is taken. A descriptor built to measure
      motion was coarse enough to hide it. Grid 8 puts a cell at 7.5m x 3.8m.

  aux_bev_motion_idx     +ax, ay      now that they are observable
  aux_bev_motion_norm    added
      Raw L1 over the old targets ran vx 49.3%, speed 49.4%, vy 1.2%,
      yaw_rate 0.1% -- two near-duplicate signals (speed = norm(vx,vy) with
      vy tiny) taking 98.7% while the component that decides turning was
      effectively unsupervised. Dividing by each component's train-split std
      rebalances to roughly 20/20/20/17/13/10.

  aux_bev_future_motion  added
      Regresses the next 3s of ego speed from the same descriptor. The
      oracles above cap what knowing the PRESENT can do at 0.2708, and the
      0.14 leaderboard entry is 48% below that -- the rest has to come from
      reading the scene for what happens next. Over 4000 val samples the
      speed 3s out differs from the current speed by 0.80 m/s on average,
      which is exactly what kinematic extrapolation throws away.

  kd_weight 0.2 -> 0, loss_plan_reg 0.0 -> 1.0
      The Qwen KD teacher was measured at 0.1798m on the 301 scenes it was
      fine-tuned over and 0.3511m on the 75 held out -- 95% worse, 5.9
      sigma. It was also the ONLY signal training ego_fut_decoder across
      stage 1, so the planner was being fitted to answers a teacher had
      memorized. The exact GT sits unused in the same annotation file.
      loss_plan_reg=1.0 matches stage 2, so the planner sees one consistent
      objective across both stages.

      loss_plan_reg=0.0 was inherited as "matching the original VAD stage1
      recipe", whose rationale is curriculum -- do not train a planner on a
      BEV that cannot see yet. Adding KD already broke that premise. The
      honest counter-argument, untested: 48 epochs of GT planning loss could
      overfit the planner or crowd out perception.

LoadTeacherWaypoints is dropped from the pipeline. kd_weight=0 switches off
the loss, not the data loading -- that transform opens the teacher cache at
dataset construction and crashes on any machine without the JSON. mmcv
REPLACES list-typed config fields rather than merging, so the whole pipeline
has to be restated to remove one entry.
"""

_base_ = ['./VAD_etri_tiny_stage1_cached_kd_nolcf.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

model = dict(
    pts_bbox_head=dict(
        type='VADHead',
        kd_weight=0.0,
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        # MUST be True, and is not inherited. Without it the descriptor
        # falls back to a single frame's global mean (aux_bev_in_dim =
        # embed_dims = 256) and BOTH aux_bev_motion_frames and
        # aux_bev_motion_grid become silent no-ops -- the head builds, the
        # config looks right, and nothing measures motion at all. A global
        # mean over BEV is shift-invariant, which is the whole reason the
        # temporal regional descriptor exists.
        aux_bev_motion_temporal=True,
        aux_bev_motion_frames=3,
        aux_bev_motion_grid=8,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_norm=(5.7040, 0.1715, 0.4625, 0.3579, 0.0547, 5.7050),
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
        aux_bev_future_motion_weight=0.5,
    ))

cached_train_pipeline = [
    dict(
        type='LoadETRIGeometryCache',
        cache_root='/workspace/VAD/work_dirs/etri_geometry_cache_v1',
        scale=0.4,
        strict=True,
        max_open_shards=8,
        require_complete_manifest=True,
        expected_scene_count=301,
        expected_ann_file=(
            '/workspace/VAD/data/etri/.causal_regen_split_301_75/'
            'vad_etri_infos_temporal_train_split.pkl'),
        expected_frame_stride=5,
        expected_crop_size=(1920, 1080),
        expected_crop_keep_top=(
            'camera_front_left', 'camera_front_right',
            'camera_rear_left', 'camera_rear_right',
            'camera_rear_wide')),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_ego=True),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
            'ego_target_point', 'ego_long_fut_trajs', 'ego_long_fut_masks',
            'ego_long_fut_valid_flag', 'gt_attr_labels'
        ],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'prev_idx', 'next_idx', 'pcd_scale_factor', 'pcd_rotation',
            'pts_filename', 'transformation_3d_flow', 'scene_token',
            'can_bus',
        ))
]

# Equal frame gaps. The default 'random' drops one of the three history
# candidates at random, so the two gaps feeding aux_bev_motion_frames=3's
# second difference are unequal in 67% of samples -- and (cur - prev1) -
# (prev1 - prev2) is acceleration only when they are equal. The term a
# mismatch injects is v*dt = 5.28m against the real a*dt^2 = 0.12m, 46x
# larger, with a sign that flips per sample. Evaluation always streams equal
# gaps (--frame-offsets 0,-5,-10), so this also removes a train/eval
# mismatch. See etri_vad_dataset.py's prepare_train_data.
data = dict(train=dict(pipeline=cached_train_pipeline,
                       history_sampling='fixed'))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage1_BEST_nolcf (GT planning, 3-frame, grid8, '
                      'accel targets, normalized, future speed)'))),
    ])
