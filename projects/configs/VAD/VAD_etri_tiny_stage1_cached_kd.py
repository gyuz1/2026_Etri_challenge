"""Stage-1 VAD + EvoDriveVLA-teacher trajectory distillation (prepared, not
run in the first ablation).

Same VADHeadKD mechanism as VADLAW_etri_tiny_cached_kd.py -- see that file's
docstring for the full explanation. Kept here so the "v3: KD from epoch 1"
pipeline is ready to launch later without writing new code, once the
stage-2-only ablation (VADLAW_etri_tiny_cached_kd.py) has shown loss_plan_kd
is actually worth the ~40h cost of redoing stage 1.

Note: stage 1's loss_plan_reg stays at weight 0.0 (pure perception target),
so loss_plan_kd here really would be the *only* signal training the ego
planning decoder during stage 1 -- unlike stage 2, where it supplements an
already-nonzero loss_plan_reg.
"""

_base_ = ['./VAD_etri_tiny_stage1_cached.py']

custom_imports = dict(
    imports=[
        'tools.cache.etri_geometry_cache',
        'projects.mmdet3d_plugin.EvoKD.vad_head_kd',
        'projects.mmdet3d_plugin.EvoKD.load_teacher_traj',
    ],
    allow_failed_imports=False)

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

teacher_cache_path = 'work_dirs/teacher_cache/etri_train_teacher_cache.json'

model = dict(
    pts_bbox_head=dict(
        type='VADHeadKD',
        kd_weight=0.2,
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
    dict(type='LoadTeacherWaypoints', cache_path=teacher_cache_path),
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
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point', 'ego_long_fut_trajs', 'ego_long_fut_masks', 'ego_long_fut_valid_flag',
            'gt_attr_labels'
        ],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img',
            'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'prev_idx', 'next_idx', 'pcd_scale_factor', 'pcd_rotation',
            'pts_filename', 'transformation_3d_flow', 'scene_token',
            'can_bus', 'teacher_ego_waypoints', 'teacher_valid',
        ))
]

data = dict(
    train=dict(pipeline=cached_train_pipeline))
