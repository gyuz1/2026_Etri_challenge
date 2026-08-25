"""Stage-2 VADLAW + EvoDriveVLA-teacher trajectory distillation, full-data
variant: warm-started from the fulldata (376-scene) stage2 checkpoint
instead of the 301/75-split one. Otherwise identical to
VADLAW_etri_tiny_cached_kd.py -- same VADHeadKD/loss_plan_kd wiring
(projects/mmdet3d_plugin/EvoKD/), same 5-epoch/lr=1e-5 short fine-tune
recipe warm-started from an already-converged (non-KD) stage2 checkpoint
rather than training from the stage1-merged init from scratch.

Launch only once BOTH of these exist:
  - work_dirs/stage2_etri_fulldata/epoch_12.pth (non-KD fulldata stage2,
    produced by fulldata/run_fulldata_pipeline.sh)
  - work_dirs/teacher_cache/etri_train_teacher_cache_fulldata.json (teacher
    cache, built via evodrive_etri_prep/generate_teacher_cache.py against
    the FULLDATA train ann-file -- vad_etri_infos_fulldata_train.pkl, not
    the 301/75-split one. The cache is keyed by scene_token::frame_idx, so
    a cache built from the 301-scene split would leave ~75 scenes' worth of
    frames with teacher_valid=False here, silently weakening the KD signal
    rather than erroring -- hence the distinct filename, to make it
    obvious the two tracks' caches are not interchangeable.)

Do not reuse for anything that wants a hold-out L2 number: there is no val
split left once the fulldata ann_file is in play (same caveat as
VADLAW_etri_tiny_cached_fulldata.py).
"""

_base_ = ['../../projects/configs/VAD/VADLAW_etri_tiny_cached_kd.py']

load_from = 'work_dirs/stage2_etri_fulldata/epoch_12.pth'
resume_from = None

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad', name='stage2_kd_fulldata')),
    ])

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

teacher_cache_path = 'work_dirs/teacher_cache/etri_train_teacher_cache_fulldata.json'

# Full copy of VADLAW_etri_tiny_cached_kd.py's cached_train_pipeline (mmcv
# list values replace on inheritance, they don't merge) with
# expected_scene_count/expected_ann_file switched to the fulldata (376-
# scene) set and teacher_cache_path pointed at the fulldata teacher cache.
# LoadTeacherWaypoints insertion and extended meta_keys are otherwise
# byte-for-byte identical to the 301/75-split KD config.
cached_train_pipeline = [
    dict(
        type='LoadETRIGeometryCache',
        cache_root='/workspace/VAD/work_dirs/etri_geometry_cache_v1',
        scale=0.4,
        strict=True,
        max_open_shards=8,
        require_complete_manifest=True,
        expected_scene_count=376,
        expected_ann_file=(
            '/workspace/VAD/data/etri/annotations/'
            'vad_etri_infos_fulldata_train.pkl'),
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
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point',
            'gt_attr_labels'
        ],
        # CustomCollect3D's default tuple, plus teacher_ego_waypoints/
        # teacher_valid. Must be spelled out in full -- mmcv config
        # inheritance replaces tuple/list values, it doesn't append.
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

# Full copy of VADLAW_etri_tiny_cached_fulldata.py's cached_history_pipeline
# -- unaffected by KD (no LoadTeacherWaypoints here; teacher supervision
# only applies to the "current" training sample, not the streamed history
# frames used to build prev_bev), just needs the same 376-scene count/
# ann-file switch as cached_train_pipeline above.
cached_history_pipeline = [
    dict(
        type='LoadETRIGeometryCache',
        cache_root='/workspace/VAD/work_dirs/etri_geometry_cache_v1',
        scale=0.4,
        strict=True,
        max_open_shards=8,
        require_complete_manifest=True,
        expected_scene_count=376,
        expected_ann_file=(
            '/workspace/VAD/data/etri/annotations/'
            'vad_etri_infos_fulldata_train.pkl'),
        expected_frame_stride=5,
        expected_crop_size=(1920, 1080),
        expected_crop_keep_top=(
            'camera_front_left', 'camera_front_right',
            'camera_rear_left', 'camera_rear_right',
            'camera_rear_wide')),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_gt=False, with_label=False, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['img', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
               'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point'])
]

data = dict(
    train=dict(
        ann_file='data/etri/annotations/'
                 'vad_etri_infos_fulldata_train.pkl',
        pipeline=cached_train_pipeline,
        history_pipeline=cached_history_pipeline))
