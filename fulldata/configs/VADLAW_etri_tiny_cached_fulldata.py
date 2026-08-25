"""Stage-2 VADLAW, full-data variant: trains on all 376 scenes (no
train/val split) for the final submission run. Otherwise identical to
VADLAW_etri_tiny_cached.py.

load_from points at the full-data stage1's own merged checkpoint (not the
301/75-split one) -- build that first via
VAD_etri_tiny_stage1_cached_fulldata.py + merge_stage1_world_model.py.

Do not reuse for anything that still wants a hold-out L2 number: there is
no val split left to score against once this config's ann_file is in play.
"""

_base_ = ['../../projects/configs/VAD/VADLAW_etri_tiny_cached.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

load_from = 'work_dirs/stage1_etri_fulldata/stage2_init_merged.pth'

# Base config's WandbLoggerHook is named "stage2_split_301/75" -- override
# so this run doesn't show up under that name too.
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad', name='stage2_fulldata')),
    ])

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
        ])
]

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
