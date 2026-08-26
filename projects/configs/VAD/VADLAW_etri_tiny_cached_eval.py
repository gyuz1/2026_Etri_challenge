"""Cached-geometry test pipeline for eval_holdout_l2.py against stage-2
(VADLAW) checkpoints.

Same rationale as VAD_etri_tiny_stage1_cached_eval.py, based on the VADLAW
model config instead of plain VAD so build_model/load_checkpoint construct
the right architecture (bev_world_model included) for a stage-2 checkpoint.
"""

_base_ = ['./VADLAW_etri_tiny_cached.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

cached_test_pipeline = [
    dict(
        type='LoadETRIGeometryCache',
        cache_root='/workspace/VAD/work_dirs/etri_geometry_cache_val_v1',
        scale=0.4,
        strict=True,
        max_open_shards=8,
        require_complete_manifest=True,
        expected_scene_count=75,
        expected_ann_file=(
            '/workspace/VAD/data/etri/.causal_regen_teammate_split/'
            'vad_etri_infos_temporal_val_split.pkl'),
        expected_frame_stride=5,
        expected_crop_size=(1920, 1080),
        expected_crop_keep_top=(
            'camera_front_left', 'camera_front_right',
            'camera_rear_left', 'camera_rear_right',
            'camera_rear_wide')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # No RandomScaleImageMultiViewImage here -- the cache already
            # stores images at scale=0.4.
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                 with_label=False, with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                    'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point', 'gt_attr_labels'
                ])
        ])
]

data = dict(
    test=dict(pipeline=cached_test_pipeline),
    val=dict(pipeline=cached_test_pipeline))
