"""Stage-1 VAD config using the prebuilt NVMe geometry cache.

The original stage-1 config remains the source of truth.  This variant changes
the deterministic train image geometry path, skips discarded history labels,
keeps loader workers alive between epochs, and removes the now-unnecessary DDP
unused-parameter traversal.  Validation/test intentionally keep the original
image pipeline.  Epoch-4 continuation settings are supplied by a separate
post-benchmark config; this file deliberately does not pretend to be a ready
48-epoch resume recipe.
"""

_base_ = ['./VAD_etri_tiny_stage1.py']

custom_imports = dict(
    imports=['tools.cache.etri_geometry_cache'],
    allow_failed_imports=False)

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

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
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point', 'ego_long_fut_trajs', 'ego_long_fut_masks', 'ego_long_fut_valid_flag',
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
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_gt=False, with_label=False, with_ego=False),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=4,
    train=dict(
        pipeline=cached_train_pipeline,
        history_pipeline=cached_history_pipeline),
    train_dataloader=dict(
        persistent_workers=True))

# The autocast-cache fix restores graph reachability for the 82 tensors that
# were detached.  A two-rank smoke test then found no grad=None tensors
# (zero-weight planning losses still produce exact-zero gradients), so DDP's
# per-iteration unused-graph traversal is unnecessary for this graph.
find_unused_parameters = False
