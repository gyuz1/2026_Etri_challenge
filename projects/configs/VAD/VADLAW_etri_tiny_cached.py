"""Stage-2 VADLAW config using the prebuilt geometry cache.

Mirrors VAD_etri_tiny_stage1_cached.py: swaps the deterministic train image
geometry (undistort/crop/resize) for the verified offline cache, skips
discarded history-frame annotation loading, and keeps loader workers alive
between epochs. Validation/test intentionally keep the original image
pipeline (val scenes have their own, separate, not-yet-rebuilt cache).

find_unused_parameters is left True here (unlike the stage-1 cached config,
which specifically verified False is safe via a grad-update audit) because
this is VADLAW's first cached/batched run and that verification hasn't been
repeated for the world-model branch's autograd graph.
"""

_base_ = ['./VADLAW_etri_tiny.py']

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
            '/workspace/VAD/data/etri/annotations/'
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
        expected_scene_count=301,
        expected_ann_file=(
            '/workspace/VAD/data/etri/annotations/'
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
    # Unlike base VAD's union2one (which only reads ego_* off queue[-1]),
    # LAWVADCustomETRIDataset.union2one reads ego_fut_trajs/ego_fut_masks/
    # ego_fut_cmd/ego_his_trajs/ego_lcf_feat off *every* queued frame (the
    # world model applies its objective to history frames too). Detection/
    # map GT is still only needed on the current frame, so with_gt/with_label
    # stay off, but with_ego must stay on and those keys must be collected.
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_gt=False, with_label=False, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['img', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
               'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point'])
]

data = dict(
    # VADLAW's obtain_history_prediction() reruns the full pts_bbox_head
    # (incl. map_decoder) on every history frame for the world-model
    # objective, so its memory footprint per sample is much larger than
    # plain VAD stage1's. batch=4 (safe for stage1) OOM'd here at iter 0.
    samples_per_gpu=1,
    train=dict(
        pipeline=cached_train_pipeline,
        history_pipeline=cached_history_pipeline),
    train_dataloader=dict(
        persistent_workers=True))

fp16 = dict(loss_scale=512.)

find_unused_parameters = True
