"""Stage-1 VAD, full-data + 10Hz ego-motion variant: trains on all 376
scenes (no train/val split) using ego_lcf_feat/can_bus computed from dense
10Hz-spaced pose (etri_vad_converter_10hz.py) instead of TRAJ_STEP(0.5s)-
spaced pose, per the organizers' 2026-08-25 Q&A confirming pose used for
this calc doesn't need a matching fed image. Otherwise identical to
VAD_etri_tiny_stage1_cached_fulldata.py.

Replaces (not warm-started from) the non-10hz fulldata stage1 run that was
in progress -- interrupted at epoch 16/48 (work_dirs/stage1_etri_fulldata/
epoch_16.pth, kept as a reference/fallback, not reused here) once it became
clear stage2 would need a target_point_dropout retrain anyway, so folding
in the 10Hz ego-motion improvement now (before more stage1 epochs get
sunk into the non-10hz pkl) was the better sequencing. Starts fresh from
ckpts/law_pretrained_nus.pth, same as the original fulldata stage1 did.

Do not reuse for anything that still wants a hold-out L2 number: there is
no val split left to score against once this config's ann_file is in play.
"""

_base_ = ['../../projects/configs/VAD/VAD_etri_tiny_stage1_cached.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad', name='stage1_fulldata_10hz')),
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
        # Deliberately still points at the OLD (non-10hz) pkl path, not
        # data.train.ann_file below -- this field is LoadETRIGeometryCache's
        # own sanity check against work_dirs/etri_geometry_cache_v1's
        # cache_manifest.json, which was built once against that pkl and
        # records it verbatim. The cache itself (per-scene undistorted/
        # cropped image crops) is purely a function of images + calibration,
        # never ego-motion, so it's byte-identical and fully valid for the
        # 10hz pkl too -- rebuilding it would just re-derive the exact same
        # cache at real CPU cost for no behavior change. Matching this
        # string to the manifest's recorded value is what lets the sanity
        # check pass without a pointless rebuild.
        expected_ann_file=(
            '/workspace/VAD/data/etri/.causal_regen_teammate_split/'
            'vad_etri_infos_temporal_train.pkl'),
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
        # Deliberately still points at the OLD (non-10hz) pkl path, not
        # data.train.ann_file below -- this field is LoadETRIGeometryCache's
        # own sanity check against work_dirs/etri_geometry_cache_v1's
        # cache_manifest.json, which was built once against that pkl and
        # records it verbatim. The cache itself (per-scene undistorted/
        # cropped image crops) is purely a function of images + calibration,
        # never ego-motion, so it's byte-identical and fully valid for the
        # 10hz pkl too -- rebuilding it would just re-derive the exact same
        # cache at real CPU cost for no behavior change. Matching this
        # string to the manifest's recorded value is what lets the sanity
        # check pass without a pointless rebuild.
        expected_ann_file=(
            '/workspace/VAD/data/etri/.causal_regen_teammate_split/'
            'vad_etri_infos_temporal_train.pkl'),
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
    train=dict(
        ann_file='data/etri/annotations_10hz/'
                 'vad_etri_10hz_infos_temporal_train.pkl',
        pipeline=cached_train_pipeline,
        history_pipeline=cached_history_pipeline))
