"""Stage-2 VADLAW, full-data (376 scenes) + 10Hz ego-motion + target_point
shortcut fix, all combined -- the final submission recipe.

Combines:
  - VADLAW_etri_tiny_cached_fulldata.py's full-data (376-scene, no held-out
    split) geometry-cache pipeline
  - fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata_10hz.py's 10Hz
    ego-motion train ann_file convention (dense pose sampling for
    ego_lcf_feat, per the organizers' 2026-08-25 Q&A)
  - projects/configs/VAD/VADLAW_etri_tiny_targetpoint_attn.py's target_point
    shortcut fix (TP attention conditioning + corruption + BEV residual
    refinement), validated on the 301/75 split before being ported here

STATUS: not yet launched -- load_from below is a placeholder path. It
requires:
  1. The 10Hz fulldata stage1 run (work_dirs/stage1_etri_fulldata_10hz,
     currently in progress on the 3090) to finish.
  2. tools/merge_stage1_world_model.py run on that stage1's final
     checkpoint, with its output explicitly named stage2_init_merged_10hz.pth
     (not the ambiguous stage2_init_merged.pth name used previously -- see
     the stage1_etri_v2 2hz/10hz naming-confusion note from 2026-08-26) to
     land at the load_from path below.
  3. The A5000 301/75-split target_point_attn experiment (in progress) to
     finish and pass its 6-condition shortcut ablation -- if that
     architecture needs to change (e.g. target_point_mode='attn' instead of
     'both'), update the model overrides below to match before launching
     this config, not after.

Do not reuse for anything that still wants a hold-out L2 number: there is
no val split left to score against once this config's ann_file is in play.
"""

_base_ = ['../../projects/configs/VAD/VADLAW_etri_tiny_cached.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

# Placeholder -- does not exist yet. See STATUS above.
load_from = 'work_dirs/stage1_etri_fulldata_10hz/stage2_init_merged_10hz.pth'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-2026-e2e-vad',
                               name='stage2_fulldata_10hz_targetpoint_attn')),
    ])

# Same shortcut-fix architecture validated on the 301/75 split in
# projects/configs/VAD/VADLAW_etri_tiny_targetpoint_attn.py. If the A5000
# ablation there ends up favoring a different target_point_mode (e.g.
# 'attn' instead of 'both') or different noise/dropout values, update here
# to match -- this file was written before that result was known.
model = dict(
    pts_bbox_head=dict(
        target_point_mode='both',
        target_point_dropout=0.1,
        target_point_noise_std=0.2,
        bev_residual_refine=True,
    ))

find_unused_parameters = True

cached_train_pipeline = [
    dict(
        type='LoadETRIGeometryCache',
        cache_root='/workspace/VAD/work_dirs/etri_geometry_cache_v1',
        scale=0.4,
        strict=True,
        max_open_shards=8,
        require_complete_manifest=True,
        expected_scene_count=376,
        # Deliberately still the OLD (non-10hz) pkl path, matching what the
        # existing cache_manifest.json recorded -- the cache is purely
        # image geometry, independent of ego-motion/annotation pkl. Same
        # reasoning as fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata_10hz.py.
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
         with_gt=False, with_label=False, with_ego=True),
    dict(type='CustomCollect3D',
         keys=['img', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
               'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point'])
]

data = dict(
    train=dict(
        # 10Hz ego-motion train pkl, matching stage1's own ann_file so
        # ego_lcf_feat/can_bus stay consistent between the two stages.
        ann_file='data/etri/annotations_10hz/'
                 'vad_etri_10hz_infos_temporal_train.pkl',
        pipeline=cached_train_pipeline,
        history_pipeline=cached_history_pipeline))
