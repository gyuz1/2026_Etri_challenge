"""Stage-2 VADLAW + EvoDriveVLA-teacher trajectory distillation.

Adds loss_plan_kd via VADHeadKD (projects/mmdet3d_plugin/EvoKD/vad_head_kd.py)
-- a VADHead subclass, zero edits to VAD_head.py/VAD_LAW.py. The teacher
trajectory reaches the loss purely through img_metas: LoadTeacherWaypoints
(new pipeline step) sets results['teacher_ego_waypoints']/['teacher_valid'],
and CustomCollect3D's meta_keys (extended below) carries them into the same
img_metas dict VADLAW.forward_train() already passes to
pts_bbox_head.loss(..., img_metas=current_metas) for ego_fut_cmd etc.

The teacher (EvoDriveVLA/Qwen2.5-VL) never enters this graph -- only its
cached, offline-generated trajectory does -- so this changes nothing about
inference FLOPs/latency versus VADLAW_etri_tiny_cached.py.

Requires the teacher cache to exist first: see
evodrive_etri_prep/generate_teacher_cache.py. Until it exists, every sample
falls back to teacher_valid=False and loss_plan_kd stays a real zero.

Warm-started from our already-trained (non-KD) stage2 checkpoint rather than
training from the stage1-merged init from scratch -- stage2 has already
converged on the real GT objective over 12 epochs, so this is a short
additional fine-tune phase that only needs to teach the model the new
loss_plan_kd term, not re-learn detection/map/planning from zero. Peak LR
cut 5x versus the from-scratch recipe (5e-5 -> 1e-5) so the new loss term
doesn't destabilize the already-converged weights; epoch count cut 12 -> 5
since convergence, not exploration, is the point here. Launch with a NEW
--work-dir (not stage2_etri_teammate_split/, which the base non-KD run
still owns) once work_dirs/stage2_etri_teammate_split/epoch_12.pth and the
teacher cache both exist.
"""

_base_ = ['./VADLAW_etri_tiny_cached.py']

load_from = 'work_dirs/stage2_etri_teammate_split/epoch_12.pth'
resume_from = None

total_epochs = 5
optimizer = dict(lr=1e-5)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
evaluation = dict(interval=total_epochs + 1, metric='bbox', map_metric='chamfer')

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
        # Deliberately small relative to loss_plan_reg (weight 1.0) for the
        # first ablation -- GT still dominates, teacher only nudges.
        kd_weight=0.2,
    ))

# Full copy of VADLAW_etri_tiny_cached.py's cached_train_pipeline (mmcv list
# values replace on inheritance, they don't merge) with one insertion
# (LoadTeacherWaypoints) and meta_keys extended by two fields. Everything
# else -- LoadETRIGeometryCache, augmentation order, collected tensor keys --
# is byte-for-byte identical to the non-KD cached config.
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
            '/workspace/VAD/data/etri/.causal_regen_teammate_split/'
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

data = dict(
    train=dict(pipeline=cached_train_pipeline))
