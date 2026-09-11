"""Stage-1 with the GT trajectory loss ON and Qwen KD OFF, ego_lcf off.

The A/B against VAD_etri_tiny_stage1_cached_kd_nolcf.py: same everything,
except which signal trains ego_fut_decoder during stage 1's 48 epochs.

  kd_nolcf (existing) : kd_weight 0.2,  loss_plan_reg 0.0   -> Qwen target
  gt_nolcf (this)     : kd_weight 0.0,  loss_plan_reg 1.0   -> GT target

WHY THIS IS WORTH RUNNING

loss_plan_reg=0.0 in stage 1 was inherited as "matching the original VAD
stage1 recipe" (VAD_etri_tiny_stage1.py:304), whose rationale is curriculum
-- do not train a planner on a BEV that cannot see yet. But adding KD
already broke that premise: the kd_nolcf config states outright that
loss_plan_kd is the only thing training ego_fut_decoder across stage 1. So
the planner is being trained there either way, and the real question is
what it is trained against.

Right now that is a Qwen prediction measured at 0.3511m on held-out scenes
(0.1798m on the scenes it was fine-tuned on -- the gap is memorization,
5.9 sigma), while the exact GT sits unused in the same annotation file at
zero error. None of the usual reasons to prefer a teacher over GT apply
here: the target is a single hard trajectory with no soft distribution, the
teacher does not generalize better, and GT exists for every one of these
frames.

The honest counter-argument, also unmeasured: 48 epochs of GT planning loss
may overfit the planner or crowd out perception learning -- exactly what the
original recipe avoids -- and an inaccurate KD target may have been acting
as weak regularization. That is what this run settles.

WEIGHTS
loss_plan_reg=1.0 matches stage 2 (VADLAW_etri_tiny_kd_nolcf_split_distill
uses 1.0), so the planner sees a consistent objective across both stages
rather than a new scale at the handover. col/dir/bound already match stage
2 in the kd_nolcf base and are left alone, so this file changes exactly two
numbers.

Head type drops back to plain VADHead: with kd_weight at 0 the KD branch
contributes nothing, and VADHeadKD would still require the teacher cache to
be loaded through the pipeline for no reason.

COMPARE AGAINST
Whatever stage 2 is run on top of this, against the same stage 2 on the
kd_nolcf lineage. The reference points on the val split are the compliant
baseline 0.4885m and the Scheme-A student's 0.4218m, both of which came
from the KD lineage.
"""

_base_ = ['./VAD_etri_tiny_stage1_cached_kd_nolcf.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

model = dict(
    pts_bbox_head=dict(
        type='VADHead',
        kd_weight=0.0,
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
    ))

# The whole list has to be restated because mmcv REPLACES list-typed config
# fields rather than merging them -- there is no way to drop one entry of an
# inherited pipeline. This is the KD pipeline with LoadTeacherWaypoints
# removed, and the two teacher_* meta_keys it populated removed with it.
#
# Not optional: LoadTeacherWaypoints opens the teacher cache at dataset
# CONSTRUCTION time, so leaving it in crashes before training starts on any
# machine without that JSON -- which is exactly what happened on the first
# launch here. kd_weight=0 switches off the loss, not the data loading.
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

data = dict(train=dict(pipeline=cached_train_pipeline))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad',
                name=('stage1_nolcf GT-only (loss_plan_reg=1.0, Qwen KD off, '
                      'aux_bev_motion, col_dir_bound, cascade_refine_x3, '
                      'ema)'))),
    ])
