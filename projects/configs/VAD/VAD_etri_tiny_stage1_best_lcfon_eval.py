"""Eval config for the finished stage-1 TEACHER donor. Diagnostic only.

ego_lcf_feat_idx is [0..7] here, so this is not submittable -- it exists to
check whether a finding reproduces on the second lineage before acting on it.
"""

_base_ = ['./VAD_etri_tiny_stage1_best_lcfon.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

test_pipeline = [
    dict(type='FastLoadMultiViewImageFromFiles', to_float32=False,
         reduced_decode=2),
    dict(type='FastUndistortCropScaleMultiViewImage', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D',
                 class_names=class_names, with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',
                       'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                       'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                       'ego_target_point', 'ego_long_fut_trajs',
                       'ego_long_fut_masks', 'ego_long_fut_valid_flag',
                       'gt_attr_labels'])
        ])
]

data = dict(test=dict(pipeline=test_pipeline))
