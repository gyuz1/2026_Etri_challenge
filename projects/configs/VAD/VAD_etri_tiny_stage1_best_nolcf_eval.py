"""Eval config for the finished stage-1 student donor. Compliant.

Stage 1 now trains the planner too (loss_plan_reg=1.0 replaced the Qwen KD
that turned out to be memorization), so stage1_best_nolcf is a working,
ego_lcf-free planner in its own right -- not just a donor. Measuring it
answers two things stage 2 cannot answer for another day:

  1. what the stage-1 rebuild bought on its own, against A student v1's
     0.4218, with distillation and the stage-2 recipe held out entirely
  2. whether the 3-frame descriptor works end to end on the inference path,
     where prev_bev2 comes from VAD.forward_test's stream rather than from
     the training queue

The test pipeline is copied from the verified stage-2 student eval config
rather than from the old stage1 eval, which is stale in two ways that would
both fail silently: it enables ego_lcf (so it would not be compliant) and it
predates the 3-frame descriptor.
"""

_base_ = ['./VAD_etri_tiny_stage1_best_nolcf.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']

test_pipeline = [
    dict(type='FastLoadMultiViewImageFromFiles', to_float32=False,
         reduced_decode=2),
    dict(type='FastUndistortCropScaleMultiViewImage', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
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
