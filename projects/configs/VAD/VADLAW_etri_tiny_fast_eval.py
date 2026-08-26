"""Inference-speed test_pipeline for T_infer measurement / real submission.

Replaces LoadMultiViewImageFromFiles -> UndistortMultiViewImage ->
CropMultiViewImage -> (later) RandomScaleImageMultiViewImage with
FastLoadMultiViewImageFromFiles -> FastUndistortCropScaleMultiViewImage
(projects/mmdet3d_plugin/datasets/pipelines/fast_geometry.py) -- same
geometric result (verified: same crop/scale math the replaced transforms
apply, just done directly at the final resolution and in parallel across
cameras instead of sequentially at full 1920x1080), measured ~7-10x
faster. See fast_geometry.py's docstring for why this doesn't overlap
with the training-time geometry cache (that cache still undistorts at
full resolution -- it's fast only because it's computed once and reused
across many epochs, which doesn't help a single inference call).

Only test_pipeline changes; everything else (model, train config) is
inherited untouched from VADLAW_etri_tiny.py.
"""
_base_ = ['./VADLAW_etri_tiny.py']

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = ['Car', 'Pedestrian', 'Cyclist']

test_pipeline = [
    # to_float32=False: stays uint8 through undistort/crop/scale (1/4 the
    # memory traffic vs float32) -- NormalizeMultiviewImage casts to
    # float32 internally regardless of input dtype, so nothing downstream
    # needs to change.
    dict(type='FastLoadMultiViewImageFromFiles', to_float32=False,
         reduced_decode=2),
    dict(type='FastUndistortCropScaleMultiViewImage', scale=0.4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                 with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                       'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
                       'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point', 'gt_attr_labels'])])
]

data = dict(
    test=dict(pipeline=test_pipeline))
