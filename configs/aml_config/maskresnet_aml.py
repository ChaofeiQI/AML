_base_ = ['./meta-rcnn_r101_c4.py',]

custom_imports = dict(
    imports=[
        'aml.aml_detector',
        'aml.aml_roi_head',
        'aml.aml_bbox_head'], 
    allow_failed_imports=False)

pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='AML',
    pretrained=pretrained,
    # backbone=dict(type='MaskResNet',depth=50),
    backbone=dict(type='MaskResNet',depth=101),
    # backbone=dict(type='ResNetWithMetaConv', depth=50),
    # backbone=dict(type='ResNetWithMetaConv', depth=101),
    roi_head=dict(
        type='AMLRoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(
            type='AMLBBoxHead', num_classes=20, num_meta_classes=20)))
