_base_ = [
    '../../_base_/datasets/nway_kshot/few_shot_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../maskresnet_aml.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
data = dict(
    train=dict(
        save_dataset=True,
        num_used_support_shots=10,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='10SHOT')],
            num_novel_shots=10,
            num_base_shots=10,
        )),
    model_init=dict(num_novel_shots=10, num_base_shots=10))
evaluation = dict(interval=10000)
checkpoint_config = dict(interval=10000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, step=[10000])
runner = dict(max_iters=10000)
load_from = 'work_dirs/maskresnet_aml_coco_base-training/iter_100000.pth'

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=80, num_meta_classes=80)),
    with_refine=True,
    frozen_parameters=[
    'backbone', 'shared_head',  'aggregation_layer', 'rpn_head.rpn_conv',
])
