_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../maskresnet_aml.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT2_5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'),
    model_init=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=2400, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=2400)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=2400)
load_from = 'work_dirs/maskresnet_aml_voc-split2_base-training/latest.pth'

# model settings
model = dict(frozen_parameters=['backbone', 'shared_head'])