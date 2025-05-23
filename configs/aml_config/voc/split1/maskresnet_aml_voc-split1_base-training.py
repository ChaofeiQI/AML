_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../maskresnet_aml.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT1'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT1')),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'),
    model_init=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=100, step=[12000, 16000])
evaluation = dict(interval=3000)
checkpoint_config = dict(interval=3000)
runner = dict(max_iters=18000)
optimizer = dict(lr=0.01)
# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15)))