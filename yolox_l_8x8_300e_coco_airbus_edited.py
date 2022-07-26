import os

_base_ = ['/content/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py']

img_scale = (640, 640)  # height, width

# dataset settings
data_root = "../SSP22_AI_Workshop/mini_test"

dataset_type = 'CocoDataset'
classes = ('Airplane',)

img_norm_cfg = dict(
    mean=[135.62, 133.61, 118.17], std=[52.2, 47.8, 46.4], to_rgb=True)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',   # Mosaic augmentation
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_root, 'train/sliced_bbox_coco.json'),
        img_prefix=os.path.join(data_root, 'train/sliced_images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomCrop', crop_size=img_scale)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[   # Augmentations
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad',
                 pad_to_square=True,
                 pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_root, 'sliced_bbox_coco.json'),
        img_prefix=os.path.join(data_root, 'sliced_images/'),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=os.path.join(data_root, 'sliced_bbox_coco.json'),
        img_prefix=os.path.join(data_root, 'sliced_images/'),
        pipeline=test_pipeline))

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MlflowLoggerHook', exp_name="airbus_aircraft_yolox"),
        dict(type='TensorboardLoggerHook')
    ]
)

model = dict(
    input_size=img_scale,
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256, num_classes=1))

# optimizer
# default 8 gpu
max_epochs = 200
num_last_epochs = 15
interval = 10
workflow = [('train', 1), ('val', 1)]


