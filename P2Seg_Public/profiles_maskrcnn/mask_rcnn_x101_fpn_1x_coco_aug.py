_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

# dataset settings
dataset_type = 'CocoFmtDataset'
data_root = '/home/ubuntu/dataset/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
#     pretrained='open-mmlab://resnext101_32x4d',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.3],
#         contrast_limit=[0.1, 0.3],
#         p=0.2),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(
#                 type='RGBShift',
#                 r_shift_limit=10,
#                 g_shift_limit=10,
#                 b_shift_limit=10,
#                 p=1.0),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=20,
#                 sat_shift_limit=30,
#                 val_shift_limit=20,
#                 p=1.0)
#         ],
#         p=0.1),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]



train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(
#         type='Expand',
#         mean=img_norm_cfg['mean'],
#         to_rgb=img_norm_cfg['to_rgb'],
#         ratio_range=(1, 2)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#         min_crop_size=0.3),    
    dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_masks': 'masks',
#             'gt_bboxes': 'bboxes'
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True),    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
        img_scale=[(1333, 800), (1333, 704)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])
# evaluation = dict(
#     interval=1, metric=['bbox', 'segm'],
#     iou_thrs=[0.25, 0.5, 0.75],  # set None mean use 0.5:1.0::0.05
# #     proposal_nums=[1000],
#     do_final_eval=True,
#     cocofmt_kwargs=dict(
#         ignore_uncertain=True,
#         use_ignore_attr=True,
#         use_iod_for_ignore=True,
#         iod_th_of_iou_f="lambda iou: iou",  #"lambda iou: (2*iou)/(1+iou)",
#         cocofmt_param=dict(
#             evaluate_standard='coco',  # or 'coco'
# #             iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
# #             maxDets=[1, 10, 100],         # set this same as set evaluation.proposal_nums
#         )
#     )
# )
evaluation = dict(interval=1, metric=['bbox', 'segm'])

# evaluation = dict(
#     interval=1, metric=['bbox', 'segm'],
#     iou_thrs=[0.25, 0.5, 0.75],  # set None mean use 0.5:1.0::0.05
#     proposal_nums=[200],
#     do_final_eval=True,
#     cocofmt_kwargs=dict(
#         ignore_uncertain=True,
#         use_ignore_attr=True,
#         use_iod_for_ignore=True,
#         iod_th_of_iou_f="lambda iou: iou",  #"lambda iou: (2*iou)/(1+iou)",
#         cocofmt_param=dict(
#             evaluate_standard='tiny',  # or 'coco'
#             # iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
#             # maxDets=[200],              # set this same as set evaluation.proposal_nums
#         )
#     )
# )