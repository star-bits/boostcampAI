from mmdet.datasets.pipelines import Albu, MixUp, Mosaic
from datetime import datetime


img_width = 1024 # 512, 1024
img_height = img_width
img_scale = (img_width, img_height)


max_epochs = 15


# version_number = 6
work_dir = './work_dirs/_swin_faster'
gpu_ids = [0]
seed = 2022


# mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py


##############################################################################################################
# model settings
# models/faster_rcnn_r50_fpn.py

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
pretrained = "./swin_large_patch4_window7_224_22kto1k.pth" # swin-L

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='SwinTransformer',
        # embed_dims=96,
        embed_dims=192, # swin-L
        # depths=[2, 2, 6, 2],
        depths=[2, 2, 18, 2], # swin-L
        # num_heads=[3, 6, 12, 24],
        num_heads=[6, 12, 24, 48], # swin-L
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        # in_channels=[96, 192, 384, 768], 
        in_channels=[192, 384, 768, 1536], # swin-L
        out_channels=256,
        num_outs=5),    
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            # loss_bbox=dict(type='CIoULoss', loss_weight=2.0),
        )),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0., # 0.05 by default
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


##############################################################################################################
# dataset settings
# datasets/coco_detection.py

dataset_type = 'CocoDataset'
data_root = '../../dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(
    #     type="Resize",
    #     img_scale=[(256, 256), (384, 384), (512, 512), (768, 768), (1024, 1024), (1280, 1280)],
    #     multiscale_mode="value",
    #     keep_ratio=True
    # ),
    # dict(
    #     type='Resize', 
    #     img_scale=[(512, 341), (512, 768)], 
    #     multiscale_mode='range', 
    #     keep_ratio=True
    # ),
    
    # dict(type="MixUp", img_scale=img_scale),
    # dict(type="Mosaic", img_scale=img_scale),

    dict(type='RandomFlip', flip_ratio=0.5),
    
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(type='RandomBrightnessContrast', contrast_limit=0.0, p=0.5),
    #         dict(type='Blur', p=0.5),
    #         dict(type='GaussNoise', p=0.5),
    #         dict(type='Cutout', num_holes=16, max_h_size=img_height//10, max_w_size=img_width//10, fill_value=64, p=0.5),
    #     ],
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(
            #     type="Resize",
            #     img_scale=[(256, 256), (384, 384), (512, 512), (768, 768), (1024, 1024), (1280, 1280)],
            #     multiscale_mode="value",
            #     keep_ratio=True
            # ),
            # dict(
            #     type='Resize', 
            #     img_scale=[(512, 341), (512, 768)], 
            #     multiscale_mode='range', 
            #     keep_ratio=True
            # ),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2, # batch_size: 8 for 512, 2 for 1024
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json', # 'train.json', 'train_split.json'
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valid_split.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(
    save_best='auto', # save the best epoch model in terms of bbox mAP
    interval=1, # run validation by `interval` epochs
    metric='bbox')


##############################################################################################################
# optimizer
# schedules/schedule_1x.py

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11] # for max 12 epochs
    # step=[16, 22] # for max 24 epochs
)

loss_scale = dict(
    init_scale=65536.0,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)
fp16 = dict(
    loss_scale=loss_scale,
)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs, meta=dict(fp16=fp16))


##############################################################################################################
# default_runtime.py

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
# yapf:disable
current_time = datetime.now()
current_time = current_time.strftime('%m-%d-%H-%M')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='mmdetection', name='_swin_faster_v' + str(version_number) + '_' + current_time))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None # put .pth file location here to resume from a certain epoch
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
