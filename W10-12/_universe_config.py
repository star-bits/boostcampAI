from mmdet.datasets.pipelines import Albu, MixUp, Mosaic


img_width = 1024 # 512, 1024
img_height = img_width
img_scale = (img_width, img_height)


max_epochs = 30


work_dir = './work_dirs/_universe'
gpu_ids = [0]
seed = 2022


##############################################################################################################
# model settings
# models/universenet101_2008d.py

pretrained = "./swin_large_patch4_window7_224_22kto1k.pth" # swin-L

model = dict(
    type='GFL',
    # backbone=dict(
    #     type='Res2Net',
    #     depth=101,
    #     scales=4,
    #     base_width=26,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     # norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     norm_cfg=dict(type='BN', requires_grad=True),  
    #     norm_eval=False,
    #     style='pytorch',
    #     dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
    #     stage_with_dcn=(False, True, True, True),
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
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
    neck=[
        # dict(
        #     type='FPN',
        #     in_channels=[256, 512, 1024, 2048],
        #     out_channels=256,
        #     start_level=1,
        #     add_extra_convs='on_output',
        #     num_outs=5),
        dict(
            type='FPN',
            # in_channels=[96, 192, 384, 768], 
            in_channels=[192, 384, 768, 1536], # swin-L
            out_channels=256,
            num_outs=5),  
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=True,  # please set imgs/gpu >= 4
            pnorm_eval=False,
            lcnorm_eval=False,
            lcconv_padding=1)
    ],
    bbox_head=dict(
        type='GFLSEPCHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


##############################################################################################################
# dataset
# datasets/trash_detection.py

dataset_type = 'CocoDataset'
data_root = '../../dataset/' # 😀😀😀

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")  # 😀

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
    
    # dict(type="Mixup"), # not in the pipeline registry
    # dict(type="RandomRotate90", rotate_ratio=0.5), # not in the pipeline registry
    
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

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
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

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        # img_scale=[(256, 256), (512, 512)],
        img_scale=img_scale,
        flip=True,
        transforms=[
            dict(type="Resize", keep_ratio=True),
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
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2, # batch_size: 8 for 512, 2 for 1024
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json', # 😀 'train.json', 'train_split.json'
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes = classes, # 😀
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_split.json', # 😀
        img_prefix=data_root,
        pipeline=val_pipeline,
        classes = classes, # 😀
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# evaluation = dict(interval=1, metric='bbox')
evaluation = dict(
    save_best='auto', # save the best epoch model in terms of bbox mAP
    interval=1, # run validation by `interval` epochs
    metric='bbox')


##############################################################################################################
# optimizer

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=5e-2)
optimizer_config = dict(grad_clip=None)

# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr = 1e-10,
#     warmup='linear',
#     warmup_iters=3,
#     warmup_ratio=1e-4,
#     warmup_by_epoch=True
#     )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 37])

# runner = dict(type='EpochBasedRunner', max_epochs=48)

# fp16 = dict(loss_scale=512.)

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
# default runtime

# checkpoint_config = dict(interval=1)
checkpoint_config = dict(max_keep_ckpts=3, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = './work_dirs/_universe/epoch_22.pth' # None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
