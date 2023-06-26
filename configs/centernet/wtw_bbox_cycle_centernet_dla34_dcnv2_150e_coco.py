TEST_NAME = "31_long_check"
EVAL_LAG = 30
CHECKPOINT_LAG = 5
EPOCHS = 150
LR = 0.00125
BACKBONE = "DLANetMMDet3D"
BATCH = 8
local_maximum_kernel = 1
LOG_LAG = 100
TAGS = [
    f"local_maximum_kernel={local_maximum_kernel}",
    "1hm",
    f"{EPOCHS}_epoch",
    f"{LR}_lr",
    f"{EVAL_LAG}_eval_lag",
    f"{CHECKPOINT_LAG}_chkpt_lag",
    f"{BACKBONE}_backbone",
    f"{BATCH}_batch",
]
# DATA AND AUG
dataset_type = "CocoDataset"
data_root = "/home/aiarhipov/datasets/WTW-dataset/"

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

data = dict(
    samples_per_gpu=BATCH,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="/home/aiarhipov/datasets/WTW-dataset/train/train.json",
        img_prefix="train/images/",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True, color_type="color"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="PhotoMetricDistortion",
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            ),
            dict(
                type="RandomCenterCropPad",
                crop_size=(1024, 1024),
                ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_pad_mode=None,
            ),
            dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False,
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
        data_root="/home/aiarhipov/datasets/WTW-dataset/",
        classes=("box",),
    ),
    val_loss=dict(
        type=dataset_type,
        ann_file="/home/aiarhipov/datasets/WTW-dataset/test/test.json",
        img_prefix="test/images/",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True, color_type="color"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="PhotoMetricDistortion",
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            ),
            dict(
                type="RandomCenterCropPad",
                crop_size=(1024, 1024),
                ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_pad_mode=None,
            ),
            dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False,
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
        data_root="/home/aiarhipov/datasets/WTW-dataset/",
        classes=("box",),
    ),
    val=dict(
        type=dataset_type,
        ann_file="/home/aiarhipov/datasets/WTW-dataset/test/test.json",
        img_prefix="test/images/",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True),
            dict(
                type="MultiScaleFlipAug",
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=["logical_or", 31],
                        test_pad_add_pix=1,
                    ),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False,
                    ),
                    dict(type="DefaultFormatBundle"),
                    dict(
                        type="Collect",
                        meta_keys=(
                            "filename",
                            "ori_filename",
                            "ori_shape",
                            "img_shape",
                            "pad_shape",
                            "scale_factor",
                            "flip",
                            "flip_direction",
                            "img_norm_cfg",
                            "border",
                        ),
                        keys=["img"],
                    ),
                ],
            ),
        ],
        data_root="/home/aiarhipov/datasets/WTW-dataset/",
        classes=("box",),
    ),
    test=dict(
        type=dataset_type,
        ann_file="/home/aiarhipov/datasets/WTW-dataset/test/test.json",
        img_prefix="test/images/",
        pipeline=[
            dict(type="LoadImageFromFile", to_float32=True),
            dict(
                type="MultiScaleFlipAug",
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(
                        type="RandomCenterCropPad",
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=["logical_or", 31],
                        test_pad_add_pix=1,
                    ),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False,
                    ),
                    dict(type="DefaultFormatBundle"),
                    dict(
                        type="Collect",
                        meta_keys=(
                            "filename",
                            "ori_filename",
                            "ori_shape",
                            "img_shape",
                            "pad_shape",
                            "scale_factor",
                            "flip",
                            "flip_direction",
                            "img_norm_cfg",
                            "border",
                        ),
                        keys=["img"],
                    ),
                ],
            ),
        ],
        data_root="/home/aiarhipov/datasets/WTW-dataset/",
        classes=("box",),
    ),
)


# MODEL CycleCenterNet(dcnv2) DLANetMMDet3D,
load_from = None
resume_from = None

model = dict(
    type="CenterNet",
    backbone=dict(
        type=BACKBONE,
        depth=34,
        norm_cfg=dict(type="BN"),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="http://dl.yf.io/dla/models/imagenet/dla34%2Btricks-24a49e58.pth",
        ),
    ),
    neck=dict(
        type="CTResNetNeck",
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True,
    ),
    bbox_head=dict(
        type="CycleCenterNetHead",
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type="GaussianFocalLoss", loss_weight=1.0),
        loss_offset=dict(type="L1Loss", loss_weight=1.0),
        loss_c2v=dict(type="L1Loss", loss_weight=1.0),
        loss_v2c=dict(type="L1Loss", loss_weight=0.5),
    ),
    train_cfg=None,
    test_cfg=dict(
        topk=3000,
        local_maximum_kernel=local_maximum_kernel,
        max_per_img=3000,
        # nms=dict(type="nms", iou_threshold=0.475, split_thr=2000),
    ),
)


# GPU
gpu_ids = [5]
device = "cuda"


# OPTIMIZATION
optimizer = dict(type="SGD", lr=LR, momentum=0.9, weight_decay=0.0001)
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


# LEARNING POLICY
runner = dict(type="EpochBasedRunner", max_epochs=EPOCHS)  # the real epoch is 28*5=140

# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[90, 120],  # the real step is [18*5, 24*5]
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)


# LOGGING
work_dir = f"/home/aiarhipov/centernet/exps/{TEST_NAME}"
INTERVAL = int(10976 / (LOG_LAG * BATCH)) + (10976 % (LOG_LAG * BATCH) > 0)
log_config = dict(
    interval=INTERVAL,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": "CenterNet",
                "entity": "centernet",
                "name": TEST_NAME,
                "dir": "/home/aiarhipov/centernet/exps/wandb",
                "tags": TAGS,
            },
            interval=INTERVAL,
            log_checkpoint=True,
            log_checkpoint_metadata=False,
            num_eval_images=5,
        ),
    ],
)
log_level = "INFO"


# EVALUATION
evaluation = dict(interval=EVAL_LAG, metric="bbox")
checkpoint_config = dict(interval=CHECKPOINT_LAG, max_keep_ckpts=5)


# RUNTIME
seed = 0

custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
workflow = [("train", 1), ("val", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"
