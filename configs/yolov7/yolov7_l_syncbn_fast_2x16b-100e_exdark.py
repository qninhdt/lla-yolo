_base_ = "./yolov7_l_syncbn_fast_8x16b-300e_coco.py"

data_root = "./data/exdark_coco/"
class_name = (
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
)
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (0, 0, 110),
        (0, 0, 230),
        (0, 0, 230),
        (0, 0, 230),
    ],
)

anchors = [
    [(45, 54), (69, 128), (140, 100)],
    [(116, 223), (240, 171), (186, 344)],
    [(369, 232), (329, 457), (549, 313)],
]

max_epochs = 100
train_batch_size_per_gpu = 16
train_num_workers = 8

_base_.optim_wrapper.optimizer.lr = 0.001

find_unused_parameters = True

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth"  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
    ),
)

train_pipeline = [
    *_base_.pre_transform,
    dict(type="mmdet.RandomFlip", prob=0.5),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
        ),
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/"),
    )
)

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/test.json",
        data_prefix=dict(img="images/"),
    )
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + "annotations/val.json")
test_evaluator = dict(ann_file=data_root + "annotations/test.json")

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best="auto"),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type="LoggerHook", interval=5),
)
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
visualizer = dict(
    vis_backends=[
        dict(
            type="WandbVisBackend",
            init_kwargs=dict(project="lla-yolo", resume="allow", allow_val_change=True),
        ),
    ],
)  # noqa
