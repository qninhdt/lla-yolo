# python tools/train.py configs/yolov7/yolov7_tiny_syncbn_fast_2x16b-100e_exdark.py \
python tools/train.py configs/lla_yolo/lla_yolo_tiny_syncbn_fast_tiny_dip_2x16b-100e_exdark.py \
    --amp \
    --cfg-options \
        train_dataloader.batch_size=8 
        # visualizer.vis_backends.0.init_kwargs.id="x9ov3ri6"
    # --resume auto \
