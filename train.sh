python tools/train.py configs/yolov7/yolov7_l_syncbn_fast_1x16b-100e_exdark.py \
    --amp \
    --cfg-options \
        train_dataloader.batch_size=4
        # visualizer.vis_backends.0.init_kwargs.id="4gjriogu"
    # --resume auto \
