python tools/train.py configs/yolov7/yolov7_tiny_sync_fast_2x16b-100e_exdark.py \
    --amp \
    --cfg-options \
        train_batch_size_per_gpu=8
        # visualizer.vis_backends.0.init_kwargs.id="4gjriogu"
    # --resume auto \
