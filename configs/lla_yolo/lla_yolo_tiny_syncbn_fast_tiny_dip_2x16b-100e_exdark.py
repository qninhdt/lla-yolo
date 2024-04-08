_base_ = "../yolov7/yolov7_tiny_syncbn_fast_2x16b-100e_exdark.py"

_base_.model.type = "LLAYOLODetector"
_base_.model.refiner = dict(type="TinyDIP")
