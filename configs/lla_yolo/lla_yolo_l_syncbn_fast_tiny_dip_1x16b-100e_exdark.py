_base_ = "../yolov7/yolov7_l_syncbn_fast_1x16b-100e_exdark.py"

_base_.model.type = "LLAYOLODetector"
_base_.model.refiner = dict(type="TinyDIP")
