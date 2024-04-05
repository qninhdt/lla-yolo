from torch import Tensor
from .yolo_detector import YOLODetector

from mmyolo.registry import MODELS


@MODELS.register_module()
class LLAYOLODetector(YOLODetector):

    def __init__(self, refiner, **kwargs):
        super().__init__(**kwargs)
        self.refiner = MODELS.build(refiner)

    def extract_feat(self, x: Tensor) -> Tensor:
        x = self.refiner(x)
        x = super().extract_feat(x)
        return x
