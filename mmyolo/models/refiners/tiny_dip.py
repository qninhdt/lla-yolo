import torch
import torchvision.transforms.functional as F

from mmyolo.registry import MODELS


@MODELS.register_module()
class TinyDIP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # convert to uint8
        uint_x = (x * 255).to(torch.uint8)
        uint_x = torch.stack([F.equalize(uint_x[i]) for i in range(x.size(0))])
        x = uint_x.float() / 255
        return x
