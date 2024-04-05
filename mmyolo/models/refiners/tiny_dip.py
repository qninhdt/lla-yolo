import torch
import torch.nn.functional as F

from mmyolo.registry import MODELS


@MODELS.register_module()
class TinyDIP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)

        # alpha, beta, gamma
        self.linear1 = torch.nn.Linear(32, 3)

    def compute_params(self, x):
        x = F.interpolate(x, scale_factor=0.125)  # 1/8

        x = F.relu(self.conv1(x))  # 1/16
        x = F.relu(self.conv2(x))  # 1/32
        x = F.relu(self.conv3(x))  # 1/64

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)

        x = self.linear1(x)

        return x

    def soft_clip(self, x, min_val, max_val):
        scale = (max_val - min_val) / 2
        x = (scale * torch.tanh((x - min_val) / (max_val - min_val) + 1)) + min_val
        return x

    def forward(self, x):
        params = self.compute_params(x)

        alpha = params[:, 0].view(-1, 1, 1, 1)
        beta = params[:, 1].view(-1, 1, 1, 1)
        gamma = params[:, 2].view(-1, 1, 1, 1)

        gamma = self.soft_clip(gamma, 0.1, 2)

        # I(x) = alpha * I(x) + beta
        x = alpha * x + beta

        # I(x) = I(x) ** gamma
        x = x**gamma

        return x
