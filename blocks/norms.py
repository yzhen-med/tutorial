""" 归一化模块 """


import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# 1 Batch
# 2 Instance
# 3 Layer
# 4 Group
# 5 Pixel
# 6 Spectral
# 7 Switchable
# 8 Filter Response
# 9 Weight


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)


# ============== examples ==============
if __name__ == '__main__':
    x = torch.randn(2, 16, 64, 64)

    conv = spectral_norm(nn.Conv2d(3, 64, 3))

    norm = nn.BatchNorm2d(num_features=16)
    z = norm(x)

    norm = nn.InstanceNorm2d(num_features=16, affine=True)
    z = norm(x)

    norm = nn.LayerNorm([16, 64, 64])
    z = norm(x)

    norm = nn.GroupNorm(num_groups=8, num_channels=16)
    z = norm(x)