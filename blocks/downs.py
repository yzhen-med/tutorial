""" 下采样模块 """


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 1 maxpool
# 2 avgpool
# 3 adaptive avgpool
# 4 pixel unshuffle
# 5 pix2Channel
# 6 convnet (stride > 1), see conv_blocks
# 7 grid sample


class Pix2Channel2D(nn.Module):
    def __init__(self, downscale_factor=2):
        super().__init__()
        self.r = downscale_factor
        
    def forward(self, x):
        x = rearrange(x, 'b c (r1 h) (r2 w) -> b (r1 r2 c) h w',
                      r1=self.r, r2=self.r)
        return x


class Pix2Channel3D(nn.Module):
    def __init__(self, downscale_factor=2):
        super().__init__()
        self.r = downscale_factor
        
    def forward(self, x):
        x = rearrange(x, 'b c (r1 d) (r2 h) (r3 w) -> b (r1 r2 r3 c) h w',
                      r1=self.r, r2=self.r, r3=self.r)
        return x


class DownGridSample2D(nn.Module):
    def __init__(self, H, W, downscale_factor=2, align_corners=True):
        super().__init__()
        self.r = downscale_factor
        self.align_corners = align_corners
        h, w = H // self.r, W // self.r

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # 可以通过神经网络习得
        self.register_buffer('grid', grid)

    def forward(self, x):
        grid = self.grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return F.grid_sample(x, grid, mode='bilinear', align_corners=self.align_corners)


class DownGridSample3D(nn.Module):
    def __init__(self, D, H, W, downscale_factor=2, align_corners=True):
        super().__init__()
        self.r = downscale_factor
        self.align_corners = align_corners
        d, h, w = D // self.r, H // self.r, W // self.r

        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        self.register_buffer('grid', grid)

    def forward(self, x):
        grid = self.grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        return F.grid_sample(x, grid, mode='bilinear', align_corners=self.align_corners)


# ============== examples ==============
if __name__ == '__main__':
    x = torch.randn(2, 16, 64, 64)
    
    down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
    z = down(x)  # (2, 16, 32, 32)
    
    down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    z = down(x)  # (2, 16, 32, 32)
    
    down = nn.AdaptiveAvgPool2d(output_size=(32, 32))
    z = down(x)  # (2, 16, 32, 32)
    
    down = nn.PixelUnshuffle(downscale_factor=2)  # 重排, 信息保存在通道中
    z = down(x)  # (2, 64, 32, 32)
    
    down = Pix2Channel2D(downscale_factor=2)
    z = down(x)
    
    down = DownGridSample2D(64, 64)
    z = down(x)

    print(z.shape)