""" 上采样模块 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 1 Upsample
# 2 ConvTranspose
# 3 PixelShuffle
# 4 Channel2Pix
# 5 grid sample


class Channel2Pix2D(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.r = scale_factor
        
    def forward(self, x):
        x = rearrange(x, 'b (r1 r2 c) h w -> b c (r1 h) (r2 w)',
                      r1=self.r, r2=self.r)
        return x


class Channel2Pix3D(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.r = scale_factor
        
    def forward(self, x):
        x = rearrange(x, 'b (r1 r2 r3 c) h w -> b c (r1 d) (r2 h) (r3 w)',
                      r1=self.r, r2=self.r, r3=self.r)
        return x


class UpGridSample2D(nn.Module):
    def __init__(self, H, W, scale_factor=2, align_corners=True):
        super().__init__()
        self.r = scale_factor
        self.align_corners = align_corners
        h, w = H * self.r, W * self.r

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # 可以通过神经网络习得
        self.register_buffer('grid', grid)

    def forward(self, x):
        grid = self.grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return F.grid_sample(x, grid, mode='bilinear', align_corners=self.align_corners)


class UpGridSample3D(nn.Module):
    def __init__(self, D, H, W, scale_factor=2, align_corners=True):
        super().__init__()
        self.r = scale_factor
        self.align_corners = align_corners
        d, h, w = D * self.r, H * self.r, W * self.r

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
    x = torch.randn(2, 16, 32, 32)
    
    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    z = up(x)  # (2, 16, 64, 64)
    
    up = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
    z = up(x)  # (2, 16, 64, 64)
    
    up = nn.PixelShuffle(upscale_factor=2)
    z = up(x)
    
    up = Channel2Pix2D(scale_factor=2)
    z = up(x)
    
    up = UpGridSample2D(H=32, W=32)
    z = up(x)

    print(z.shape)