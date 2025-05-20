"""
基于Pytorch的数据处理, 通常用于预处理和数据增强
目前支持3D: 区域生长, 高斯模糊, gamma矫正, 随机剪裁, 高斯噪声, 随机强度, 随机翻转, 随机旋转
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==== utils ==== #
def gaussian_kernel_1d(kernel_size, sigma):
    """创建一维高斯核"""
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    return kernel / kernel.sum()  # 归一化


# ==== 基于图像的操作 ==== #
def gamma_correction(x, gamma_range=(0.8, 1.2)):
    """
    x^(gamma)
    """
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    return x.pow(gamma)


def gaussian_blur_3d(x, kernel_size_max=7, sigma_range=(0.5, 1.5), dev='cuda'):
    """
    sigma越大越模糊(影响最大)\n
    kernel越大越模糊
    """
    b, c, d, h, w = x.shape

    # 随机选择核大小和sigma
    kernel_size = random.randrange(3, kernel_size_max + 1, 2)
    sigma = random.uniform(sigma_range[0], sigma_range[1])

    # 创建高斯核
    gaussian_1d = gaussian_kernel_1d(kernel_size, sigma).to(dev)

    # 扩展高斯核，准备进行卷积
    weight_d = gaussian_1d.view(1, 1, -1, 1, 1)  # D维度卷积核
    weight_d = weight_d.expand(c, 1, -1, 1, 1)   # 每个通道使用相同的D方向卷积核
    weight_h = gaussian_1d.view(1, 1, 1, -1, 1)  # H维度卷积核
    weight_h = weight_h.expand(c, 1, 1, -1, 1)
    weight_w = gaussian_1d.view(1, 1, 1, 1, -1)  # W维度卷积核
    weight_w = weight_w.expand(c, 1, 1, 1, -1)

    pad = kernel_size // 2

    x = F.conv3d(x, weight_d, padding=(pad, 0, 0), groups=c)  # groups=c保证每个通道独立卷积
    x = F.conv3d(x, weight_h, padding=(0, pad, 0), groups=c)
    x = F.conv3d(x, weight_w, padding=(0, 0, pad), groups=c)
    return x


def random_crop_3d(x, min_zoom=1.1, max_zoom=1.4):
    """随机裁剪"""
    b, c, d, h, w = x.shape
    result = []
    
    for j in range(b):
        # 计算裁剪尺寸
        crop_d = int(d / random.uniform(min_zoom, max_zoom))
        crop_h = int(h / random.uniform(min_zoom, max_zoom))
        crop_w = int(w / random.uniform(min_zoom, max_zoom))
        
        # 随机选择裁剪起始位置
        st_d = random.randint(0, d - crop_d)
        st_h = random.randint(0, h - crop_h)
        st_w = random.randint(0, w - crop_w)
        
        # 裁剪
        cropped = x[j:j+1, :,
                    st_d:st_d+crop_d, 
                    st_h:st_h+crop_h,
                    st_w:st_w+crop_w]

        resized = F.interpolate(cropped, size=(d, h, w), mode='trilinear', align_corners=False)
        result.append(resized)
    return torch.cat(result, dim=0)


def gaussian_noise(x, mean=0.0, std_range=(0.0, 0.05), dev='cuda'):
    """高斯噪声"""
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(x, device=dev) * std + mean
    return x + noise


def random_intensity_shift(x, shift_range=(-0.1, 0.1)):
    """随机增加/减少图像强度"""
    b = x.shape[0]
    shifted_volumes = []

    for i in range(b):
        shift_factor = random.uniform(shift_range[0], shift_range[1])
        shifted_volumes.append(x[i] + shift_factor)

    return torch.stack(shifted_volumes, dim=0)


def random_intensity_scale(volume, scale_range=(0.9, 1.1)):
    """随机缩放图像强度"""
    b = volume.shape[0]
    scaled_volumes = []
    
    for i in range(b):
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        scaled_volumes.append(volume[i] * scale_factor)
    
    return torch.stack(scaled_volumes, dim=0)


# ==== 基于标签的操作 ==== # 
def region_grow_3d(x_tgt, x_seed, max_steps=None):
    """
    y_true (torch.Tensor): bool [D, H, W]
    y_pred (torch.Tensor): bool [D, H, W]
    max_steps (int, optional): 最大生长步数, None表示不限制步数
    返回:
    torch.Tensor: 生长后的分割结果, 与输入相同大小
    """
    device = x_tgt.device
    depth, height, width = x_tgt.shape

    if max_steps == 0: return x_seed.clone()
    if max_steps is None:
        max_steps = depth + height + width  # 最大可能步数

    result = x_seed.clone()
    
    # 创建前沿区域, 当前步骤要检查的点
    frontier = x_seed.clone()
    frontier_expanded = frontier[None, None].float()
    
    # 创建已访问标记
    visited = x_seed.clone()
    visited_expanded = visited[None, None]

    x_tgt_expanded = x_tgt[None, None]
    
    # 创建3D卷积核以检查6个方向的邻居
    # 使用3D卷积计算邻居状态
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device)
    kernel[0, 0, 0, 1, 1] = 1  # 前面
    kernel[0, 0, 2, 1, 1] = 1  # 后面
    kernel[0, 0, 1, 0, 1] = 1  # 上面
    kernel[0, 0, 1, 2, 1] = 1  # 下面
    kernel[0, 0, 1, 1, 0] = 1  # 左边
    kernel[0, 0, 1, 1, 2] = 1  # 右边

    # 开始区域生长
    for _ in range(max_steps):
        if not torch.any(frontier): break
        
        # 使用卷积操作查找所有前沿点的邻居
        # padding=1 确保边界点也被考虑
        neighbors = F.conv3d(frontier_expanded, kernel, padding=1)
        
        # 找出所有满足条件的新前沿点:
        # 1. 点是当前前沿点的邻居
        # 2. 点未被访问过
        # 3. 点在真实标签中是血管区域
        # new_frontier = (neighbors > 0) & (~visited_expanded.bool()) & y_true_expanded.bool()
        new_frontier = (neighbors > 0) & (~visited_expanded) & x_tgt_expanded
        
        # 更新已访问标记
        visited_expanded = visited_expanded | new_frontier
        
        # 更新结果
        result = result | new_frontier[0, 0]
        
        # 更新前沿
        frontier_expanded = new_frontier.float()
    return result


# ==== 通用的空间变换 ==== #
def random_flip(x):
    """随机沿一个或多个轴翻转图像"""
    axes = [2, 3, 4]  # d, h, w轴
    flip_axes = [axis for axis in axes if random.random() < 0.5]
    
    if not flip_axes: return x

    for axis in flip_axes:
        x = torch.flip(x, [axis])
    return x


def random_rotate(x, max_angle=45.0, dev='cuda', mode='bilinear'):
    """
    对整个batch进行3D旋转
    使用仿射变换进行旋转，确保计算高效
    """
    angles = [random.uniform(-max_angle, max_angle) * np.pi / 180.0 for _ in range(3)]
    
    # 获取体积形状
    b, c, d, h, w = x.shape

    # 创建旋转矩阵
    ones_col = torch.tensor([[[0], [0], [0]]], dtype=torch.float32, device=dev)  # [1, 3, 1]
    theta_dh = torch.tensor([
        [np.cos(angles[0]), -np.sin(angles[0]), 0],
        [np.sin(angles[0]), np.cos(angles[0]), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=dev).unsqueeze(0)
    theta_dh = torch.cat([theta_dh, ones_col], dim=2)  # [1, 3, 4]
    theta_dh = theta_dh.expand(b, -1, -1)  # [b, 3, 4]

    theta_dw = torch.tensor([
        [np.cos(angles[1]), 0, -np.sin(angles[1])],
        [0, 1, 0],
        [np.sin(angles[1]), 0, np.cos(angles[1])]
    ], dtype=torch.float32, device=dev).unsqueeze(0)
    theta_dw = torch.cat([theta_dw, ones_col], dim=2)  # [1, 3, 4]
    theta_dw = theta_dw.expand(b, -1, -1)  # [b, 3, 4]

    theta_hw = torch.tensor([
        [1, 0, 0],
        [0, np.cos(angles[2]), -np.sin(angles[2])],
        [0, np.sin(angles[2]), np.cos(angles[2])]
    ], dtype=torch.float32, device=dev).unsqueeze(0)
    theta_hw = torch.cat([theta_hw, ones_col], dim=2)  # [1, 3, 4]
    theta_hw = theta_hw.expand(b, -1, -1)  # [b, 3, 4]

    # 在 D-H 平面旋转
    grid_dh = F.affine_grid(theta_hw, [b, c, d, h, w], align_corners=False)
    x = F.grid_sample(x, grid_dh, align_corners=False, mode=mode, padding_mode='border')

    # 在 D-W 平面旋转
    grid_dw = F.affine_grid(theta_dw, [b, c, d, h, w], align_corners=False)
    x = F.grid_sample(x, grid_dw, align_corners=False, mode=mode, padding_mode='border')

    # 在 H-W 平面旋转
    grid_hw = F.affine_grid(theta_dh, [b, c, d, h, w], align_corners=False)
    x = F.grid_sample(x, grid_hw, align_corners=False, mode=mode, padding_mode='border')
    return x


class ElasticDeformation3D:
    def __init__(self, sigma=15, alpha=50, device='cpu'):
        """
        sigma: 高斯滤波器的标准差，控制形变的平滑度
        alpha: 形变强度，值越大形变越明显
        device: 计算设备 (CPU/GPU)
        """
        self.sigma = sigma
        self.alpha = alpha
        self.device = device
        
    def _gaussian_filter(self, input_tensor, kernel_size=15):
        """
        应用高斯滤波器进行平滑
        
        参数:
            input_tensor: 输入张量
            kernel_size: 高斯核大小
        """
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 获取通道数
        channels = input_tensor.shape[0]
        
        # 创建1D高斯核
        x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32, device=self.device)
        gaussian_1d = torch.exp(-x**2 / (2 * self.sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 进行单独的高斯滤波而不是用分组卷积
        result = torch.zeros_like(input_tensor)
        
        for c in range(channels):
            channel_data = input_tensor[c].unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            
            # 沿D维度进行卷积
            pad_size = kernel_size // 2
            padded = F.pad(channel_data, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
            gaussian_1d_d = gaussian_1d.view(1, 1, -1, 1, 1)
            out_d = F.conv3d(padded, gaussian_1d_d, padding=0)
            
            # 沿H维度进行卷积
            padded = F.pad(out_d, (0, 0, pad_size, pad_size, 0, 0), mode='replicate')
            gaussian_1d_h = gaussian_1d.view(1, 1, 1, -1, 1)
            out_h = F.conv3d(padded, gaussian_1d_h, padding=0)
            
            # 沿W维度进行卷积
            padded = F.pad(out_h, (pad_size, pad_size, 0, 0, 0, 0), mode='replicate')
            gaussian_1d_w = gaussian_1d.view(1, 1, 1, 1, -1)
            out_w = F.conv3d(padded, gaussian_1d_w, padding=0)
            
            result[c] = out_w.squeeze(0).squeeze(0)
            
        return result
    
    def _create_displacement_field(self, shape):
        """
        创建随机位移场
        
        参数:
            shape: 输入图像的形状 (D, H, W)
        """
        d, h, w = shape
        
        # 为每个维度创建一个随机场
        displacement_field = torch.randn(3, d, h, w, device=self.device) * self.alpha
        
        # 应用高斯滤波平滑随机场
        displacement_field = self._gaussian_filter(displacement_field)
        return displacement_field
    
    def _create_sampling_grid(self, displacement_field, shape):
        """
        基于位移场创建采样网格
        
        参数:
            displacement_field: 位移场
            shape: 输入图像的形状 (D, H, W)
        """
        d, h, w = shape
        
        # 创建标准网格
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, d, device=self.device),
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij')
        
        # 标准网格的参考点
        grid = torch.stack([x, y, z], dim=0)  # (3, D, H, W)
        
        # 归一化位移场以适应网格坐标范围 (-1, 1)
        norm_factor = torch.tensor([w/2, h/2, d/2], device=self.device).view(3, 1, 1, 1)
        displacement_field = displacement_field / norm_factor
        
        # 应用位移
        grid = grid + displacement_field
        
        # 重新排列轴顺序，使其符合PyTorch的grid_sample期望
        # 从 (3, D, H, W) 到 (D, H, W, 3)
        grid = grid.permute(1, 2, 3, 0)
        
        return grid
        
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, device=self.device)
            
        # 确保图像是5D张量 (B, C, D, H, W)
        if image.dim() == 3:  # (D, H, W)
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:  # (C, D, H, W)
            image = image.unsqueeze(0)
            
        assert image.dim() == 5, "输入图像必须是5D张量 (B, C, D, H, W)"
        
        _, _, d, h, w = image.shape
        
        # 创建位移场
        displacement_field = self._create_displacement_field((d, h, w))
        
        # 创建采样网格
        sampling_grid = self._create_sampling_grid(displacement_field, (d, h, w))
        
        # 应用网格采样进行形变
        deformed_image = F.grid_sample(
            image, 
            sampling_grid.unsqueeze(0).repeat(image.size(0), 1, 1, 1, 1), 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return deformed_image