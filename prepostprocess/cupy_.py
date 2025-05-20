"""
基于 Cupy 的数据处理\n
目前支持3D: 区域生长, 膨胀, 腐蚀, 开运算, 闭运算, 重采样, 翻转, 平移, 旋转, 距离变换
"""

import cupy as cp
from cupyx.scipy import ndimage
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, binary_fill_holes


def region_growth_3d(x_tgt, x_seed, max_steps=None):
    """
    x_tgt (numpy.Array): bool [D, H, W]
    x_seed (numpy.Array): bool [D, H, W]
    max_steps (int, optional): 最大生长步数, None表示不限制步数
    返回:
    numpy.Array: 生长后的分割结果, 与输入相同大小
    """
    x_tgt_cp  = cp.asarray(x_tgt)
    x_seed_cp = cp.asarray(x_seed)
    result_cp = cp.copy(x_seed_cp)
    depth, height, width = x_tgt_cp.shape
    if max_steps is None: max_steps = depth + height + width  # 最大可能步数

    visited_cp  = cp.copy(x_seed_cp)  # 已访问标记
    frontier_cp = cp.copy(x_seed_cp)  # 当前前沿区域
    
    # 定义卷积核以检查6个方向的邻居
    kernel = cp.zeros((3, 3, 3), dtype=cp.bool_)
    kernel[0, 1, 1] = True  # 前面
    kernel[2, 1, 1] = True  # 后面
    kernel[1, 0, 1] = True  # 上面
    kernel[1, 2, 1] = True  # 下面
    kernel[1, 1, 0] = True  # 左边
    kernel[1, 1, 2] = True  # 右边
    kernel = kernel.astype(cp.float32)
    
    # 开始区域生长
    for _ in range(max_steps):
        # 如果前沿为空，提前结束
        if not cp.any(frontier_cp): break
        
        # 使用Cupy的卷积操作查找所有前沿点的邻居
        # mode='constant'和cval=0确保边界外的点不被考虑
        neighbors = ndimage.convolve(frontier_cp.astype(cp.float32),
                                     kernel, mode='constant', cval=0.0)
        
        # 找出所有满足条件的新前沿点:
        # 是当前前沿点的邻居 & 未被访问过 & 在真实标签中是血管区域
        new_frontier = (neighbors > 0) & (~visited_cp) & x_tgt_cp

        visited_cp |= new_frontier  # 更新已访问标记
        result_cp  |= new_frontier  # 更新结果
        frontier_cp = new_frontier  # 更新前沿

    result_np = cp.asnumpy(result_cp)
    return result_np


def morph_operations_3d(mask, operation='dilate', kernel_size=3):
    """
    mask: numpy, [D, H, W]
    operation: 'dilate', 'erode', 'open', 'close', 'tophat', 'blackhat', 'gradient'
    kernel_size: 结构元素尺寸
    """
    mask_cp = cp.asarray(mask)
    
    # 创建3D结构元素
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    
    # 创建立方体或球形结构元素
    struct = cp.ones(kernel_size, dtype=cp.uint8)
    
    # 执行形态学操作
    if operation == 'dilate':      # 膨胀
        result = ndimage.binary_dilation(mask_cp, structure=struct)
    elif operation == 'erode':     # 腐蚀
        result = ndimage.binary_erosion(mask_cp, structure=struct)
    elif operation == 'open':      # 腐蚀 + 膨胀
        result = ndimage.binary_opening(mask_cp, structure=struct)
    elif operation == 'close':     # 膨胀 + 腐蚀
        result = ndimage.binary_closing(mask_cp, structure=struct)
    elif operation == 'tophat':    # 白顶帽：原图像减去开运算结果 => 细物体
        opened = ndimage.binary_opening(mask_cp, structure=struct)
        result = mask_cp - opened
    elif operation == 'blackhat':  # 黑顶帽：闭运算结果减去原图像 => 缝隙区域
        closed = ndimage.binary_closing(mask_cp, structure=struct)
        result = closed - mask_cp
    elif operation == 'gradient':  # 形态学梯度：膨胀减腐蚀 => 边缘
        dilated = ndimage.binary_dilation(mask_cp, structure=struct)
        eroded = ndimage.binary_erosion(mask_cp, structure=struct)
        result = dilated ^ eroded
    else:
        raise ValueError(f"not supported: {operation} ['dilate', 'erode', 'open', 'close', 'tophat', 'blackhat', 'gradient']")

    result_np = cp.asnumpy(result)
    return result_np


def resample_3d(mask, new_shape=None, scale_factor=None, order=0):
    """
    mask: numpy [D, H, W]
    new_shape: (new_D, new_H, new_W)
    scale_factor: 浮点数或元组, 各维度的缩放因子
    order: 整数, 0=最近邻, 1=线性, 2=二次, 3=三次
    """
    mask_cp = cp.asarray(mask)
    old_shape = mask_cp.shape
    
    # 计算新形状
    if new_shape is not None:
        zoom_factors = tuple(new / old for new, old in zip(new_shape, old_shape))
    elif scale_factor is not None:
        zoom_factors = (scale_factor,) * 3
    else: raise

    result = ndimage.zoom(mask_cp, zoom_factors, order=order)

    result_np = cp.asnumpy(result)
    return result_np


def flip_3d(mask, axes=None):
    """
    mask: numpy数组或torch.Tensor, 3D掩码 [D, H, W]
    axes: 整数或整数元组, 要翻转的轴。0=z轴, 1=y轴, 2=x轴
    """
    mask_cp = cp.asarray(mask)
    
    # 执行翻转操作
    result = cp.flip(mask_cp, axes)

    result_np = cp.asnumpy(result)
    return result_np


def translate_3d(mask, shifts=(0,0,0), order=0):
    """
    mask: numpy [D, H, W]
    shifts: 元组，(z_shift, y_shift, x_shift)，各维度的平移量（像素）
    order: 整数，插值阶数
    """
    mask_cp = cp.asarray(mask)
    
    # 执行平移操作
    result = ndimage.shift(mask_cp, shifts, order=order)
    
    # 转回原始类型
    result_np = cp.asnumpy(result)
    
    return result_np

def rotate_3d(mask, angles=(0,0,0), axes=(0,1,2), reshape=False, order=0):
    """
    mask: numpy [D, H, W]
    angles: 各轴的旋转角度(度), 按顺序绕axes指定的轴旋转
    axes: 旋转轴的顺序，例如(0,1,2)表示分别绕z,y,x轴旋转
    reshape: 布尔值, 是否调整输出形状以容纳完整的旋转图像
    order: 整数, 插值阶数
    """
    mask_cp = cp.asarray(mask)
    result = mask_cp
    
    # 按指定轴顺序依次旋转
    for i, (axis1, axis2) in enumerate([(1, 2), (0, 2), (0, 1)]):
        if i < len(angles) and angles[i] != 0:
            # 注意：CuPy的rotate是逆时针旋转，而很多其他库是顺时针
            result = ndimage.rotate(result, angles[i], axes=(axis1, axis2), 
                                   reshape=reshape, order=order)

    result_np = cp.asnumpy(result)
    return result_np


def distance_transform_3d(mask):
    """
    mask: bool, numpy
    距离变换
    """
    mask_cp = cp.asarray(mask)
    out = ndimage.distance_transform_edt(mask_cp)
    out = cp.asnumpy(out)
    return out


def connected_components(mask, idx=None):
    """ 连通域分析 """
    mask_cp = cp.asarray(mask)
    labeled_cp, num_labels = ndimage.label(mask_cp)

    labels, counts = cp.unique(labeled_cp[labeled_cp > 0], return_counts=True)
    sorted_indices = cp.argsort(counts)[::-1]
    sorted_labels = labels[sorted_indices]

    mask_out = cp.zeros_like(labeled_cp, dtype=cp.int32)
    if idx is not None:
        selected_labels = sorted_labels[:idx]
        for v, label in enumerate(selected_labels):
            mask_out[labeled_cp == label] = v + 1
    else:
        mask_out = labeled_cp

    return cp.asnumpy(mask_out)


def sobel_edges_2d(img):
    img_cp = cp.asarray(img)
    dx = ndimage.sobel(img_cp, axis=0)  # x 方向梯度
    dy = ndimage.sobel(img_cp, axis=1)  # y 方向梯度
    edges = cp.hypot(dx, dy)  # 计算梯度幅值
    return cp.asnumpy(edges)


def sobel_edges_3d(volume):
    volume_cp = cp.asarray(volume)
    dx = ndimage.sobel(volume_cp, axis=0)
    dy = ndimage.sobel(volume_cp, axis=1)
    dz = ndimage.sobel(volume_cp, axis=2)
    edges = cp.sqrt(dx**2 + dy**2 + dz**2)
    return cp.asnumpy(edges)


def gaussian_smoothing_3d(volume, sigma=1.0):
    """
    高斯滤波, sigma越大越模糊
    """
    volume_cp = cp.asarray(volume)
    smoothed = gaussian_filter(volume_cp, sigma=sigma)
    return cp.asnumpy(smoothed)


def mean_filter_3d(volume, size=3):
    """
    均值卷积, size越大越模糊
    """
    volume_cp = cp.asarray(volume)
    filtered = uniform_filter(volume_cp, size=size)
    return cp.asnumpy(filtered)


def laplacian_sharpen_3d(volume, alpha=1.0):
    """拉普拉斯锐化, alpha值越大, 纹理越清晰"""
    volume_cp = cp.asarray(volume)
    laplacian = ndimage.laplace(volume_cp)  # 计算拉普拉斯
    sharpened = volume_cp - alpha * laplacian  # 原图 - Laplacian
    return cp.asnumpy(sharpened)


def fill_holes_3d(mask):
    """ 填充空洞 """
    mask_cp = cp.asarray(mask)
    filled = binary_fill_holes(mask_cp)
    return cp.asnumpy(filled)