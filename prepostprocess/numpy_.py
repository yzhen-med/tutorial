""" 基于 Numpy 的数据处理 """

import numpy as np
from numba import njit, prange


@njit(parallel=True)  # 使用Numba加速的区域生长函数
def region_grow(x_tgt, x_seed, max_steps=None):
    """
    y_true (numpy.Array): bool [D, H, W]
    y_pred (numpy.Array): bool [D, H, W]
    max_steps (int, optional): 最大生长步数, None表示不限制步数
    返回:
    numpy.Array: 生长后的分割结果, 与输入相同大小
    """
    out = x_seed.copy()
    depth, height, width = x_tgt.shape
    
    # 方向偏移量: 左右上下前后
    offsets = np.array([
        [0, 0, 1], [0, 0, -1],
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0]])
    
    # 访问标记数组
    visited = np.zeros((depth, height, width), dtype=np.bool_)
    visited |= x_seed  # 标记所有初始种子点为已访问
    
    # 创建前沿区域 - 当前步骤中要检查的点
    frontier = np.zeros((depth, height, width), dtype=np.bool_)
    frontier |= x_seed  # 初始前沿为预测区域
    
    # 创建下一步前沿区域
    next_frontier = np.zeros((depth, height, width), dtype=np.bool_)
    
    # 根据max_steps设置迭代次数
    n_steps = max_steps if max_steps is not None else depth + height + width
    
    # 开始区域生长
    for _ in range(n_steps):
        # 如果前沿为空，提前结束
        if not np.any(frontier): break
            
        # 并行处理每个维度
        for d in prange(depth):
            for h in range(height):
                for w in range(width):
                    # 只处理前沿区域中的点
                    if frontier[d, h, w]:
                        # 检查周围六个方向
                        for offset in offsets:
                            nd, nh, nw = d + offset[0], h + offset[1], w + offset[2]
                            
                            # 边界检查
                            if (0 <= nd < depth and 0 <= nh < height and 0 <= nw < width):
                                # 如果点未访问过且目标区域有值
                                if not visited[nd, nh, nw] and x_tgt[nd, nh, nw]:
                                    # 标记为已访问
                                    visited[nd, nh, nw] = True
                                    # 更新结果
                                    out[nd, nh, nw] = True
                                    # 加入下一步的前沿
                                    next_frontier[nd, nh, nw] = True
        
        # 更新前沿区域为下一步前沿
        frontier[:] = next_frontier[:]
        # 清空下一步前沿
        next_frontier[:] = False
    return out