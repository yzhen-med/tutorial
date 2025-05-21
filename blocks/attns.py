""" 注意力模块 """


import torch
import torch.nn as nn
from blocks.convs import ConvNorm


class SoftGate(nn.Module):
    """
    对每个空间特征计算权重
    see paper: https://arxiv.org/pdf/1804.03999v3
    see codes: https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, c_in1, c_in2, c_out, k=1, s=1, p=0, d=1, g=1,
                 conv_layer=ConvNorm):
        super().__init__()
        self.conv1 = ConvNorm(c_in1, c_out, k, s, p, d, g)
        self.conv2 = ConvNorm(c_in2, c_out, k, s, p, d, g)
        self.gate = nn.Sequential(
            ConvNorm(c_out, 1, k, s, p, d, g),
            nn.Sigmoid())
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2):
        attn = self.relu(self.conv1(x1) + self.conv2(x2))
        attn = self.gate(attn)  # [b,1,h,w]
        x2 = attn * x2
        return x2