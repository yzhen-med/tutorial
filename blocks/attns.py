""" 注意力模块 """


import torch
import torch.nn as nn
from blocks.convs import ConvNorm, ConvAct


class SoftGate(nn.Module):
    """
    对每个空间特征计算权重
    see paper: https://arxiv.org/pdf/1804.03999v3
    see codes: https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, c_in1, c_in2, c_out, k=1, s=1, p=0, d=1, g=1,
                 conv_layer=ConvNorm):
        super().__init__()
        self.conv1 = conv_layer(c_in1, c_out, k, s, p, d, g)
        self.conv2 = conv_layer(c_in2, c_out, k, s, p, d, g)
        self.gate = nn.Sequential(
            conv_layer(c_out, 1, k, s, p, d, g),
            nn.Sigmoid())
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2):
        attn = self.relu(self.conv1(x1) + self.conv2(x2))
        attn = self.gate(attn)  # [b,1,h,w]
        x2 = attn * x2
        return x2


class SqueezeExcitation(torch.nn.Module):
    """
    通道注意力
    see code: https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L225
    """
    def __init__(self, c_in, c_squeeze,
                 fc_layer=ConvAct):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            fc_layer(c_in, c_squeeze, k=1, s=1, p=0),
            fc_layer(c_squeeze, c_in, k=1, s=1, p=0, ACT=nn.Sigmoid))

    def _scale(self, x):
        scale = self.avgpool(x)
        scale = self.se(scale)
        return scale

    def forward(self, x):
        x = self._scale(x) * x
        return x