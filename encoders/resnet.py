"""
see paper: https://arxiv.org/pdf/1512.03385v1
see codes:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn
from blocks.convs import ConvNormAct


class BasicBlock(nn.Module):
    """ conv3x3 => conv3x3 """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 conv_layer=ConvNormAct):
        super().__init__()
        self.layer = nn.Sequential(
            conv_layer(c_in, c_out, k, s, p, d, g),  # downsample when s>1
            conv_layer(c_out, c_out, ACT=nn.Identity))
        self.act   = nn.ReLU(inplace=True)
        if (s != 1) or (c_in != c_out):
            self.res = conv_layer(c_in, c_out, k=1, s=s, p=0, ACT=nn.Identity)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        x = self.layer(x) + self.res(x)
        x = self.act(x)
        return x


class Bottleneck(BasicBlock):
    """ conv1x1 => conv3x3 => conv1x1 """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 conv_layer=ConvNormAct):
        super().__init__(c_in, c_out, s=s, conv_layer=conv_layer)
        self.layer = nn.Sequential(
            conv_layer(c_in , c_out, k=1, s=1, p=0),
            conv_layer(c_out, c_out, k=k, s=s, p=p, d=d, g=g),
            conv_layer(c_out, c_out, k=1, s=1, p=0, ACT=nn.Identity))


class Block(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1, depth=2,
                 block=BasicBlock):
        super().__init__()
        self.layer = nn.Sequential(
            block(c_in, c_out, k, s, p, d, g),
            *[block(c_out, c_out) for _ in range(1, depth)])

    def forward(self, x):
        x = self.layer(x)
        return x