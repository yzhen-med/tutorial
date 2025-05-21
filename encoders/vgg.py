""" VGG """

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from blocks.convs import ConvNormAct, ConvAct
from functools import partial


# ==== Settings ==== #
CONV = nn.Conv2d
NORM = nn.BatchNorm2d
ACT  = nn.ReLU
ConvAct = partial(ConvAct, CONV=CONV, ACT=ACT)
ConvNormAct = partial(ConvNormAct, CONV=CONV, NORM=NORM, ACT=ACT)
# ================== #


class InConv(nn.Module):
    """ Conv => Act """
    def __init__(self, c_in, c_out, k=7, s=1, p=3, d=1, g=1):
        super().__init__()
        self.conv = ConvAct(c_in, c_out, k, s, p, d, g)

    def forward(self, img):
        zs = [self.conv(img)]
        return zs


class InConvMultiScale(InConv):
    """ 多尺度图像输入 """
    def forward(self, img, multi_scale_imgs):
        zs = [self.conv(img)]
        for v in multi_scale_imgs:
            zs.append(self.conv(v))
        return zs


class Down(nn.Module):
    """ (Downsample => N*Conv) """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1, depth=2):
        super().__init__()
        self.down  = CONV(c_in, c_in, 2, 2, 0)
        self.layer = nn.Sequential(
            ConvNormAct(c_in, c_out, k, s, p, d, g),
            *[ConvNormAct(c_out, c_out) for _ in range(1, depth)])

    def forward(self, x):
        z = self.down(x)
        z = self.layer(z)
        return z


class Encoder(nn.Module):
    def __init__(self, in_channel, in_out, depths):
        super().__init__()
        self.in_conv = InConv(in_channel, in_out[0][0])
        self.layers = nn.ModuleList()
        for (c_in, c_out), depth in zip(in_out, depths):
            self.layers.append(Down(c_in, c_out, depth=depth))

    def forward(self, img):
        zs = self.in_conv(img)
        for i, layer in enumerate(self.layers):
            z = layer(zs[i])
            zs.append(z)
        return zs


class EncoderMultiScale(nn.Module):
    def __init__(self, in_channel, in_out, depths):
        super().__init__()
        self.in_conv = InConvMultiScale(in_channel, in_out[0][0])
        self.layers = nn.ModuleList()
        for (c_in, c_out), depth in zip(in_out, depths):
            self.layers.append(Down(c_in, c_out, depth=depth))

    def forward(self, img, multi_scale_imgs):
        zs = self.in_conv(img, multi_scale_imgs)
        z = zs[0]
        for i, layer in enumerate(self.layers):
            if (i > 0) and (i <= len(multi_scale_imgs)):
                z = torch.cat([zs[i], zs[-1]], dim=1)
            z = layer(z)
            zs.append(z)
        return zs