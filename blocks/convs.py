""" 卷积模块 """

import torch
import torch.nn as nn


class ConvNorm(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 CONV=nn.Conv2d, NORM=nn.BatchNorm2d):
        super().__init__()
        self.layer = nn.Sequential(
            CONV(c_in, c_out, k, s, p, d, g),
            NORM(c_out))

    def forward(self, x):
        return self.layer(x)


class ConvAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 CONV=nn.Conv2d, ACT=nn.ReLU):
        super().__init__()
        self.layer = nn.Sequential(
            CONV(c_in, c_out, k, s, p, d, g),
            ACT())

    def forward(self, x):
        return self.layer(x)


class ConvNormAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 CONV=nn.Conv2d, NORM=nn.BatchNorm2d, ACT=nn.ReLU):
        super().__init__()
        self.layer = nn.Sequential(
            CONV(c_in, c_out, k, s, p, d, g),
            NORM(c_out),
            ACT())

    def forward(self, x):
        return self.layer(x)