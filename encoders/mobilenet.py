"""
see paper: https://arxiv.org/pdf/1704.04861, https://arxiv.org/pdf/1905.02244
see codes: https://github.com/GOATmessi8/RFBNet/blob/master/models/RFB_Net_mobile.py
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
"""

import torch
import torch.nn as nn
from blocks.convs import ConvNormAct
from blocks.attns import SqueezeExcitation


class SepConv(nn.Module):
    """ DWConv => PWConv """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 conv_layer=ConvNormAct):
        super().__init__()
        self.layer = nn.Sequential(
            conv_layer(c_in, c_in , k=k, s=s, p=p, d=d, g=c_in),  # DWConv
            conv_layer(c_in, c_out, k=1, s=1, p=0))               # PWConv

    def forward(self, x):
        x = self.layer(x)
        return x


class InvertedResidual(nn.Module):
    """ PWConv => DWConv => PWConv """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 conv_layer=ConvNormAct, attn=SqueezeExcitation):
        super().__init__()
        c_hid = c_in * 4
        self.layer = nn.Sequential(
            conv_layer(c_in , c_hid, k=1, s=1, p=0),               # PW
            conv_layer(c_hid, c_hid, k=k, s=s, p=p, d=d, g=c_in),  # DW
            attn(c_hid, c_hid//4) if attn else nn.Identity(),      # SE
            conv_layer(c_hid, c_out, k=1, s=1, p=0))               # PW
        if (s > 1) or (c_in != c_out):
            self.res = None

    def forward(self, x):
        if self.res:
            x += self.layer(x)
        else:
            x = self.layer(x)
        return x


class InvertedResidualV2(nn.Module):
    """ PWConv => DWConv => PWConv """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 conv_layer=ConvNormAct, attn=SqueezeExcitation):
        super().__init__()
        c_hid = c_in * 4
        self.layer = nn.Sequential(
            conv_layer(c_in , c_hid, k=1, s=1, p=0),                            # PW
            conv_layer(c_hid, c_hid, k=k, s=s, p=p, d=d, g=c_in),               # DW
            attn(c_hid, c_hid//4) if attn else nn.Identity(),                   # SE
            conv_layer(c_hid, c_out, k=1, s=1, p=0, ACT=nn.Identity))           # PW
        if (s > 1) or (c_in != c_out):
            self.res = conv_layer(c_in, c_hid, k=1, s=1, p=0, ACT=nn.Identity)  # PW
        else:
            self.res = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.res(x) + self.layer(x)
        x = self.act(x)
        return x


class Block(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1, depth=2,
                 block=SepConv):
        super().__init__()
        self.layer = nn.Sequential(
            block(c_in, c_out, k, s, p, d, g),
            *[block(c_out, c_out) for _ in range(1, depth)])

    def forward(self, x):
        x = self.layer(x)
        return x