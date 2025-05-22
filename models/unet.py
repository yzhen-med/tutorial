"""
see paper: https://arxiv.org/pdf/1505.04597v1
see codes: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from blocks.convs import ConvNormAct, ConvAct
from encoders.vgg import Block as vgg_block


# ==== Settings ==== #
CONV = nn.Conv2d
NORM = nn.BatchNorm2d
ACT  = nn.ReLU
ConvAct = partial(ConvAct, CONV=CONV, ACT=ACT)
ConvNormAct = partial(ConvNormAct, CONV=CONV, NORM=NORM, ACT=ACT)
vgg_block = partial(vgg_block, conv_layer=ConvNormAct)
# ================== #


class InConv(ConvAct):
    """ Conv => Act """
    def forward(self, img, multi_scale_imgs=[]):
        zs = [self.layer(img)]
        if len(multi_scale_imgs):
            for v in multi_scale_imgs:
                zs.append(self.layer(v))
        return zs


class Encoder(nn.Module):
    def __init__(self, in_channel, in_out, depths):
        super().__init__()
        self.in_conv = self.build_in_conv(in_channel, in_out[0][0])
        self.enc_layers = nn.ModuleList()
        for (c_in, c_out), depth in zip(in_out, depths):
            self.enc_layers.append(nn.Sequential(
                CONV(c_in, c_in, kernel_size=2, stride=2, padding=0),  # downsample
                vgg_block(c_in, c_out, depth=depth)))

    def build_in_conv(self, c_in, c_out):
        return InConv(c_in, c_out)

    def forward(self, x):
        enc_zs = self.in_conv(x)
        for layer in self.enc_layers:
            x = layer(enc_zs[-1])
            enc_zs.append(x)
        return enc_zs


class EncoderMultiScale(Encoder):
    def forward(self, x, multi_scale_imgs):
        enc_zs = self.in_conv(x, multi_scale_imgs)[::-1]
        N = len(multi_scale_imgs)
        for i, layer in enumerate(self.enc_layers):
            x = enc_zs[-1]
            if 0 < i <= N:
                x = torch.cat([x, enc_zs[N - i]], dim=1)
            x = layer(x)
            enc_zs.append(x)
        return enc_zs


class Decoder(nn.Module):
    def __init__(self, in_out, depths):
        super().__init__()
        self.dec_layers = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        for (c_in, c_out), depth in zip(in_out, depths):
            self.dec_layers.append(nn.Sequential(
                vgg_block(c_in, c_out, depth=depth)))

    def forward(self, enc_zs):
        dec_zs = []
        z = enc_zs.pop()
        for i, layer in enumerate(self.dec_layers):
            z = torch.cat([self.up(z), enc_zs.pop()], dim=1)
            z = layer(z)
            dec_zs.append(z)
        return dec_zs