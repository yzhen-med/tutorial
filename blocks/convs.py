""" 卷积模块 """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1,
                 CONV=nn.Conv2d, NORM=None, ACT=None):
        super().__init__()
        layers = [CONV(c_in, c_out, k, s, p, d, g)]
        if NORM is not None:
            layers.append(NORM(c_out))
        if ACT is not None:
            layers.append(ACT())
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ConvNorm(ConvBlock):
    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(c_in, c_out, NORM=nn.BatchNorm2d, **kwargs)


class ConvAct(ConvBlock):
    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(c_in, c_out, ACT=nn.ReLU, **kwargs)


class ConvNormAct(ConvBlock):
    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(c_in, c_out, NORM=nn.BatchNorm2d, ACT=nn.ReLU, **kwargs)


class ASPP(nn.Module):
    """ Features with multiple fields-of-view """
    def __init__(self, c_in, c_out, dilations=[1, 6, 12, 18],
                 conv_layer=ConvNormAct):
        super().__init__()
        self.aspp_layers = nn.ModuleList([])
        self.aspp_layers.append(conv_layer(c_in, c_out, k=1, s=1, p=0, d=dilations[0]))
        for d in dilations[1:]:
            self.aspp_layers.append(conv_layer(c_in, c_out, k=3, s=1, p=d, d=d))
        self.aspp_layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), conv_layer(c_in, c_out, k=1, s=1, p=0)))
        
        self.out_layer = nn.Sequential(
            conv_layer(5*c_out, c_out, k=1, s=1, p=0), nn.Dropout(0.5))

    def forward(self, x):
        zs = [ff(x) for ff in self.aspp_layers]
        zs[-1] = F.interpolate(zs[-1], size=zs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat(zs, dim=1)
        return self.out_layer(x)


class RecurrentConv(nn.Module):
    """
    循环卷积: (x + x_t) => Conv => (x + x_t+1)
    """
    def __init__(self, c_out, k=3, s=1, p=1, d=1, g=1, t=2,
                 conv_layer=ConvNormAct):
        super().__init__()
        self.t = t
        self.conv = conv_layer(c_out, c_out, k, s, p, d, g)

    def forward(self, x):
        xt = self.conv(x)
        for i in range(self.t):
            xt = self.conv(x + xt)
        return xt