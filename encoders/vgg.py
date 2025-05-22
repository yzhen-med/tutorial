""" VGG: N * (Conv => ReLU) => MaxPool
see paper: 
see codes: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb """

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from blocks.convs import ConvNormAct, ConvAct
from functools import partial


class Block(nn.Module):
    """ N * Conv """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1, depth=2,
                 conv_layer=ConvAct):
        super().__init__()
        self.layer = nn.Sequential(
            conv_layer(c_in, c_out, k, s, p, d, g),
            *[conv_layer(c_out, c_out) for _ in range(1, depth)])

    def forward(self, x):
        x = self.layer(x)
        return x