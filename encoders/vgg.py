""" VGG: N * (Conv => ReLU) => MaxPool
see paper: https://arxiv.org/pdf/1409.1556v6
see codes: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb """

import torch.nn as nn
from blocks.convs import ConvAct


class Block(nn.Module):
    """ N * Conv """
    def __init__(self, c_in, c_out, k=3, s=1, p=1, d=1, g=1, depth=2,
                 block=ConvAct):
        super().__init__()
        self.layer = nn.Sequential(
            block(c_in, c_out, k, s, p, d, g),
            *[block(c_out, c_out) for _ in range(1, depth)])

    def forward(self, x):
        x = self.layer(x)
        return x