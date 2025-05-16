""" Accelerator """

import torch
import torch.nn as nn
from torch.optim import Adam
from accelerate import Accelerator

# We could avoid x.to(accelerator.device) if we set the accelerator with `device_placement=True`.
def train(net, dataloader, args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    net = net.to(accelerator.device)
    opt = Adam(params=net.parameters(), lr=args.lr)

    net, opt = accelerator.prepare(net, opt)
    for epoch in range(args.epochs):
        net.train()
        for j, (x, y) in enumerate(dataloader):
            x, y = x.to(accelerator.device), y.to(accelerator.device)
            y_pred = net(x)
            loss = nn.MSELoss(y_pred, y)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
    accelerator.end_training()