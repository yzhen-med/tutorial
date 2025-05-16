from encoders.vgg import *


if __name__ == '__main__':
    in_channel = 3
    in_out = [[32, 64], [64, 128], [128, 256], [256, 512]]
    depths = [2, 2, 2, 4]
    
    net = Encoder(in_channel, in_out, depths)
    x   = torch.randn(2, 3, 512, 512)
    zs  = net(x)
    for z in zs:
        print(z.shape)