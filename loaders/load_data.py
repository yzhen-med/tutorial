""" 读取数据 """

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_cifar10(root='/data0/yzhen/data/cifar10', transform=None, bsz=64, num_workers=2):
    valid_train = [datasets.CIFAR10(root=root, train=j, transform=transform)
                   for j in range(2)]
    
    valid_train = [DataLoader(v, batch_size=bsz, shuffle=j, num_workers=num_workers)
                   for j,v in enumerate(valid_train)]
    return valid_train


if __name__ == '__main__':
    root_cifar10 = '/data0/yzhen/data/cifar10'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_train = load_cifar10(root_cifar10, transform)
    for img, lbl in valid_train[0]:
        print(img.shape)
        print(lbl)
        plt.imshow(img[0].permute(1,2,0))
        plt.show()
        break