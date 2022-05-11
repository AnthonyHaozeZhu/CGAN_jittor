# -*- coding: UTF-8 -*-
"""
@Project ：计图 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/5/11 09:26
"""
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

import argparse


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(10, 10)
        self.linear1 = nn.Linear((opt.latent_dim + opt.n_classes), 128)
        self.linear2 = nn.Linear(128, 256)
        self.batch_normal1 = nn.BatchNorm1d(256, 0.8)
        self.linear3 = nn.Linear(256, 512)
        self.batch_normal2 = nn.BatchNorm1d(512, 0.8)
        self.linear4 = nn.Linear(512, 1024)
        self.batch_normal3 = nn.BatchNorm1d(1024, 0.8)
        self.linear5 = nn.Linear(1024, int(opt.channels * opt.img_size * opt.img_size))
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, noise, lab):
        x = torch.concat((self.embedding(lab), noise), dim=1)
        x = self.linear1(x)
        x = self.linear2(self.relu(x))
        x = self.linear3(self.relu(self.batch_normal1(x)))
        x = self.linear4(self.relu(self.batch_normal2(x)))
        x = self.linear5(self.relu(self.batch_normal3(x)))
        return x.view((opt.channels, opt.channels, opt.img_size, opt.img_size))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10, 10)
        self.linear1 = nn.Linear((opt.n_classes + int(opt.channels * opt.img_size * opt.img_size)), 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 10)
        self.drop_out = nn.Dropout(0.4)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, img, lab):
        x = torch.concat((img.view((img.shape[0], (- 1))), self.label_embedding(lab)), dim=1)



if __name__ == "__main__":
    # 设置超参数 batch_size，其值代表一个批次中含有多少个数据。
    # batch_size = 64
    #
    # # 创建 MNIST 训练集数据加载器
    # train_loader = MNIST(train=True, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=True)
    # # 创建 MNIST 测试集数据加载器
    # val_loader = MNIST(train=False, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
    opt = parser.parse_args()
    print(opt)

    number = '18713012939'
    z = torch.tensor(np.random.normal(0, 1, (len(number), opt.latent_dim)))
    labels = torch.tensor(np.array([int(number[num]) for num in range(len(number))]))
    # gen_imgs = generator(z, labels)