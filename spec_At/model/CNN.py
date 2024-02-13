import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from os.path import isdir, isfile


class Flatten(nn.Module):

    def forward(self, input):
        # input:[batch, channel, H, W]
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.cnn2 = nn.Conv2d(10, 20, 3, 1)
        self.batch_norm2 = nn.BatchNorm2d(20)
        self.flatten = Flatten()
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3, stride=1),
        #     nn.BatchNorm2d(10), nn.LeakyReLU(), nn.Conv2d(10, 20, 3, 1),
        #     nn.BatchNorm2d(20), nn.LeakyReLU(), Flatten())
        # self.fc = nn.Sequential(nn.Linear(10240, 64), nn.Linear(64, 2), nn.Softmax())
        self.fc = nn.Sequential(nn.Linear(10240, 64), nn.Linear(64, 2))

    def forward_once(self, x):
        # output = self.cnn(x)
        output = self.cnn1(x)
        output = self.batch_norm1(output)
        output = F.leaky_relu(output)
        output = self.cnn2(output)
        output = self.batch_norm2(output)
        output = F.leaky_relu(output)
        output = self.flatten(output)
        return output

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        return output
