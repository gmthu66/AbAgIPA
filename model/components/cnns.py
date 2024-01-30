import torch
import torch.nn as nn


class My1DCNN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size=3):
        super(My1DCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        # x 的形状：[batch_size, sequence_length, feature_dim]
        x = x.permute(0, 2, 1)  # 将 l 维度移动到第二个维度
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # 将 l 维度移回到最后一个维度
        return x
