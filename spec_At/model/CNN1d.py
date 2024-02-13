import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self):
        # 初始化网络层
        super(CNN1D, self).__init__()
        self.onedCNN = nn.Conv1d(20, 400, 3, 1, padding=1)
        self.droplayer = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2, 1)
        self.flatten = nn.Flatten()
        self.denselayer = nn.Sequential(nn.Linear(3600, 50), nn.ReLU())
        self.classlayer = nn.Linear(50, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.droplayer(self.onedCNN(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.flatten(x)
        x = self.denselayer(x)
        return self.classlayer(x)


def get_1dCNN_Net(input_size=(10, 20)):    
    # params = [['CONV', 400, 3, 1],
    #         ['DROP', 0.5],
    #         ['POOL', 2, 1],
    #         ['FLAT'],
    #         ['DENSE', 50]]
    # # 创建网络实例
    # cnn_model = CNN1D(input_size, params)
    cnn_model = CNN1D()
    return cnn_model
