import torch
import torch.nn as nn
from src import activation_function
from src.basic_model import BasicModule
import math

# 返回一维结构
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# 定义MNIST_CNN模型（激活函数为ReLU）
# 28*28*1——28*28*num_channels[0]——ReLU——14*14*num_channels[1]——ReLU——14*14*num_channels[2]——
# ReLU——7*7*num_channels[3]——ReLU——(7*7*num_channels[3])*1——hidden_size*1——ReLU——10*1
class MNIST_CNN(BasicModule):
    def __init__(self, num_channels=[32,32,64,64], hidden_size=20000):
        super(MNIST_CNN, self).__init__()
        # num_channels：通道数
        # hidden_size：隐藏层神经元数量，默认为20000
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1), nn.ReLU(),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1), nn.ReLU(),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), nn.ReLU(),
                          nn.Linear(hidden_size, 10))
    # 前向传播
    def forward(self, x):
        return self.cnn(x)

# MNIST_CNN模型（激活函数可选，可带偏差）
# 28*28*1——28*28*num_channels[0]——激活函数——14*14*num_channels[1]——激活函数——
# 14*14*num_channels[2]——激活函数——7*7*num_channels[3]——激活函数——(7*7*num_channels[3])*1——
# hidden_size*1——激活函数——10*1
class SparseMNIST_CNN(BasicModule):
    def __init__(self, sp1, sp2, func, num_channels, hidden_size=20000, bias=True):
        super(SparseMNIST_CNN, self).__init__()
        # sp1：sp1稀疏比；也可以为k值
        # sp2：sp2稀疏比
        # func：k-WTA/ReLU激活函数
        # num_channels：每一层的通道数
        # hidden_size：隐藏层神经元数量，默认为20000
        # bias：是否求偏差，默认为True
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1, bias=bias), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2, bias=bias), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1, bias=bias), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2, bias=bias), activation_function.sparse_func_dict[func](sp1),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), activation_function.Sparsify1D(sp2),
                          # nn.Linear(hidden_size, hidden_size), activation_function.Sparsify1D(sp2),
                          nn.Linear(hidden_size, 10))
    # 前向传播
    def forward(self, x):
        return self.cnn(x)

# 带BatchNorm2D的MNIST_CNN模型（激活函数可选，可带偏差）
# 28*28*1——28*28*num_channels[0]——BatchNorm2D——激活函数——14*14*num_channels[1]——BatchNorm2D——激活函数
# ——14*14*num_channels[2]——BatchNorm2D——激活函数——7*7*num_channels[3]——BatchNorm2D——激活函数——
# (7*7*num_channels[3])*1——hidden_size*1——激活函数——10*1
class SparseMNIST_CNN_BN(BasicModule):
    def __init__(self, sp1, sp2, func, num_channels, hidden_size=20000, bias=True):
        super(SparseMNIST_CNN_BN, self).__init__()
        # sp1：sp1稀疏比；也可以为k值
        # sp2：sp2稀疏比
        # func：k-WTA/ReLU激活函数
        # num_channels：每一层的通道数
        # hidden_size：隐藏层神经元数量，默认为20000
        # bias：是否求偏差，默认为True
        self.cnn = nn.Sequential(nn.Conv2d(1, num_channels[0], 3, padding=1, bias=bias), nn.BatchNorm2D(channels[0]), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1, stride=2, bias=bias), nn.BatchNorm2D(channels[1]), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1, bias=bias), nn.BatchNorm2D(channels[2]), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1, stride=2, bias=bias), nn.BatchNorm2D(channels[3]), activation_function.sparse_func_dict[func](sp1),
                          Flatten(),
                          nn.Linear(7*7*num_channels[3], hidden_size), activation_function.Sparsify1D(sp2),
                          # nn.Linear(hidden_size, hidden_size), activation_function.Sparsify1D(sp),
                          nn.Linear(hidden_size, 10))
    # 前向传播
    def forward(self, x):
        return self.cnn(x)

# MNIST_CNN模型（一些层激活函数可选，一些层为ReLU激活函数）
# 28*28*1——28*28*32——激活函数——14*14*32——ReLU——14*14*64——ReLU——
# 7*7*64——ReLU——(7*7*64)*1——hidden_size*1——ReLU——10*1
class PartialMNIST_CNN(BasicModule):
    def __init__(self, sp, hidden_size=20000):
        super(PartialMNIST_CNN, self).__init__()
        # sp：稀疏比；也可以为k值
        # hidden_size：隐藏层神经元数量，默认为20000
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), activation_function.sparse_func_dict[func](sp1),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, hidden_size), nn.ReLU(),
                          nn.Linear(hidden_size, 10))
    # 前向传播
    def forward(self, x):
        return self.cnn(x)