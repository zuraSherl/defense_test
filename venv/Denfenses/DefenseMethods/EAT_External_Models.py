import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.basic_model import BasicModule

# 返回一维结构
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# MNIST_A模型
class MNIST_A(BasicModule):
    def __init__(self, num_channels=[32,32,64,64], hidden_size=20000):
        super(MNIST_A, self).__init__()
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

# MNIST_B模型
class MNIST_B(BasicModule):
    def __init__(self, num_channels=[64,64,128,128], hidden_size=20000):
        super(MNIST_B, self).__init__()
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

# MNIST_C模型
class MNIST_C(BasicModule):
    def __init__(self, num_channels=[128,128,256,256], hidden_size=20000):
        super(MNIST_C, self).__init__()
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

# MNIST_D模型
class MNIST_D(BasicModule):
    def __init__(self, hidden_size=20000):
        super(MNIST_D, self).__init__()
        # hidden_size：隐藏层神经元数量，默认为20000
        # (28*28*1)*1——hidden_size*1——ReLU——hidden_size*1——ReLU——10*1
        self.nn = nn.Sequential(Flatten(), nn.Linear(28*28, hidden_size), nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                nn.Linear(hidden_size, 10))
        # 遍历网络中的每个模块
        for m in self.modules():
            # 如果是线性模块
            if isinstance(m, nn.Linear):
                # n：权重数量
                n = m.weight.data.shape[1]
                # 初始化权重
                m.weight.data.normal_(0, math.sqrt(1. / n))
                # 初始化偏差
                m.bias.data.normal_(0, 1)
    # 前向传播
    def forward(self, x):
        return self.nn(x)

# ResNet网络的模块
class BasicBlock(BasicModule):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet网络的模块
class Bottleneck(BasicModule):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, bias=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义ResNet网络
class ResNet(BasicModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.relu = nn.ReLU()
        self.activation = {}
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook
    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet18
def CIFAR10_A():
    return ResNet(BasicBlock, [2, 2, 2, 2])
# ResNet50
def CIFAR10_B():
    return ResNet(Bottleneck, [3, 4, 6, 3])
# ResNet101
def CIFAR10_C():
    return ResNet(Bottleneck, [3, 4, 23, 3])
# ResNet152
def CIFAR10_D():
    return ResNet(Bottleneck, [3, 8, 36, 3])












