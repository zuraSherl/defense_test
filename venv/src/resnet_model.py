import torch
import torch.nn as nn
import torch.nn.functional as F
from src import activation_function
from src.basic_model import BasicModule

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

# ResNet的网络模块（激活函数可选）
class SparseBasicBlock(BasicModule):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=False):
        super(SparseBasicBlock, self).__init__()
        # sparsity：稀疏比，默认为0.5，其他激活函数的参数
        # use_relu：是否使用ReLU激活函数，默认为True
        # sparse_func：自定义的激活函数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        # 是否使用ReLU激活函数
        self.use_relu = use_relu
        # 其他激活函数
        self.sparse1 = activation_function.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = activation_function.sparse_func_dict[sparse_func](sparsity)
        # ReLU函数
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # 如果使用ReLU函数，那么先经过ReLU激活再经过其他激活函数
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # 如果使用ReLU函数，那么先经过ReLU激活再经过其他激活函数
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)
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

# ResNet的网络模块（激活函数可选）
class SparseBottleneck(BasicModule):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, use_relu=True, sparse_func='reg', bias=True):
        super(SparseBottleneck, self).__init__()
        # use_relu：是否使用ReLU激活函数，默认为True
        # sparsity：稀疏比，默认为0.5，其他激活函数的参数
        # sparse_func：其他激活函数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 其他激活函数
        self.sparse1 = activation_function.sparse_func_dict[sparse_func](sparsity)
        self.sparse2 = activation_function.sparse_func_dict[sparse_func](sparsity)
        self.sparse3 = activation_function.sparse_func_dict[sparse_func](sparsity)
        # 是否使用ReLU激活函数
        self.use_relu = use_relu
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # 如果使用ReLU激活函数，则先经过ReLU激活，再经过其他激活函数
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse1(out)
        out = self.bn2(self.conv2(out))
        # 如果使用ReLU激活函数，则先经过ReLU激活，再经过其他激活函数
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        # 如果使用ReLU激活函数，则先经过ReLU激活，再经过其他激活函数
        if self.use_relu:
            out = self.relu(out)
        out = self.sparse3(out)
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

# 定义ResNet网络（激活函数可选）
class SparseResNet(BasicModule):
    def __init__(self, block, num_blocks, sparsities, num_classes=10, use_relu=True, sparse_func='reg', bias=True):
        super(SparseResNet, self).__init__()
        # sparsities：激活函数的参数数组
        # use_relu：是否使用ReLU函数，默认为True
        # sparse_func：其他激活函数
        self.in_planes = 64
        # 是否使用ReLU函数
        self.use_relu = use_relu
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        # 传入其他激活函数和对应参数
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, sparsity=sparsities[0],
                                       sparse_func=sparse_func, bias=bias)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, sparsity=sparsities[1],
                                       sparse_func=sparse_func, bias=bias)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, sparsity=sparsities[2],
                                       sparse_func=sparse_func, bias=bias)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, sparsity=sparsities[3],
                                       sparse_func=sparse_func, bias=bias)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.relu = nn.ReLU()
        self.activation = {}
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.cpu().detach()
        return hook
    def register_layer(self, layer, name):
        layer.register_forward_hook(self.get_activation(name))
    def _make_layer(self, block, planes, num_blocks, stride, sparsity=0.5, sparse_func='reg', bias=True):
        # sparsity：其他激活函数的参数
        # sparse_func：其他激活函数
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, sparsity, self.use_relu, sparse_func=sparse_func, bias=bias))
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

# 定义ResNet18网络
# c1=32*32*3-32*32*64-BatchNorm2d-ReLU-32*32*64-BatchNorm2d-ReLU-32*32*64-BatchNorm2d
# r1=32*32*3-32*32*64-BatchNorm2d
# r2=c1+r1-ReLU
# c2=32*32*64-BatchNorm2d-ReLU-32*32*64-BatchNorm2d
# c2+r2-ReLU
# c3=16*16*128-BatchNorm2d-ReLU-16*16*128-BatchNorm2d
# r3=c2+r2-ReLU-16*16*128-BatchNorm2d
# r4=c3+r3-ReLU
# c4=16*16*128-BatchNorm2d-ReLU-16*16*128-BatchNorm2d
# c4+r4-ReLU
# c5=8*8*256-BatchNorm2d-ReLU-8*8*256-BatchNorm2d
# r5=c4+r4-ReLU-8*8*256-BatchNorm2d
# r6=c5+r5-ReLU
# c6=8*8*256-BatchNorm2d-ReLU-8*8*256-BatchNorm2d
# c6+r6-ReLU
# c7=4*4*512-BatchNorm2d-ReLU-4*4*512-BatchNorm2d
# r7=c6+r6-ReLU-4*4*512-BatchNorm2d
# r8=c7+r7-ReLU
# c8=4*4*512-BatchNorm2d-ReLU-4*4*512-BatchNorm2d
# c=c8+r8-ReLU
# c-maxpool-1*1*512-1*512-1*10-ReLU
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# 定义ResNet34网络
def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

# 定义ResNet50网络
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

# 定义ResNet101网络
def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

# 定义ResNet152网络
def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

# 定义ResNet18网络（激活函数可选）
def SparseResNet18(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, [2,2,2,2], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# 定义ResNet34网络（激活函数可选）
def SparseResNet34(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBasicBlock, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# 定义ResNet50网络（激活函数可选）
def SparseResNet50(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3,4,6,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# 定义ResNet101网络（激活函数可选）
def SparseResNet101(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3,4,23,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)

# 定义ResNet152网络（激活函数可选）
def SparseResNet152(relu=False, sparsities=[0.5,0.4,0.3,0.2], sparse_func='reg', bias=False):
    return SparseResNet(SparseBottleneck, [3,8,36,3], sparsities, use_relu=relu, sparse_func=sparse_func, bias=bias)
