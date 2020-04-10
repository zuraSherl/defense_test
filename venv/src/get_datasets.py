import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision.transforms import  ToPILImage

# 下载和划分MNIST的训练集和验证集，然后分组
def get_mnist_train_validate_loader(dir_name, batch_size, valid_size=0.1, shuffle=True, random_seed=100, num_workers=1):
    # dir_name：下载数据后保存位置的路径
    # valid_size：划分训练集的比例
    # shuffle：Ture进行顺序打乱，否则不进行顺序打乱
    # random_seed：随机种子的参数值
    # batch_size：划分组的大小
    # num_workers：用于数据加载的子进程数

    assert 0.0 <= valid_size <= 1.0, 'the size of validation set should be in the range of [0, 1]'
    # 下载MNIST训练数据，root为下载下来保存的路径名称，transform对图像进行变换
    train_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=True, transform=transforms.ToTensor(), download=True)
    valid_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=True, transform=transforms.ToTensor(), download=True)
    # 训练集的总数
    num_train = len(train_mnist_dataset)
    # 产生0，1，2，3，...，num_train-1的list
    indices = list(range(num_train))
    # 获取训练集的一部分1/10，向下取整
    split = int(np.floor(valid_size * num_train))
    # 如果shuffle是True的话，将indices的顺序打乱
    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # 将indices划分为以spilit为界限的两部分，训练集和验证集
    train_idx, valid_idx = indices[split:], indices[:split]
    # 将训练集和验证集提取出来
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # 对训练集和验证集进行分组
    train_loader = torch.utils.data.DataLoader(train_mnist_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_mnist_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    return train_loader, valid_loader

# 下载MNIST测试集并分组
def get_mnist_test_loader(dir_name, batch_size, shuffle=False, num_worker=1):
    # dir_name：下载数据后保存位置的路径
    # batch_size：划分组的大小
    # num_workers：用于数据加载的子进程数
    # shuffle：默认值为False
    # 下载MNIST测试数据，root为下载下来保存的路径名称，transform对图像进行变换
    test_mnist_dataset = torchvision.datasets.MNIST(root=dir_name, train=False, download=True, transform=transforms.ToTensor())
    # 对测试集进行分组
    test_loader = torch.utils.data.DataLoader(dataset=test_mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return test_loader

# 下载和划分CIFAR10训练集和验证集，然后分组
def get_cifar10_train_validate_loader(dir_name, batch_size, valid_size=0.1, augment=True, shuffle=True, random_seed=100, num_workers=1):
    # dir_name：下载数据后保存位置的路径
    # valid_size：划分训练集的比例
    # shuffle：Ture进行顺序打乱，否则不进行顺序打乱
    # random_seed：随机种子的参数值
    # batch_size：划分组的大小
    # num_workers：用于数据加载的子进程数
    # augment：为Ture时对图像进行多种变换

    # transforms.RandomAffine()：仿射变换
    # transforms.RandomHorizontalFlip()：依概率水平翻转，默认值为0.5
    # transforms.ToTensor()：转为tensor
    # 训练集数据增强
    if augment is True:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
    # 验证集数据增强
    valid_transform = transforms.Compose([transforms.ToTensor()])
    # 下载CIFAR10训练数据，root为下载下来保存的路径名称，transform对图像进行变换
    train_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=True, download=True, transform=train_transform)
    valid_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=True, download=True, transform=valid_transform)
    # 训练集的总数
    num_train = len(train_cifar10_dataset)
    # 产生0，1，2，3，...，num_train-1的list
    indices = list(range(num_train))
    # 获取训练集的一部分1/10，向下取整
    split = int(np.floor(valid_size * num_train))
    # 如果shuffle是True的话，将indices的顺序打乱
    if shuffle is True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # 将indices划分为以spilit为界限的两部分，训练集和验证集
    train_idx, valid_idx = indices[split:], indices[:split]
    # 将训练集和验证集提取出来
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # 对训练集和验证集进行分组
    train_loader = torch.utils.data.DataLoader(train_cifar10_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_cifar10_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    return train_loader, valid_loader

# 下载CIFAR10测试集并分组
def get_cifar10_test_loader(dir_name, batch_size, shuffle=False, num_worker=1):
    # dir_name：下载数据后保存位置的路径
    # batch_size：划分组的大小
    # num_workers：用于数据加载的子进程数
    # shuffle：默认值为False
    # 下载CIFAR测试数据，root为下载下来保存的路径名称，transform对图像进行变换
    test_cifar10_dataset = torchvision.datasets.CIFAR10(root=dir_name, train=False, download=True, transform=transforms.ToTensor())
    # 对测试集进行分组
    test_loader = torch.utils.data.DataLoader(dataset=test_cifar10_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return test_loader