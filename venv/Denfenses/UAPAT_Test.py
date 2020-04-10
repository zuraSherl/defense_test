import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from src.train_mnist import MNIST_Training_Parameters
from src.train_cifar10 import CIFAR10_Training_Parameters
from src.get_datasets import get_mnist_train_validate_loader
from src.get_datasets import get_cifar10_train_validate_loader

from Denfenses.DefenseMethods.UAPAT import UAPATDefense

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # dataset：数据集名称MNIST/CIFAR10
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    # 获取训练的超参数，获取模型，划分训练集和验证集
    # training_parameters：训练超参数
    # model_framework：网络
    # train_loader：训练集
    # valid_loader：验证集
    if dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNIST_CNN().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
        uap_train_loader, uap_valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=1, valid_size=0.9,
                                                                     shuffle=True)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = ResNet18().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       shuffle=True)
        uap_train_loader, uap_valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/', batch_size=1, valid_size=0.9,
                                                                       shuffle=True)
    # defense_name：防御名称
    # nat_params：对抗训练的参数
    defense_name = 'UAPAT'
    uapat_params = {
        'fool_rate': args.fool_rate,
        'epsilon': args.epsilon,
        'max_iter_universal': args.max_iter_universal,
        'overshoot': args.overshoot,
        'max_iter_deepfool': args.max_iter_deepfool
    }
    # 将参数传入UAPAT防御中
    uapat = UAPATDefense(model=model_framework, defense_name=defense_name, dataset=dataset, training_parameters=training_parameters, device=device,
                     **uapat_params)
    # 进行对抗训练
    uapat.defense(train_loader=train_loader, validation_loader=valid_loader, uap_train_loader=uap_train_loader, uap_validation_loader=uap_valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The UAPAT Defenses')
    # dataset：数据集名称MNIST/CIFAR10
    # gpu_index：gpu数量，默认为0
    # seed：随机种子，默认为100
    # fool_rate：UAP扰动要达到的错误率
    # epsilon：UAP添加的最大扰动值
    # max_iter_universal：最大的UAP迭代次数
    # overshoot：DeepFool添加的扰动大小
    # max_iter_deepfool：DeepFool的最大迭代次数
    # **********选择不同数据集MNIST/CIFAR10**********
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    # **********根据数据集不同选择**********
    parser.add_argument('--fool_rate', type=float, default=1.0, help='the fooling rate')
    # **********根据数据集不同选择**********
    parser.add_argument('--epsilon', type=float, default=0.1, help='controls the magnitude of the perturbation')
    # **********根据数据集不同选择**********
    parser.add_argument('--max_iter_universal', type=int, default=20, help='the maximum iterations for UAP')
    # **********根据数据集不同选择**********
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot parameter for DeepFool')
    # **********根据数据集不同选择**********
    parser.add_argument('--max_iter_deepfool', type=int, default=10, help='the maximum iterations for DeepFool')
    arguments = parser.parse_args()
    main(arguments)
