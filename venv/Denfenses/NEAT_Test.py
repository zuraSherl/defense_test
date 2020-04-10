import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
# ****************引入用于对抗训练的网络****************
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from src.train_mnist import MNIST_Training_Parameters
from src.train_cifar10 import CIFAR10_Training_Parameters
from src.get_datasets import get_mnist_train_validate_loader, get_mnist_test_loader
from src.get_datasets import get_cifar10_train_validate_loader, get_cifar10_test_loader
from Denfenses.DefenseMethods.NEAT import NEATDefense

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

    # dataset：将数据集名称转化为大写形式MNIST/CIFAR10
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    # 获取MNIST/CIFAR10的模型训练参数，模型，训练集，验证集和测试集
    if dataset == 'MNIST':
        training_parameters = MNIST_Training_Parameters
        model_framework = MNISTConvNet().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
        test_loader = get_mnist_test_loader(dir_name='../data/MNIST/', batch_size=batch_size)
    else:
        training_parameters = CIFAR10_Training_Parameters
        model_framework = resnet20_cifar().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       shuffle=True)
        test_loader = get_cifar10_test_loader(dir_name='../data/CIFAR10/', batch_size=batch_size)
    # defense_name：防御名称为EAT
    defense_name = 'NEAT'
    # eat_params：防御参数
    neat_params = {
        'eps': args.eps,
        'alpha': args.alpha
    }
    # 将各种参数传入到NEAT防御中
    neat = NEATDefense(model=model_framework, defense_name=defense_name, dataset=dataset, training_parameters=training_parameters, device=device,
                     **neat_params)
    # 如果为True则需要重新训练预定义模型
    if args.train_externals:
        print('\nStart to train the external models ......\n')
        neat.train_external_model_group(train_loader=train_loader, validation_loader=valid_loader)
    # 加载预训练模型
    pre_train_models = neat.load_external_model_group(model_dir='../DefenseEnhancedModels/NEAT/', test_loader=test_loader)
    # 进行集成对抗训练，将最佳的模型参数保存到DefenseEnhancedModels/NEAT/MNIST_NEAT_enhanced.pt中或者CIFAR10
    neat.defense(pre_trained_models=pre_train_models, train_loader=train_loader, validation_loader=valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The NEAT Defenses')
    # dataset：数据集为MNIST/CIFAR10
    # gpu_index：gpu数量，默认为0
    # seed：随机种子数，默认为100
    # train_externals：如果为True则需要重新训练预定义的模型，为False时不需要重新训练
    # eps：随机扰动和FGSM扰动的总和
    # alpha：随机扰动的权重
    # ******************数据集可选MNIST/CIFAR10******************
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    # ******************预训练模型是否已经经过训练******************
    parser.add_argument("--train_externals", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='True if required to train the external models additionally, otherwise False')
    # ******************根据数据集不同进行选择******************
    parser.add_argument("--eps", type=float, default=0.3, help="FGSM attack scale")
    # ******************根据数据集不同进行选择******************
    parser.add_argument("--alpha", type=float, default=0.05, help="RFGSM random perturbation scale")
    arguments = parser.parse_args()
    main(arguments)
