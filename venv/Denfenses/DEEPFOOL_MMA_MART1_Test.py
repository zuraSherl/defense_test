import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from src.train_mnist import MMA_MNIST_Training_Parameters
from src.train_cifar10 import MMA_CIFAR10_Training_Parameters
from src.get_datasets import get_mnist_train_validate_loader
from src.get_datasets import get_cifar10_train_validate_loader

from Denfenses.DefenseMethods.DEEPFOOL_MMA_MART1 import DEEPFOOLMMAMART1Defense
from Denfenses.DefenseMethods.DEEPFOOL_MMA_MART1 import add_indexes_to_loader

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
        training_parameters = MMA_MNIST_Training_Parameters
        model_framework = MNIST_CNN().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=batch_size, valid_size=0.1,
                                                                     shuffle=True)
    else:
        training_parameters = MMA_CIFAR10_Training_Parameters
        model_framework = ResNet18().to(device)
        batch_size = training_parameters['batch_size']
        train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/', batch_size=batch_size, valid_size=0.1,
                                                                       shuffle=True)
    # 对训练集/验证集的每个分组进行编号
    add_indexes_to_loader(train_loader)
    add_indexes_to_loader(valid_loader)

    # defense_name：防御名称
    # nat_params：对抗训练的参数
    defense_name = 'DEEPFOOL_MMA_MART1'
    deepfoolmmamart1_params = {
        'nb_iter': args.nb_iter,
        'max_iters': args.max_iters,
        'overshoot': args.overshoot,
        'test_eps': args.test_eps,
        'test_eps_iter': args.test_eps_iter,
        'clean_loss_fn': args.clean_loss_fn,
        'attack_loss_fn': args.attack_loss_fn,
        'hinge_maxeps': args.hinge_maxeps,
        'lamda': args.lamda,
        'disp_interval': args.disp_interval
    }

    # 将参数传入DEEPFOOLMMAMART1防御中
    deepfoolmmamart1 = DEEPFOOLMMAMART1Defense(loader=train_loader, dataname="train", verbose=True, model=model_framework, defense_name=defense_name, dataset=dataset, training_parameters=training_parameters, device=device,
                     **deepfoolmmamart1_params)
    # 进行对抗训练
    deepfoolmmamart1.test_defense(validation_loader=valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The DEEPFOOLMMAMART1 Defenses')
    # dataset：数据集名称MNIST/CIFAR10
    # gpu_index：gpu数量，默认为0
    # seed：随机种子，默认为100
    # **********选择不同数据集MNIST/CIFAR10**********
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    # **********根据数据集不同进行参数选择**********
    parser.add_argument('--nb_iter', type=int, default=40, help='the iterations of PGD attacks')
    parser.add_argument('--max_iters', type=float, default=50, help='the max iterations of DeepFool attacks')
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot of DeepFool attacks')
    parser.add_argument('--test_eps', type=float, default=0.3, help='the init perturbation of PGD attacks test')
    parser.add_argument('--test_eps_iter', type=float, default=0.01, help='the parameter for calculating the step length of PGD attacks test')
    # **********自由选择，与数据集无关**********
    parser.add_argument('--clean_loss_fn', type=str, default='xent', help='the loss function')
    parser.add_argument('--attack_loss_fn', type=str, default='slm', help='the loss function')
    parser.add_argument('--hinge_maxeps', type=float, help='the max ANPGD perturbation')
    parser.add_argument('--lamda', type=float, default=6.0, help='the regularization coefficient')
    parser.add_argument('--disp_interval', type=int, default=100, help='the output node')
    arguments = parser.parse_args()
    main(arguments)
