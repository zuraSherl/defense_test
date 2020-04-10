import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from src.get_datasets import get_mnist_train_validate_loader, get_mnist_test_loader
from src.get_datasets import get_cifar10_train_validate_loader, get_cifar10_test_loader

from Denfenses.DefenseMethods.NRC import NRCDefense

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

    # dataset：数据集名称转化为大写形式MNIST/CIFAR10
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    # batch_size：每个分组的大小为1000
    batch_size = 1000
    # 获取MNIST/CIFAR10的模型，测试集
    model_location = '{}/{}/model/{}_raw.pt'.format('../data', dataset, dataset)
    if dataset == 'MNIST':
        raw_model = MNIST_CNN().to(device)
        test_loader = get_mnist_test_loader(dir_name='../data/MNIST/', batch_size=batch_size)
    else:
        raw_model = ResNet18().to(device)
        test_loader = get_cifar10_test_loader(dir_name='../data/CIFAR10/', batch_size=batch_size)

    # 加载MNIST/CIFAR10的模型
    raw_model.load(path=model_location, device=device)
    # defense_name：防御名称为NRC
    defense_name = 'NRC'
    # 将参数传入NRC防御中
    nrc = NRCDefense(model=raw_model, defense_name=defense_name, dataset=dataset, device=device, num_points=args.num_points)

    # 如果要进行最优半径的搜索
    if args.search:
        # get the validation dataset (10% with the training dataset)
        print('start to search the radius r using validation dataset ...')
        # 获取MNIST/CIFAR10的验证集
        if dataset == 'MNIST':
            _, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=batch_size, valid_size=0.02,
                                                              shuffle=True)
        else:
            _, valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/', batch_size=batch_size, valid_size=0.02,
                                                                shuffle=True)
        # radius：通过验证集得到最优的半径值
        radius = nrc.search_best_radius(validation_loader=valid_loader, radius_min=args.radius_min, radius_max=args.radius_max,
                                       radius_step=args.radius_step)
    # 否则半径值为默认的0.01
    else:
        radius = round(args.radius, 2)
    print('######\nthe radius for NRC is set or searched as: {}\n######'.format(radius))

    # 计算NRC模型在测试集上的分类精度
    print('\nStart to calculate the accuracy of region-based classification defense on testing dataset')
    raw_model.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            nrc_labels = nrc.region_based_classification(samples=images, radius=radius, mean=args.mean, std=args.std)
            nrc_labels = torch.from_numpy(nrc_labels)
            total += labels.size(0)
            correct += (nrc_labels == labels).sum().item()
        ratio = correct / total
        print('\nTest accuracy of the {} model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%\n'.format(raw_model.model_name, correct, total,
                                                                                                         ratio * 100))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The NRC Defenses')
    # dataset：数据集名称MNIST/CIFAR10
    # seed：随机种子数，默认值为100
    # gpu_index：gpu的数量，默认为0
    # search：是否寻找半径的最佳值，默认为False，不进行寻找
    # radius：半径，默认值为0.02
    # num_points：每个样本产生的数据点的数量，默认为1000
    # radius_min：半径的最小值，默认为0.0
    # radius_max：半径的最大值，默认为1.0
    # radius_step：半径的迭代步长，默认为0.01
    # *************选择不同的数据集MNIST/CIFAR10*************
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--search', default=False, type=lambda x: (str(x).lower() == 'true'), help='indicate whether search the radius r')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--radius', type=float, default=0.02, help='in the case of not search radius r, we set the radius of the hypercube')
    parser.add_argument('--num_points', type=int, default=1000, help='number of points chosen in the adjacent region for each image')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--mean', type=float, default=0.0, help='mean value of Gaussian distribution')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--std', type=float, default=1.0, help='standard deviation of Gaussian distribution')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--radius_min', type=float, default=0.0, help='lower bound of radius')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--radius_max', type=float, default=0.1, help='upper bound of radius')
    # *************根据不同的数据集进行选择*************
    parser.add_argument('--radius_step', type=float, default=0.01, help='step size of radius in searching')
    arguments = parser.parse_args()
    main(arguments)
