import argparse
import os
import random
import shutil
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from src.get_datasets import get_cifar10_test_loader, get_mnist_test_loader

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # dataset：数据集名称MNIST或CIFAR10
    # num：选取的干净样本的数量
    # dataset_location：数据集所在位置data/CIFAR10/或者MNIST
    # raw_model_location：模型所在位置data/CIFAR10/model/CIFAR10_raw.pt或者MNIST
    dataset = args.dataset.upper()
    num = args.number
    # *****************数据集存放的位置*****************
    dataset_location = '../data/{}/'.format(dataset)
    raw_model_location = '../data/{}/model/{}_raw.pt'.format(dataset, dataset)
    print("\nStarting to select {} {} Candidates Example, which are correctly classified by the Raw Model from {}\n".format(num, dataset,
                                                                                                                      raw_model_location))
    # 加载模型，获取测试集
    # raw_model：模型
    # test_loader：测试集
    # load the raw model and testing dataset
    assert args.dataset == 'MNIST' or args.dataset == 'CIFAR10'
    if dataset == 'MNIST':
        raw_model = MNIST_CNN().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_mnist_test_loader(dir_name=dataset_location, batch_size=1, shuffle=False)
    else:
        raw_model = ResNet18().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_cifar10_test_loader(dir_name=dataset_location, batch_size=1, shuffle=False)
    # 获取分类正确的测试集
    # successful：测试集经过模型，保留被正确预测的图像和标签以及它们对应softmax最小输出的标签
    successful = []
    raw_model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = raw_model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted == label:
                _, least_likely_class = torch.min(output.data, 1)
                successful.append([image, label, least_likely_class])
    print(len(successful))
    # 随机选取num个正确分类的图像
    candidates = random.sample(successful, num)

    candidate_images = []
    candidate_labels = []
    candidates_llc = []
    candidate_targets = []
    for index in range(len(candidates)):
        # 将选择的图片，标签和最不可能的标签分开
        image = candidates[index][0].cpu().numpy()
        image = np.squeeze(image, axis=0)
        candidate_images.append(image)
        label = candidates[index][1].cpu().numpy()[0]
        llc = candidates[index][2].cpu().numpy()[0]
        # 生成0~9的10个标签，去除真实标签，随机选择一个标签
        classes = [i for i in range(10)]
        classes.remove(label)
        target = random.sample(classes, 1)[0]
        # 将随机目标标签，最不可能分类目标标签和真实标签转化为one-hot标签保存
        one_hot_label = [0 for i in range(10)]
        one_hot_label[label] = 1
        one_hot_llc = [0 for i in range(10)]
        one_hot_llc[llc] = 1
        one_hot_target = [0 for i in range(10)]
        one_hot_target[target] = 1
        candidate_labels.append(one_hot_label)
        candidates_llc.append(one_hot_llc)
        candidate_targets.append(one_hot_target)
    # 图像
    candidate_images = np.array(candidate_images)
    # 图像对应真实one-hot标签
    candidate_labels = np.array(candidate_labels)
    # 图像对应最不可能分类的one-hot标签
    candidates_llc = np.array(candidates_llc)
    # 图像对应非真实标签的随机one-hot标签
    candidate_targets = np.array(candidate_targets)
    # 打开CIFAR10/或MNIST/文件夹
    if dataset not in os.listdir('./'):
        os.mkdir('./{}/'.format(dataset))
    else:
        shutil.rmtree('{}'.format(dataset))
        os.mkdir('./{}/'.format(dataset))
    # 将图片，标签，最不可能分类的标签，目标标签存入clean_datasets/CIFAR10/CIFAR10_inputs.npy等等或者MNIST
    np.save('./{}/{}_inputs.npy'.format(dataset, dataset), candidate_images)
    np.save('./{}/{}_labels.npy'.format(dataset, dataset), candidate_labels)
    np.save('./{}/{}_llc.npy'.format(dataset, dataset), candidates_llc)
    np.save('./{}/{}_targets.npy'.format(dataset, dataset), candidate_targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Candidate Selection for Clean Data set')
    # dataset：数据集名称MNIST或CIFAR10，默认为CIFAR10
    # seed：随机种子数，默认为100
    # gpu_index：gpu数量，默认为0
    # number：选取的即将被攻击的干净样本数量，默认为1000
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset (MNIST or CIFAR10)')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--number', type=int, default=1000, help='the total number of candidate samples that will be randomly selected')
    arguments = parser.parse_args()
    main(arguments)
