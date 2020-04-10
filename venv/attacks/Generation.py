import os
import shutil
from abc import ABCMeta
import numpy as np
import torch

# **************引入对应网络用于产生对抗样本**************
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18

class Generation(object):
    __metaclass__ = ABCMeta
    def __init__(self, dataset='MNIST', attack_name='FGSM', targeted=False, raw_model_location='../data/',
                 clean_data_location='../clean_datasets/', adv_examples_dir='../AdversarialExampleDatasets/',
                 device=torch.device('cpu')):
        # dataset：数据集MNIST或者CIFAR10
        # attack_name：攻击名称
        # targeted：目标攻击还非目标攻击
        # raw_model_location：训练网络所在位置
        # clean_data_location：用于生产对抗样本的干净数据集存放位置
        # adv_examples_dir：产生的对抗样本存放位置
        # 如果数据集不为MNIST或者CIFAR10，给出提示
        self.dataset = dataset.upper()
        if self.dataset not in {'MNIST', 'CIFAR10'}:
            raise ValueError("The data set must be MNIST or CIFAR10")
        # 如果攻击名称不为如下的其中之一，给出提示
        self.attack_name = attack_name.upper()
        supported = {'FGSM', 'RFGSM', 'BIM', 'PGD', 'DEEPFOOL', 'LLC', "RLLC", 'ILLC', 'JSMA', 'CW2'}
        if self.attack_name not in supported:
            raise ValueError(self.attack_name + 'is unknown!\nCurrently, our implementation support the attacks: ' + ', '.join(supported))
        # 加载模型和模型参数
        # 模型位置data/CIFAR10/model/CIFAR10_raw.pt或者data/MNIST/model/MNIST_raw.pt
        # **************根据不同模型可改变模型名称**************
        # **************衡量白盒攻击需要改变名称raw_model_location/防御名称/数据集名称_防御名称_enhanced.pt**************
        raw_model_location = '{}{}/model/{}_raw.pt'.format(raw_model_location, self.dataset, self.dataset)
        # 根据数据集名称的不同加载不同模型
        # **************根据不同模型加载模型参数**************
        if self.dataset == 'MNIST':
            self.raw_model = MNIST_CNN().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = ResNet18().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        # 加载非目标攻击将要被攻击的干净数据集和标签
        print('Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... ')
        # 加载将要被攻击的干净数据集clean_datasets/CIFAR10/CIFAR10_inputs.npy或者clean_datasets/MNIST/MNIST_inputs.npy
        self.nature_samples = np.load('{}{}/{}_inputs.npy'.format(clean_data_location, self.dataset, self.dataset))
        # 加载将要被攻击的干净标签clean_datasets/CIFAR10/CIFAR10_labels.npy或者clean_datasets/MNIST/MNIST_labels.npy
        self.labels_samples = np.load('{}{}/{}_labels.npy'.format(clean_data_location, self.dataset, self.dataset))

        # 如果为目标攻击，获取目标标签
        if targeted:
            print('For Targeted Attacks, loading the randomly selected targeted labels that will be attacked ......')
            # 如果是LLC, RLLC, ILLC目标攻击，则加载最不可能分类标签clean_datasets/CIFAR10/CIFAR10_llc.npy或者是MNIST
            if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
                print('#### Especially, for LLC, RLLC, ILLC, loading the least likely class that will be attacked')
                self.targets_samples = np.load('{}{}/{}_llc.npy'.format(clean_data_location, self.dataset, self.dataset))
            # 否则加载非真实标签的随机标签clean_datasets/CIFAR10/CIFAR10_targets.npy或者是MNIST
            else:
                self.targets_samples = np.load('{}{}/{}_targets.npy'.format(clean_data_location, self.dataset, self.dataset))

        # 对抗样本的存放位置
        # AdversarialExampleDatasets/attact_name/CIFAR10/或者AdversarialExampleDatasets/attact_name/MNIST/
        self.adv_examples_dir = adv_examples_dir + self.attack_name + '/' + self.dataset + '/'
        # 创建攻击名称文件夹AdversarialExampleDatasets/attact_name
        if self.attack_name not in os.listdir(adv_examples_dir):
            os.mkdir(adv_examples_dir + self.attack_name + '/')
        # 则创建数据集名称文件夹AdversarialExampleDatasets/attact_name/CIFAR10/或者AdversarialExampleDatasets/attact_name/MNIST/
        if self.dataset not in os.listdir(adv_examples_dir + self.attack_name + '/'):
            os.mkdir(self.adv_examples_dir)
        # 直接创建文件夹
        else:
            shutil.rmtree('{}'.format(self.adv_examples_dir))
            os.mkdir(self.adv_examples_dir)
        self.device = device

    def generate(self):
        print("abstract method of Generation is not implemented")
        raise NotImplementedError
