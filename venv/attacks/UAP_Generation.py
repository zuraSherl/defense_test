import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from attacks.attack_methods.attack_utils import predict
from attacks.attack_methods.UAP import UniversalAttack
from attacks.Generation import Generation
from src.get_datasets import get_cifar10_train_validate_loader, get_mnist_train_validate_loader

class UAPGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                 device, max_iter_uni, frate,
                 epsilon, overshoot, max_iter_df):
        super(UAPGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location,
                                            adv_examples_dir, device)
        self.max_iter_uni = max_iter_uni
        self.fooling_rate = frate
        self.epsilon = epsilon
        self.overshoot = overshoot
        self.max_iter_df = max_iter_df

    def generate(self):
        # 将参数传入到UAP攻击中
        attacker = UniversalAttack(model=self.raw_model, fooling_rate=self.fooling_rate,
                                   max_iter_universal=self.max_iter_uni,
                                   epsilon=self.epsilon, overshoot=self.overshoot, max_iter_deepfool=self.max_iter_df)
        assert self.dataset.upper() == 'MNIST' or self.dataset.upper() == 'CIFAR10', "dataset should be MNIST or CIFAR10!"
        # 获取MNIST/CIFAR10的训练集和验证集
        if self.dataset.upper() == 'MNIST':
            samples_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST/', batch_size=1,
                                                                           valid_size=0.9,
                                                                           shuffle=True)
        else:  # 'CIFAR10':
            samples_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='../data/CIFAR10/',
                                                                             batch_size=1, valid_size=0.9,
                                                                             augment=False, shuffle=True)
        # 计算UAP扰动并转化为numpy形式
        universal_perturbation = attacker.universal_perturbation(dataset=samples_loader, validation=valid_loader,
                                                                 device=self.device)
        universal_perturbation = universal_perturbation.cpu().numpy()
        # 将UAP扰动存放在AdversarialExampleDatasets/UAP_MNIST_universal_perturbation中
        np.save('{}{}_{}_universal_perturbation'.format(self.adv_examples_dir, self.attack_name, self.dataset),
                universal_perturbation)
        # 产生UAP对抗样本
        adv_samples = attacker.perturbation(xs=self.nature_samples, uni_pert=universal_perturbation, device=self.device)
        # UAP对抗样本的预测标签softmax
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        # UAP对抗样本的预测标签转化为numpy
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        # 将UAP对抗样本、UAP对抗样本标签和真实标签进行保存
        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)
        # 计算添加了UAP扰动对抗样本的误分类率
        mis = 0
        for i in range(len(adv_samples)):
            if self.labels_samples[i].argmax(axis=0) != adv_labels[i]:
                mis = mis + 1
        print(
            '\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset,
                                                                                        mis, len(adv_samples),
                                                                                        mis / len(adv_labels) * 100))
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
    # name：攻击名称UAP
    name = 'UAP'
    # targeted：False为非目标攻击
    targeted = False
    # 将参数传入UAP攻击中
    df = UAPGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                       clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                       max_iter_uni=args.max_iter_universal,
                       frate=args.fool_rate, epsilon=args.epsilon, overshoot=args.overshoot,
                       max_iter_df=args.max_iter_deepfool)
    # 计算UAP并产生UAP对抗样本
    df.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The UAP Attack Generation')
    # dataset：数据集名称NINST/CIFAR10
    # modelDir：MNIST/CIFAR10模型所在位置
    # cleanDir：用于产生对抗样本的正确分类的干净数据集存放位置
    # adv_saver：产生的对抗样本的存放位置
    # seed：随机种子，默认为100
    # gpu_index：gpu的数量，默认为0
    # fool_rate：UAP扰动要达到的错误率，默认为1.0
    # epsilon：UAP添加的最大扰动值，默认为0.1
    # max_iter_universal：最大的UAP迭代次数，默认为20
    # overshoot：DeepFool添加的扰动大小，默认为0.02
    # max_iter_deepfool：DeepFool的最大迭代次数，默认为10
    # **********************数据集可选**********************
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/',
                        help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # **********************根据不同数据集进行选择**********************
    parser.add_argument('--fool_rate', type=float, default=1.0, help="the fooling rate")
    # **********************根据不同数据集进行选择**********************
    parser.add_argument('--epsilon', type=float, default=0.1, help='controls the magnitude of the perturbation')
    # **********************根据不同数据集进行选择**********************
    parser.add_argument('--max_iter_universal', type=int, default=20, help="the maximum iterations for UAP")
    # **********************根据不同数据集进行选择**********************
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot parameter for DeepFool')
    # **********************根据不同数据集进行选择**********************
    parser.add_argument('--max_iter_deepfool', type=int, default=10, help='the maximum iterations for DeepFool')
    arguments = parser.parse_args()
    main(arguments)