import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from attacks.attack_methods.attack_utils import predict
from attacks.attack_methods.RFGSM import RFGSMAttack
from attacks.Generation import Generation

class RFGSMGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                 device,
                 attack_batch_size, eps, alpha_ratio):
        super(RFGSMGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location,
                                              adv_examples_dir, device)
        # dataset：数据集MNIST/CIFAR10
        # attack_name：攻击名称
        # targeted：目标攻击（True）还是非目标攻击（False）
        # raw_model_location：原始模型所在位置
        # clean_data_location：1000个干净样本所在位置
        # adv_examples_dir：存放对抗样本的位置
        # attack_batch_size：产生对抗样本的分组大小
        # eps：扰动的大小
        # alpha_ratio：随机扰动的权重
        self.attack_batch_size = attack_batch_size
        self.epsilon = eps
        self.alpha_ratio = alpha_ratio

    def generate(self):
        # nature_samples：干净数据集
        # labels_samples：干净数据集的真实标签
        # 将参数传入RFGSM攻击中
        attacker = RFGSMAttack(model=self.raw_model, epsilon=self.epsilon, alpha_ratio=self.alpha_ratio)
        # 产生对抗样本
        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys=self.labels_samples,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        # 预测对抗样本标签，转为numpy格式
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        # 将对抗性样本存入AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_AdvExamples.npy或者MNIST
        # 将对抗性样本标签存入AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_AdvLabels.npy或者MNIST
        # 将样本真实标签存入AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_TrueLabels.npy或者MNIST
        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)
        # 如果对抗样本的预测结果与真实标签不同，则mis+1，计算对抗样本的制作成功率
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

    name = 'RFGSM'
    targeted = False
    # 将参数传入
    rfgsm = RFGSMGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                            clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                            eps=args.epsilon, attack_batch_size=args.attack_batch_size, alpha_ratio=args.alpha_ratio)
    # 产生对抗样本
    rfgsm.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The RFGSM Attack Generation')
    # dataset：数据集名称MNIST/CIFAR10
    # modelDir：原始模型存放位置data/
    # cleanDir：1000个干净数据集存放位置clean_datasets/
    # adv_saver：对抗样本的存放位置AdversarialExampleDatasets/
    # seed：随机种子，默认为100
    # gpu_index：gpu的数量，默认为0
    # epsilon：扰动大小，默认为0.1
    # alpha_ratio：随机添加扰动的权重，默认为0.5
    # attack_batch_size：产生对抗样本的分组大小，默认为100
    # ************数据集可选MNIST/CIFAR10************
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/',
                        help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # ************添加的扰动大小根据数据集不同选择************
    parser.add_argument('--epsilon', type=float, default=0.1, help='the epsilon value of RFGSM')
    # ************alpha根据数据集不同选择************
    parser.add_argument('--alpha_ratio', type=float, default=0.5, help='the ratio of alpha related to epsilon in RFGSM')
    parser.add_argument('--attack_batch_size', type=int, default=100,
                        help='the default batch size for adversarial example generation')
    arguments = parser.parse_args()
    main(arguments)
