import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from attacks.attack_methods.attack_utils import predict
from attacks.attack_methods.DEEPFOOL import DeepFoolAttack
from attacks.Generation import Generation

class DeepFoolGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, overshoot, max_iters):
        super(DeepFoolGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                                                 device)
        self.overshoot = overshoot
        self.max_iters = max_iters
    def generate(self):
        # 将参数传入到DeepFool攻击中
        attacker = DeepFoolAttack(model=self.raw_model, overshoot=self.overshoot, max_iters=self.max_iters)
        # 产生DeepFool对抗样本
        adv_samples = attacker.perturbation(xs=self.nature_samples, device=self.device)
        # prediction for the adversarial examples
        # 计算DeepFool对抗样本经过原始模型的softmax预测结果
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        # 获取DeepFool对抗样本经过原始模型的标签
        adv_labels = torch.max(adv_labels, 1)[1]
        # 将标签转化为numpy形式
        adv_labels = adv_labels.cpu().numpy()
        # 将DeepFool对抗样本保存
        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        # 将DeepFool对抗样本对应标签保存
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        # 将DeepFool的真实标签保存
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)
        # mis：误分类率，计算1000个DeepFool对抗样本经过原始模型的误分类率
        mis = 0
        for i in range(len(adv_samples)):
            if self.labels_samples[i].argmax(axis=0) != adv_labels[i]:
                mis = mis + 1
        print('\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset, mis, len(adv_samples),
                                                                                          mis / len(adv_labels) * 100))
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
    name = 'DeepFool'
    targeted = False
    # 将参数传入DeepFool中
    df = DeepFoolGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                            clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device, max_iters=args.max_iters,
                            overshoot=args.overshoot)
    # 产生对抗样本并计算误分类率
    df.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The DeepFool Attack Generation')
    # dataset：数据集名称MNIST/CIFAR10
    # modelDir：MNIST/CIFAR10的模型存放位置
    # cleanDir：用于将要被攻击的干净数据集的存放位置
    # adv_saver：DeepFool产生对抗样本的存放位置
    # seed：随机种子，默认为100
    # gpu_index：gpu的数量，默认为0
    # max_iters：最大迭代次数，默认为50
    # overshoot：添加的扰动，使用论文中的值，默认为0.02
    # **************************可选值为MNIST/CIFAR10**************************
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # **************************最大迭代次数根据数据集不同选择**************************
    parser.add_argument('--max_iters', type=int, default=50, help="the max iterations")
    # **************************overshoot参数根据数据集不同选择**************************
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot')

    arguments = parser.parse_args()
    main(arguments)
