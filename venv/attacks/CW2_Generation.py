import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from attacks.attack_methods.attack_utils import predict
from attacks.attack_methods.CW2 import CW2Attack
from attacks.Generation import Generation

class CW2Generation(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                 device, attack_batch_size,
                 kappa, init_const, lr, binary_search_steps, max_iterations, lower_bound, upper_bound):
        super(CW2Generation, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location,
                                            adv_examples_dir, device)
        # dataset：数据集MNIST/CIFAR10
        # attack_name：攻击名称
        # targeted：目标标签
        # raw_model_location：原始模型所在位置
        # clean_data_location：1000个干净数据集所在位置
        # adv_examples_dir：对抗样本的存放位置
        # attack_batch_size：攻击的分组大小
        # kappa：k值
        # init_const：初始c值
        # lr：学习率
        # binary_search_steps：搜索c值的迭代次数
        # max_iterations：寻找对抗样本的最大迭代次数
        # lower_bound：像素的最小值
        # upper_bound：像素的最大值
        self.attack_batch_size = attack_batch_size
        self.kappa = kappa
        self.init_const = init_const
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate(self):
        # 将参数传入到CW2攻击中
        attacker = CW2Attack(model=self.raw_model, kappa=self.kappa, init_const=self.init_const, lr=self.lr,
                             binary_search_steps=self.binary_search_steps, max_iters=self.max_iter,
                             lower_bound=self.lower_bound,
                             upper_bound=self.upper_bound)
        # get the targeted labels
        # targets_samples：为目标分类标签one-hot形式
        # targets：获取真实标签（0~9）
        targets = np.argmax(self.targets_samples, axis=1)
        # generating
        # adv_samples：分组产生CW2对抗样本
        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys_target=targets,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        # adv_labels：对抗样本标签
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        # 转化为0~9的numpy形式
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        # 保存对抗样本、对抗样本标签和真实标签
        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)
        # mis_target：误分类率（成功产生CW2对抗样本）
        mis_target = 0.0
        for i in range(len(adv_samples)):
            if targets[i] == adv_labels[i]:
                mis_target += 1
        print(
            '\nFor **{}**(targeted attack) on **{}**, {}/{}={:.1f}% samples are misclassified as the specified targeted label\n'.format(
                self.attack_name, self.dataset, mis_target, len(adv_samples), mis_target / len(adv_samples) * 100.0))

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

    name = 'CW2'
    targeted = True
    # 产生CW2对抗样本保存并计算误分类率
    cw2 = CW2Generation(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                        clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                        attack_batch_size=args.attack_batch_size, kappa=args.confidence, init_const=args.initial_const,
                        binary_search_steps=args.search_steps, lr=args.learning_rate, lower_bound=args.lower_bound,
                        upper_bound=args.upper_bound, max_iterations=args.iteration)
    cw2.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The CW2 Attack Generation')
    # dataset：数据集MNIST/CIFAR10
    # modelDir：MNIST/CIFAR10模型存放位置
    # cleanDir：用于产生对抗样本的1000个干净样本的存放位置
    # adv_saver：产生的对抗样本的存放位置
    # seed：随机种子数，默认为100
    # gpu_index：gpu的数量
    # confidence：用于控制错误分类置信度k的值，默认为0
    # initial_const：初始化c的值，默认为0.001
    # learning_rate：求扰动的学习率，默认为0.02
    # iteration：求扰动的最大迭代次数，默认为10000
    # lower_bound：像素的最小值，默认为0.0
    # upper_bound：像素的最大值，默认为1.0
    # search_steps：寻找c值的循环次数，默认为10
    # attack_batch_size：产生对抗样本的分组大小，默认为100
    # ************数据集可选MNIST/CIFAR10************
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/',
                        help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # ************置信度可选************
    parser.add_argument('--confidence', type=float, default=0, help='the confidence of adversarial examples')
    # ************初始值可选************
    parser.add_argument('--initial_const', type=float, default=0.001,
                        help="the initial value of const c in the binary search.")
    parser.add_argument('--learning_rate', type=float, default=0.02, help="the learning rate of gradient descent.")
    parser.add_argument('--iteration', type=int, default=10000, help='maximum iteration')
    parser.add_argument('--lower_bound', type=float, default=0.0,
                        help='the minimum pixel value for examples (default=0.0).')
    parser.add_argument('--upper_bound', type=float, default=1.0,
                        help='the maximum pixel value for examples (default=1.0).')
    parser.add_argument('--search_steps', type=int, default=10,
                        help="the binary search steps to find the optimal const.")
    parser.add_argument('--attack_batch_size', type=int, default=100,
                        help='the default batch size for adversarial example generation')
    arguments = parser.parse_args()
    main(arguments)