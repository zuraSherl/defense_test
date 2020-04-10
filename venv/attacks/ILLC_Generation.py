import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from attacks.Generation import Generation
from attacks.attack_methods.ILLC import ILLCAttack
from attacks.attack_methods.attack_utils import predict

class ILLCGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                 device, eps, eps_iter,
                 num_steps, attack_batch_size):
        super(ILLCGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location,
                                             adv_examples_dir, device)
        # dataset：数据集名称
        # attack_name：攻击名称
        # targeted：目标攻击（True）还是非目标攻击（False）
        # raw_model_location：原始模型所在位置
        # clean_data_location：1000个干净样本所在位置
        # adv_examples_dir：存放对抗样本的位置
        # eps：扰动大小
        # eps_iter：迭代步长
        # num_steps：迭代次数
        # attack_batch_size：产生对抗样本的分组大小
        self.attack_batch_size = attack_batch_size
        self.epsilon = eps
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
    # ILLC产生对抗样本
    def generate(self):
        # 将参数传入ILLC攻击中
        attacker = ILLCAttack(model=self.raw_model, epsilon=self.epsilon, eps_iter=self.epsilon_iter,
                              num_steps=self.num_steps)
        # nature_samples：干净数据集
        # targets_samples：目标分类的标签，ILLC时为最不可能分类的标签
        # llc_labels：目标分类索引
        llc_labels = np.argmax(self.targets_samples, 1)
        # 产生对抗样本
        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys_target=llc_labels,
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
        # 如果对抗样本的预测结果与目标类不同，则mis+1，计算对抗样本的制作成功率
        mis_target = 0
        for i in range(len(adv_samples)):
            if llc_labels[i] == adv_labels[i]:
                mis_target += 1
        print('\nFor the **{}** (targeted attack) on **{}**, the misclassified rate = {}/{} = {:.1f}%\n'.format(
            self.attack_name, self.dataset, mis_target, len(adv_samples), mis_target / len(adv_samples) * 100))

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

    name = 'ILLC'
    targeted = True
    # 将参数传入类中
    illc = ILLCGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                          clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                          eps=args.epsilon, attack_batch_size=args.attack_batch_size, eps_iter=args.epsilon_iter,
                          num_steps=args.num_steps)
    # 产生对抗样本
    illc.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The ILLC Attack Generation')
    # ************数据集可选MNIST/CIFAR10************
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/',
                        help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # ************添加的扰动范围根据数据集不同选择************
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='the max epsilon value that is allowed to be perturbed')
    # ************添加的扰动步长大小根据数据集不同选择************
    parser.add_argument('--epsilon_iter', type=float, default=0.05, help='the one iterative eps of ILLC')
    # ************循环添加扰动的次数根据数据集不同选择************
    parser.add_argument('--num_steps', type=int, default=10, help='the number of perturbation steps')
    parser.add_argument('--attack_batch_size', type=int, default=100,
                        help='the default batch size for adversarial example generation')
    arguments = parser.parse_args()
    main(arguments)
