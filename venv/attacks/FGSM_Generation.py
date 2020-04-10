import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from attacks.attack_methods.attack_utils import predict
from attacks.attack_methods.FGSM import FGSMAttack
from attacks.Generation import Generation

class FGSMGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                 device, eps,
                 attack_batch_size):
        # dataset：数据集
        # attack_name：攻击算法名称
        # targeted：False（非目标攻击），Ture（目标攻击）
        # attack_batch_size：产生对抗性样本的分组大小
        # eps：扰动大小
        # raw_model_location：模型所在位置
        # clean_data_location：干净数据集存放位置
        # adv_examples_dir：对抗性样本的存放位置
        super(FGSMGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location,
                                             adv_examples_dir, device)
        self.attack_batch_size = attack_batch_size
        self.epsilon = eps
    def generate(self):
        # nature_samples：干净数据集
        # labels_samples：干净标签
        # raw_model：加载的CNN或者ResNet模型
        # 将模型raw_model和扰动大小epsilon参数传入FGSMAttack类中
        attacker = FGSMAttack(model=self.raw_model, epsilon=self.epsilon)
        # 产生对抗性样本
        adv_samples = attacker.batch_perturbation(xs=self.nature_samples, ys=self.labels_samples,
                                                  batch_size=self.attack_batch_size,
                                                  device=self.device)
        # 对对抗性样本进行封装，返回预测结果标签
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        # 得到对应的标签值
        adv_labels = torch.max(adv_labels, 1)[1]
        # 将标签值转化为数组形式
        adv_labels = adv_labels.cpu().numpy()
        # 将对抗样本保存到AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_AdvExamples.npy或者MNIST
        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        # 将对抗样本标签到AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_AdvLabels.npy或者MNIST
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        # 将样本真实标签保存到AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_TrueLabels.npy或者MNIST
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
    # dataset：为MNIST或CIFAR10
    # attack_name：扰动名称
    # targeted：目标攻击或者非目标攻击
    # raw_model_location：模型存放的位置，默认为../RawModels/
    # clean_data_location：将要被攻击的干净数据集，默认为../CleanDatasets/
    # adv_examples_dir：加入干扰后的数据集存放位置，默认为../AdversarialExampleDatasets/
    # eps：扰动大小参数
    # attack_batch_size：生成对抗性样本的分组大小

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

    name = 'FGSM'
    targeted = False
    # 将参数传入FGSMGeneration中
    fgsm = FGSMGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                          clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device,
                          eps=args.epsilon,
                          attack_batch_size=args.attack_batch_size)
    # 产生对抗性样本
    fgsm.generate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The FGSM Attack Generation')
    # dataset：为MNIST或CIFAR10
    # modelDir：为模型存放的位置，默认为../data/
    # cleanDir：将要被攻击的干净数据集，默认为../clean_datasets/
    # adv_save：加入干扰后的数据集存放位置，默认为../AdversarialExampleDatasets/
    # seed：默认值为100
    # gpu_index：默认值为0
    # epsilon：扰动大小，默认值为0.1
    # attack_batch_size：分组大小，默认值为100
    # **************************可选值为MNIST/CIFAR10**************************
    parser.add_argument('--dataset', type=str, default='MNIST', help='the dataset should be MNIST or CIFAR10')
    # *******************若要产生白盒攻击样本，则改变模型位置为防御模型所在位置DefenseEnhancedModels*******************
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/',
                        help='the directory for the clean dataset that will be attacked')
    # *******************若要产生黑盒攻击/白盒攻击，则改变文件名*******************
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # **********************对应MNIST/CIFAR10数据集的扰动值**********************
    parser.add_argument('--epsilon', type=float, default=0.3, help='the epsilon value of FGSM')
    parser.add_argument('--attack_batch_size', type=int, default=100,
                        help='the default batch size for adversarial example generation')
    arguments = parser.parse_args()
    main(arguments)