import argparse
import os
import random
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18
from attacks.attack_methods.attack_utils import predict

class SecurityEvaluate:
    def __init__(self, DataSet='MNIST', AttackName='LLC', AdvExamplesDir='../AdversarialExampleDatasets/', device=torch.device('cpu')):
        # DataSet：数据集名称
        # dataset：数据集名称（大写）
        # AttackName：攻击名称
        # attack_name：攻击名称（大写）
        # AdvExamplesDir：对抗性样本存放位置
        self.device = device
        assert DataSet.upper() in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        self.dataset = DataSet.upper()
        # raw_model：加载模型
        # ***********不同模型名称***********
        raw_model_location = '{}{}/model/{}_raw.pt'.format('../data/', self.dataset, self.dataset)
        if self.dataset == 'MNIST':
            self.raw_model = MNIST_CNN().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = ResNet18().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        self.raw_model.eval()
        self.attack_name = AttackName.upper()
        supported_un_targeted = ['FGSM', 'RFGSM', 'BIM', 'PGD', 'DEEPFOOL', 'UAP']
        supported_targeted = ['LLC', "RLLC", 'ILLC', 'JSMA', 'CW2']
        assert self.attack_name in supported_un_targeted or self.attack_name in supported_targeted, \
            "\nCurrently, our implementation support attacks of FGSM, RFGSM, BIM, UMIFGSM, DeepFool, LLC, RLLC, ILLC, TMIFGSM, JSMA, CW2,....\n"

        # 设置Targeted是目标攻击还是非目标攻击
        if self.attack_name.upper() in supported_un_targeted:
            self.Targeted = False
            print('the # {} # attack is a kind of Un-targeted attacks'.format(self.attack_name))
        else:
            self.Targeted = True
            print('the # {} # attack is a kind of Targeted attacks'.format(self.attack_name))

        # adv_samples，adv_labels，true_labels：加载对抗性样本，对抗性样本标签和真实标签
        self.adv_samples = np.load('{}{}/{}/{}_AdvExamples.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name)).astype(
            np.float32)
        self.adv_labels = np.load('{}{}/{}/{}_AdvLabels.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name))
        self.true_labels = np.load('{}{}/{}/{}_TrueLabels.npy'.format(AdvExamplesDir, self.attack_name, self.dataset, self.attack_name))

        # targets_samples：获取目标攻击的标签
        if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
            self.targets_samples = np.load('{}{}/{}_llc.npy'.format('../clean_datasets/', self.dataset, self.dataset))
        else:
            self.targets_samples = np.load('{}{}/{}_targets.npy'.format('../clean_datasets/', self.dataset, self.dataset))

    # 对对抗性样本经过防御后的标签预测
    def defense_predication(self, DefenseModelDirs, defense_name, **kwargs):
        # DefenseModelDirs：防御模型所在位置
        # defense_name：防御名称（大写）
        re_train_defenses = {'NAT', 'RLT', 'RLT1', 'RLT2', 'RLT3', 'EAT', 'UAPAT', 'NEAT', 'NRC', 'RAT', 'RAT1',
                             'RAT2', 'RAT3', 'RAT4', 'RAT5', 'RAT6', 'RAT7', 'RAT8', 'RAT9', 'RAT10', 'RAT11', 
                             'MART', 'NEW_MART', 'NEW_MART1', 'NEW_MMA'}
        other_defenses = {'NRC'}
        defense_name = defense_name.upper().strip()
        assert defense_name in re_train_defenses or input_transformation_defenses or other_defenses
        # 如果是重新训练网络防御
        if defense_name in re_train_defenses:
            print('\n##{}## defense is a kind of complete defenses that retrain the model'.format(defense_name))
            # 加载防御模型
            defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format(DefenseModelDirs, defense_name, self.dataset, defense_name)
            defended_model = MNIST_CNN().to(self.device) if self.dataset == 'MNIST' else ResNet18().to(self.device)
            defended_model.load(path=defended_model_location, device=self.device)
            defended_model.eval()
            # 进行标签预测
            predication = predict(model=defended_model, samples=self.adv_samples, device=self.device)
            # 返回标签行向量
            labels = torch.argmax(predication, 1).cpu().numpy()
            return labels
        else:
            if defense_name == 'NRC':
                print('\n##{}## defense is a kind of region-based classification defenses ... '.format(defense_name))
                from Defenses.DefenseMethods.NRC import NRCDefense
                num_points = 1000
                assert 'nrc_radius' in kwargs
                assert 'nrc_mean' in kwargs
                assert 'nrc_std' in kwargs
                radius = kwargs['nrc_radius']
                mean = kwargs['nrc_mean']
                std = kwargs['nrc_std']
                nrc = NRCDefense(model=self.raw_model, defense_name='NRC', dataset=self.dataset, device=self.device,
                               num_points=num_points)
                labels = nrc.region_based_classification(samples=self.adv_samples, radius=radius, mean=mean, std=std)
                return labels
            else:
                raise ValueError('{} is not supported!!!'.format(defense_name))

    def success_rate(self, defense_predication):
        # defense_predication：防御预测结果标签
        # adv_labels：对抗性样本标签
        # true_labels：真实标签
        true_labels = np.argmax(self.true_labels, 1)
        # targets：目标攻击的标签
        targets = np.argmax(self.targets_samples, 1)
        assert defense_predication.shape == true_labels.shape and true_labels.shape == self.adv_labels.shape and self.adv_labels.shape == targets.shape
        original_misclassification = 0.0
        defense_success = 0.0
        for i in range(len(defense_predication)):
            # 如果为目标攻击，计算攻击成功条件下，防御成功的数量
            if self.Targeted:
                if self.adv_labels[i] == targets[i]:
                    original_misclassification += 1
                    if defense_predication[i] == true_labels[i]:
                        defense_success += 1
            # 如果为非目标攻击，计算攻击成功条件下，防御成功的数量
            else:
                if self.adv_labels[i] != true_labels[i]:
                    original_misclassification += 1
                    if defense_predication[i] == true_labels[i]:
                        defense_success += 1
        # 返回攻击成功和防御成功的数量
        return original_misclassification, defense_success

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

    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    # 将参数传入
    # *********若要衡量黑盒攻击/白盒攻击的防御效果，只需将对抗样本的文件名改为黑盒攻击/白盒攻击产生对抗样本的位置即可*********
    security_eval = SecurityEvaluate(DataSet=dataset, AttackName=args.attack, AdvExamplesDir='../AdversarialExampleDatasets/',
                                     device=device)
    defense_names = args.defenses.upper().split(',')
    params = {
        'nrc_radius': args.radius,
        'nrc_mean':args.mean,
        'nrc_std':args.std
    }
    print("\n****************************")
    print("The classification accuracy of adversarial examples ({}) w.r.t following defenses:".format(args.attack))
    # 循环进行防御衡量
    for defense in defense_names:
        defense = defense.strip()
        preds = security_eval.defense_predication(DefenseModelDirs='../DefenseEnhancedModels', defense_name=defense, **params)
        original_misclassification, defense_success = security_eval.success_rate(preds)
        # 计算防御的成功率
        print('\tFor {} defense, accuracy={:.0f}/{:.0f}={:.1f}%\n'.format(defense, defense_success, original_misclassification,
                                                                          defense_success / original_misclassification * 100))
    print("****************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset：数据集名称，CIFAR10或MNIST
    # gpu_index：gpu数量
    # seed：随机种子数
    # attack：攻击名称
    # defenses：防御名称
    # ***********数据集可选MNSIT/CIFAR10***********
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    # ***********攻击方法可选***********
    parser.add_argument('--attack', type=str, default='FGSM', help='the attack name (only one)')
    # ***********防御模型可选多种进行衡量***********
    parser.add_argument('--defenses', type=str, default='NAT, RLT, RLT1, RLT2, RLT3',
                        help="the defense methods to be evaluated (multiply defenses should be split by comma)")
    # ************根据防御模型选择（防御模型的radius）************
    parser.add_argument('--radius', type=float, default=0.02,
                        help='in the case of not search radius r, we set the radius of the hypercube')
    # *************根据防御模型和数据集进行选择（防御模型的mean）*************
    parser.add_argument('--mean', type=float, default=0.0, help='mean value of Gaussian distribution')
    # *************根据防御模型和数据集进行选择（防御模型的std）*************
    parser.add_argument('--std', type=float, default=1.0, help='standard deviation of Gaussian distribution')
    arguments = parser.parse_args()
    main(arguments)
