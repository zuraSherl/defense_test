import argparse
import os
import random
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim as SSIM

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from attacks.attack_methods.attack_utils import predict
# *************引入的是目标模型*************
from src.mnist_model import MNIST_CNN
from src.resnet_model import ResNet18

class AttackEvaluate:
    def __init__(self, DataSet='MNIST', AttackName='FGSM', RawModelLocation='../data/', CleanDataLocation='../clean_datasets/',
                 AdvExamplesDir='../AdversarialExampleDatasets/', device=torch.device('cpu')):
        # DataSet：数据集名称
        # dataset：数据集名称MNIST或CIFAR10
        # AttackName：攻击名称
        # attack_name：攻击名称
        # RawModelLocation：模型所在位置data/
        # CleanDataLocation：干净数据集所在位置clean_datasets/
        # AdvExamplesDir：对抗性样本所在位置AdversarialExampleDatasets/
        # color_mode：CIFAR10为RGB，MNIST为L
        # Targeted：False为非目标攻击，Ture为目标攻击

        self.device = device
        assert DataSet.upper() in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        self.dataset = DataSet.upper()
        self.color_mode = 'RGB' if self.dataset == 'CIFAR10' else 'L'
        self.attack_name = AttackName.upper()
        # 非目标攻击名称
        supported_un_targeted = ['FGSM', 'RFGSM', 'BIM', 'PGD', 'DEEPFOOL', 'UAP']
        # 目标攻击名称
        supported_targeted = ['LLC', "RLLC", 'ILLC', 'JSMA', 'CW2']
        assert self.attack_name in supported_un_targeted or self.attack_name in supported_targeted, \
            "\nCurrently, our implementation support attacks of FGSM, RFGSM, BIM, UMIFGSM, DeepFool, LLC, RLLC, ILLC, TMIFGSM, JSMA, CW2,....\n"
        if self.attack_name.upper() in supported_un_targeted:
            self.Targeted = False
        else:
            self.Targeted = True

        # 加载模型
        # raw_model_location：模型位置data/CIFAR10/model/CIFAR10_raw.pt或者MNIST
        # raw_model：模型
        # ********若要衡量白盒攻击将路径改为RawModelLocation/防御名称/数据集名称_防御名称_enhanced.pt********
        raw_model_location = '{}{}/model/{}_raw.pt'.format(RawModelLocation, self.dataset, self.dataset)
        if self.dataset == 'MNIST':
            self.raw_model = MNIST_CNN().to(device)
            self.raw_model.load(path=raw_model_location, device=device)
        else:
            self.raw_model = ResNet18().to(device)
            self.raw_model.load(path=raw_model_location, device=device)

        # 获取干净数据集及标签
        # nature_samples：干净数据集CleanDatasets/CIFAR10/CIFAR10_inputs.npy或者MNIST
        # labels_samples：干净数据集标签CleanDatasets/CIFAR10/CIFAR10_labels.npy或者MNIST
        self.nature_samples = np.load('{}{}/{}_inputs.npy'.format(CleanDataLocation, self.dataset, self.dataset))
        self.labels_samples = np.load('{}{}/{}_labels.npy'.format(CleanDataLocation, self.dataset, self.dataset))

        # 获取目标标签
        # 如果是LLC RLLC和ILLC攻击，为LLC RLLC和ILLC准备目标标签CleanDatasets/CIFAR10/CIFAR10_llc.npy
        if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
            self.targets_samples = np.load('{}{}/{}_llc.npy'.format(CleanDataLocation, self.dataset, self.dataset))
        # 否则目标标签为CleanDatasets/CIFAR10/CIFAR10_targets.npy
        else:
            self.targets_samples = np.load('{}{}/{}_targets.npy'.format(CleanDataLocation, self.dataset, self.dataset))

        # 获取对抗性样本
        # AdvExamplesDir：AdversarialExampleDatasets/attack_name/CIFAR10/或者MNIST
        # adv_samples：对抗性样本AdversarialExampleDatasets/attack_name/CIFAR10/attack_name_AdvExamples.npy或者MNIST
        self.AdvExamplesDir = AdvExamplesDir + self.attack_name + '/' + self.dataset + '/'
        # 如果没有这个路径则提示
        if os.path.exists(self.AdvExamplesDir) is False:
            print("the directory of {} is not existing, please check carefully".format(self.AdvExamplesDir))
        self.adv_samples = np.load('{}{}_AdvExamples.npy'.format(self.AdvExamplesDir, self.attack_name))
        # self.adv_labels = np.load('{}{}_AdvLabels.npy'.format(self.AdvExamplesDir, self.AttackName))

        # 对对抗性样本进行标签预测
        # predictions：对抗性样本预测标签
        predictions = predict(model=self.raw_model, samples=self.adv_samples, device=self.device).detach().cpu().numpy()
        # 定义softmax函数
        def soft_max(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        # 对预测标签进行softmax计算
        # softmax_prediction：经softmax的预测标签
        tmp_soft_max = []
        for i in range(len(predictions)):
            tmp_soft_max.append(soft_max(predictions[i]))
        self.softmax_prediction = np.array(tmp_soft_max)

    def successful(self, adv_softmax_preds, nature_true_preds, targeted_preds, target_flag):
        # adv_softmax_preds：对抗样本经softmax的预测标签
        # nature_true_preds：未经攻击样本的真实标签
        # targeted_preds：目标标签
        # target_flag：True为目标攻击，False为非目标攻击
        # 如果为目标攻击则达到目标标签返回True
        if target_flag:
            if np.argmax(adv_softmax_preds) == np.argmax(targeted_preds):
                return True
            else:
                return False
        # 如果为非目标攻击则与真实标签不同时返回True
        else:
            if np.argmax(adv_softmax_preds) != np.argmax(nature_true_preds):
                return True
            else:
                return False

    # 计算误分类率
    # 1 MR:Misclassification Rate
    def misclassification_rate(self):
        cnt = 0
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
        mr = cnt / len(self.adv_samples)
        print('MR:\t\t{:.1f}%'.format(mr * 100))
        return mr
    # 4 ASS: Average Structural Similarity
    def avg_SSIM(self):
        ori_r_channel = np.transpose(np.round(self.nature_samples * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        adv_r_channel = np.transpose(np.round(self.adv_samples * 255), (0, 2, 3, 1)).astype(dtype=np.float32)
        totalSSIM = 0
        cnt = 0
        """
        For SSIM function in skimage: http://scikit-image.org/docs/dev/api/skimage.measure.html
        multichannel : bool, optional If True, treat the last dimension of the array as channels. Similarity calculations are done 
        independently for each channel then averaged.
        """
        for i in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[i], nature_true_preds=self.labels_samples[i],
                               targeted_preds=self.targets_samples[i], target_flag=self.Targeted):
                cnt += 1
                totalSSIM += SSIM(X=ori_r_channel[i], Y=adv_r_channel[i], multichannel=True)
        print('ASS:\t{:.3f}'.format(totalSSIM / cnt))
        return totalSSIM / cnt
    # 6: PSD: Perturbation Sensitivity Distance
    def avg_PSD(self):
        psd = 0
        cnt = 0
        for outer in range(len(self.adv_samples)):
            if self.successful(adv_softmax_preds=self.softmax_prediction[outer], nature_true_preds=self.labels_samples[outer],
                               targeted_preds=self.targets_samples[outer], target_flag=self.Targeted):
                cnt += 1
                image = self.nature_samples[outer]
                pert = abs(self.adv_samples[outer] - self.nature_samples[outer])
                for idx_channel in range(image.shape[0]):
                    image_channel = image[idx_channel]
                    pert_channel = pert[idx_channel]
                    image_channel = np.pad(image_channel, 1, 'reflect')
                    pert_channel = np.pad(pert_channel, 1, 'reflect')
                    for i in range(1, image_channel.shape[0] - 1):
                        for j in range(1, image_channel.shape[1] - 1):
                            psd += pert_channel[i, j] * (1.0 - np.std(np.array(
                                [image_channel[i - 1, j - 1], image_channel[i - 1, j], image_channel[i - 1, j + 1], image_channel[i, j - 1],
                                 image_channel[i, j], image_channel[i, j + 1], image_channel[i + 1, j - 1], image_channel[i + 1, j],
                                 image_channel[i + 1, j + 1]])))
        print('PSD:\t{:.3f}'.format(psd / cnt))
        return psd / cnt

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
    attack = AttackEvaluate(DataSet=dataset, AttackName=args.attack, RawModelLocation=args.modelDir, CleanDataLocation=args.cleanDir,
                            AdvExamplesDir=args.adv_saver, device=device)
    attack.raw_model.eval()
    attack.misclassification_rate()
    attack.avg_SSIM()
    attack.avg_PSD()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Adversarial Attacks')
    # dataset：数据集名称CIFAR10或MNIST
    # seed：随机种子
    # gpu_index：gpu的数量
    # modelDir：模型所在位置data/
    # cleanDir：1000个干净数据集所在位置clean_datasets/
    # adv_saver：对抗性样本所在位置AdversarialExampleDatasets/
    # ************数据集名称可选MNIST/CIFAR10************
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    # ***********若要衡量白盒攻击，将其改为DefenseEnhancedModels***********
    parser.add_argument('--modelDir', type=str, default='../data/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../clean_datasets/', help='the directory for the clean dataset that will be attacked')
    # ************若要衡量黑盒攻击/白盒攻击，只需要改这个文件名为对应黑盒攻击/白盒攻击产生的对抗样本位置************
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    # ************攻击名称可选************
    # attack：攻击名称
    parser.add_argument('--attack', type=str, default='NAT', help='the attack name')
    arguments = parser.parse_args()
    main(arguments)
