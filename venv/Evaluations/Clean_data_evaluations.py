import argparse
import os
import random
import sys
import warnings
import numpy as np
import scipy.stats as jslib
import torch
import torch.nn.functional as F

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from src.get_datasets import get_mnist_test_loader, get_cifar10_test_loader
# *****************引入的是目标模型*****************
from src.resnet_model import ResNet18
from src.mnist_model import MNIST_CNN

# 防御衡量
def defense_utility_measure(pred_def, pred_raw, true_label):
    # pred_def：防御模型预测结果
    # pred_raw：原始模型预测结果
    # true_label：真实标签

    # 计算原始模型的平均分类精度
    # 预测正确为True，错误为False
    correct_prediction_raw = np.equal(np.argmax(pred_raw, axis=1), true_label)
    # True转化为1.0，False转化为0.0，求均值
    acc_raw = np.mean(correct_prediction_raw.astype(float))

    # 计算防御模型的平均分类精度
    correct_prediction_def = np.equal(np.argmax(pred_def, axis=1), true_label)
    acc_def = np.mean(correct_prediction_def.astype(float))

    # 计算防御模型与原始模型分类精度的差值CAV
    cav_result = acc_def - acc_raw

    # 寻找预测正确的索引
    idx_def = np.squeeze(np.argwhere(correct_prediction_def == True))
    idx_raw = np.squeeze(np.argwhere(correct_prediction_raw == True))
    # 求交集，同时在原始模型和防御模型都预测正确的索引
    idx = np.intersect1d(idx_def, idx_raw, assume_unique=True)
    # 在防御模型上面预测正确，而在原始模型上面预测错误的比例（校正比CRR）
    num_rectify = len(idx_def) - len(idx)
    crr_result = num_rectify / len(pred_def)
    # 在原始模型上面预测正确，而在防御模型上面预测错误的比例（牺牲比CSR）
    num_sacrifice = len(idx_raw) - len(idx)
    csr_result = num_sacrifice / len(pred_def)

    return acc_raw, acc_def, cav_result, crr_result, csr_result

# 模型对测试集进行预测经过softmax函数得到预测标签，返回预测标签及真实标签
def prediction(model, test_loader, device):
    # model：模型
    # test_loader：测试集
    print('\nThe #{}# model is evaluated on the testing dataset loader ...'.format(model.model_name))
    model = model.to(device)
    model.eval()
    prediction = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = F.softmax(logits, dim=1).cpu().numpy()
            prediction.extend(predicted)
            true_labels.extend(labels)
    prediction = np.array(prediction)
    true_labels = np.array(true_labels)
    return prediction, true_labels

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
    # dataset：数据集名称MNIST或CIFAR10
    dataset = args.dataset.upper()
    assert dataset == 'MNIST' or dataset == 'CIFAR10'
    # 加载模型和测试集
    raw_model_location = '{}{}/model/{}_raw.pt'.format('../data/', dataset, dataset)
    if dataset == 'MNIST':
        raw_model = MNIST_CNN().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_mnist_test_loader(dir_name='../data/MNIST/', batch_size=30)
    else:
        raw_model = ResNet18().to(device)
        raw_model.load(path=raw_model_location, device=device)
        test_loader = get_cifar10_test_loader(dir_name='../data/CIFAR10/', batch_size=25)
    raw_model.eval()

    # 原始模型对测试集进行预测
    predicted_raw, true_label = prediction(model=raw_model, test_loader=test_loader, device=device)

    # 需要再训练的防御
    re_train_defenses = {'NAT', 'RLT', 'RLT1', 'RLT2', 'RLT3', 'EAT', 'UAPAT'}
    # 其他防御
    other_defenses = {'NRC'}
    # defense_name：防御名称
    defense_name = args.defense.upper().strip()
    # 如果是再训练模型防御
    if defense_name in re_train_defenses:
        print('\nthe ##{}## defense is a kind of complete defenses that retrain the model'.format(defense_name))
        # 加载防御模型
        # defended_model_location：防御模型位置DefenseEnhancedModels/defense_name/CIFAR10_defense_name_enhanced.pt或者MNIST
        defended_model_location = '{}/{}/{}_{}_enhanced.pt'.format('../DefenseEnhancedModels', defense_name, dataset, defense_name)
        defended_model = MNIST_CNN().to(device) if dataset == 'MNIST' else ResNet18().to(device)
        defended_model.load(path=defended_model_location, device=device)
        defended_model.eval()
        # 利用防御模型进行标签预测
        predicted_defended, _ = prediction(model=defended_model, test_loader=test_loader, device=device)
        # 计算防御指标
        raw_acc, def_acc, cav, crr, csr = defense_utility_measure(predicted_defended, predicted_raw, true_label)
    else:
        if defense_name == 'NRC':
            print('\n##{}## defense is a kind of region-based classification defenses ... '.format(defense_name))
            from Defenses.DefenseMethods.NRC import NRCDefense
            num_points = 1000
            radius = args.radius
            mean = args.mean
            std = args.std
            nrc = NRCDefense(model=raw_model, defense_name='NRC', dataset=dataset, device=device, num_points=num_points)
            predicted_defended = []
            with torch.no_grad():
                for index, (images, labels) in enumerate(test_loader):
                    nrc_labels = nrc.region_based_classification(samples=images, radius=radius, mean=mean, std=std)
                    predicted_defended.extend(nrc_labels)
            predicted_defended = np.array(predicted_defended)
            correct_prediction_def = np.equal(predicted_defended, true_label)
            def_acc = np.mean(correct_prediction_def.astype(float))
            correct_prediction_raw = np.equal(np.argmax(predicted_raw, axis=1), true_label)
            raw_acc = np.mean(correct_prediction_raw.astype(float))
            # Classification Accuracy Variance(CAV)
            cav = def_acc - raw_acc
            # Find the index of correct predicted examples by defence-enhanced model and raw model
            idx_def = np.squeeze(np.argwhere(correct_prediction_def == True))
            idx_raw = np.squeeze(np.argwhere(correct_prediction_raw == True))
            idx = np.intersect1d(idx_def, idx_raw, assume_unique=True)
            crr = (len(idx_def) - len(idx)) / len(predicted_raw)
            csr = (len(idx_raw) - len(idx)) / len(predicted_raw)
        else:
            raise ValueError('{} is not supported!!!'.format(defense_name))
    # 输出防御指标的值
    print("****************************")
    print("The utility evaluation results of the {} defense for {} Dataset are as follow:".format(defense_name, dataset))
    print('Acc of Raw Model:\t\t{:.2f}%'.format(raw_acc * 100))
    print('Acc of {}-enhanced Model:\t{:.2f}%'.format(defense_name, def_acc * 100))
    print('CAV: {:.2f}%'.format(cav * 100))
    print('CRR: {:.2f}%'.format(crr * 100))
    print('CSR: {:.2f}%'.format(csr * 100))
    print("****************************")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset：数据集名称MNIST或CIFAR10
    # gpu_index：gpu的数量
    # seed：随机种子数
    # defense：防御名称
    # ************数据集名称可选MNIST/CIFAR10************
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='the dataset should be MNIST or CIFAR10')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    # ************防御模型可选************
    parser.add_argument('--defense', type=str, default='NAT', help="the defense method to be evaluated")
    # ************根据防御模型选择（防御模型的radius）************
    parser.add_argument('--radius', type=float, default=0.02,
                        help='in the case of not search radius r, we set the radius of the hypercube')
    # *************根据防御模型和数据集进行选择（防御模型的mean）*************
    parser.add_argument('--mean', type=float, default=0.0, help='mean value of Gaussian distribution')
    # *************根据防御模型和数据集进行选择（防御模型的std）*************
    parser.add_argument('--std', type=float, default=1.0, help='standard deviation of Gaussian distribution')
    arguments = parser.parse_args()
    main(arguments)
