import numpy as np
import torch

from Defenses.DefenseMethods.defenses import Defense
from src.train_test import validation_evaluation

class NRCDefense(Defense):
    def __init__(self, model=None, defense_name='NRC', dataset=None, device=None, num_points=100):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # num_points：数据点数量
        super(NRCDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        # Dataset：大写的数据集名称MNIST/CIFAR10
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        self.num_points = num_points

    # 寻找最佳立方体的半径r（条件是RC的分类精度要与原始模型的分类精度相当）
    def search_best_radius(self, validation_loader=None, radius_min=0.0, radius_max=1.0, radius_step=0.01):
        # validation_loader：验证集
        # radius_min：半径最小值为0.0
        # radius_max：半径最大值为1.0
        # radius_step：半径迭代的步长为0.001
        self.model.eval()
        with torch.no_grad():
            # val_acc：原始验证集的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            print('<--- original classification accuracy on validation dataset is {:.4f} --->'.format(val_acc))
            # total_step：总的迭代次数
            total_step = int((radius_max - radius_min) / radius_step)
            # 循环进行迭代
            for index in range(total_step):
                # tmp_radius：每一次的半径都加上一个步长
                tmp_radius = radius_min + radius_step * (index + 1)
                total = 0.0
                correct = 0.0
                for images, labels in validation_loader:
                    # rc_preds：返回RC预测的标签
                    # rc_labels：将RC预测标签转化为tensor
                    rc_preds = self.region_based_classification(samples=images, radius=tmp_radius)
                    rc_labels = torch.from_numpy(rc_preds)
                    # 如果预测的标签和真实标签一致，正确个数加1
                    correct += (rc_labels == labels).sum().item()
                    # total：为总的验证集数目
                    total += labels.size(0)
                # rc_acc：RC预测标签的准确率
                rc_acc = correct / total
                print('\tcurrent radius is {:.2f}, validation accuracy is {:.1f}/{:.1f}={:.5f}'.format(tmp_radius, correct, total, rc_acc))
                # 如果原始验证集的分类精度与RC模型的分类精度之差大于0.01时（RC的精度开始降低），则返回上一个迭代的半径
                if (val_acc - rc_acc) > 1e-2:
                    return round(tmp_radius - radius_step, 2)
            # 否则返回最大迭代半径1.0
            return radius_max

    # 对一个样本构造立方体，选取模型判断标签最多的那个标签值返回，也就是RC模型的一个过程
    def region_based_classification_single(self, sample, radius, mean, std):
        # sample：一个样本
        # radius：半径
        self.model.eval()
        assert sample.shape[0] == 1, "the sample parameter should be one example in numpy format"
        # copy_sample：结构为1*C*H*W
        copy_sample = np.copy(sample)
        with torch.no_grad():
            copy_sample = torch.from_numpy(copy_sample).to(self.device)
            # hypercube_samples的结构为num_points*C*H*W，第四维都是和copy_sample相同的
            hypercube_samples = copy_sample.repeat(self.num_points, 1, 1, 1).to(self.device).float()
            # random_space：与hypercube_samples相同
            random_space = torch.Tensor(*hypercube_samples.size()).to(self.device).float()
            # [-radius, radius)的均匀分布中采样的随机数
            random_space.uniform_(-radius, radius)
            # ************改变添加的噪声为正态分布噪声（均值为0，方差为1）************
            # random_space = torch.randn(*hypercube_samples.size())
            # ************改变添加的噪声为正态分布噪声（均值和方差可以自选）************
            # m = torch.tensor([mean])
            # s = torch.tensor([std])
            # m = m.expand(random_space.shape[2], random_space.shape[3], random_space.shape[0], random_space.shape[1]).permute(2,3,0,1)
            # s = s.expand(random_space.shape[2], random_space.shape[3], random_space.shape[0], random_space.shape[1]).permute(2,3,0,1)
            # random_space = torch.normal(m, s)
            # 添加随机噪声，并将每个像素值控制在[0.0, 1.0]中
            hypercube_samples = torch.clamp(hypercube_samples + random_space, min=0.0, max=1.0)
            # 对这num_points个数据点进行模型预测
            hypercube_preds = self.model(hypercube_samples)
            # hypercube_labels：返回预测的标签值(0,1,...,9)
            hypercube_labels = torch.max(hypercube_preds, dim=1)[1]
            # bin_count：返回每个标签出现的次数
            bin_count = torch.bincount(hypercube_labels)
            # rc_label：出现次数最多的那个标签值(0,1,...,9)
            rc_label = torch.max(bin_count, dim=0)[1]
            # 返回标签值
            return rc_label.cpu().numpy()

    # 返回通过RC模型的所有samples的标签
    def region_based_classification(self, samples, radius, mean, std):
        # samples：输入样本
        # radius：半径
        self.model.eval()
        # rc_labels：RC模型分类返回的所有samples的标签
        rc_labels = []
        # 循环返回RC模型分类的samples的标签
        for i in range(samples.shape[0]):
            x = samples[i: i + 1]
            label = self.region_based_classification_single(sample=x, radius=radius, mean=mean, std=std)
            rc_labels.append(label)
        return np.array(rc_labels)
    def defense(self):
        print('As the defense of RT does not retrain the model, we do not implement this method')
