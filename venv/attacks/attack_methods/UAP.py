import os
import sys
import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from attacks.attack_methods.DEEPFOOL import DeepFoolAttack
from attacks.attack_methods.attacks import Attack

class UniversalAttack(Attack):
    def __init__(self, model=None, max_iter_universal=np.inf, fooling_rate=0.5, epsilon=0.1, overshoot=0.02, max_iter_deepfool=10):
        # model：模型
        # max_iter_universal：UAP的最大迭代次数
        # fooling_rate：要达到的欺骗率
        # epsilon：UAP添加的最大扰动
        # overshoot：DeepFool扰动大小
        # max_iter_deepfool：最大DeepFool迭代次数
        super(UniversalAttack, self).__init__(model=model)
        self.model = model
        # UAP扰动参数
        self.max_iter_universal = max_iter_universal
        self.fooling_rate = fooling_rate
        self.epsilon = epsilon
        # DeepFool扰动参数
        self.overshoot_deepfool = overshoot
        self.max_iter_deepfool = max_iter_deepfool

    # 将添加的通用扰动大小限制在[-epsilon,epsilon]之间
    def projection_linf(self, v, eps):
        # v：通用扰动大小
        # eps：通用扰动的最大值
        # 将v的大小限制在[-eps,eps]之间
        v = np.sign(v) * np.minimum(abs(v), eps)
        return v

    # 计算UAP扰动并返回
    def universal_perturbation(self, dataset, validation, device):
        # dataset：用于计算UAP扰动的输入干净数据集
        # validation：验证集
        print('\n\nstarting to compute the universal adversarial perturbation with the training dataset ......\n')
        # iteration：迭代次数
        # ratio：错误率
        iteration, ratio = 0, 0.0
        # uni_pert：通用扰动初始化为0.0
        uni_pert = torch.zeros(size=iter(dataset).next()[0].shape)
        # 当错误率小于欺骗率并且迭代次数小于最大UAP迭代次数时进行循环
        while ratio < self.fooling_rate and iteration < self.max_iter_universal:
            print('iteration: {}'.format(iteration))
            self.model.eval()
            # 遍历输入数据集的每一个数据
            for index, (image, label) in enumerate(dataset):
                # original：每个输入干净图像通过模型的预测标签
                original = torch.max(self.model(image.to(device)), 1)[1]  # prediction of the nature image
                # perturbed_image：扰动图像=原图像+通用扰动，并将像素限制在[0.0，1.0]之间
                perturbed_image = torch.clamp(image + uni_pert, 0.0, 1.0)  # predication of the perturbed image
                # current：添加通用扰动后图像通过模型的预测标签
                current = torch.max(self.model(perturbed_image.to(device)), 1)[1]
                # 如果添加扰动后的标签与干净图像的标签相同，则添加DeepFool扰动更新通用扰动的值
                if original == current:
                    # 将参数传入DeepFool攻击中
                    deepfool = DeepFoolAttack(model=self.model, overshoot=self.overshoot_deepfool, max_iters=self.max_iter_deepfool)
                    # 返回对抗样本，添加的扰动和迭代的次数
                    _, delta, iter_num = deepfool.perturbation_single(sample=perturbed_image.numpy(), device=device)
                    # update the universal perturbation
                    # 如果添加DeepFool的扰动次数小于最大DeepFool迭代次数减1（说明构造了成功的DeepFool对抗样本）
                    if iter_num < self.max_iter_deepfool - 1:
                        # 更新通用扰动=原来的通用扰动+DeepFool扰动
                        uni_pert += torch.from_numpy(delta)
                        # 将通用扰动uni_pert的大小限制在[-epsilon,epsilon]之间
                        uni_pert = self.projection_linf(v=uni_pert, eps=self.epsilon)
            # 迭代次数加1
            iteration += 1
            print('\tcomputing the fooling rate w.r.t current the universal adversarial perturbation ......')
            # success：添加通用扰动后的标签与干净图像的标签不一致的数量
            # total：总的输入数据集的数量
            success, total = 0.0, 0.0
            # 遍历输入数据集的每一个数据
            for index, (v_image, label) in enumerate(validation):
                # label：真实标签
                label = label.to(device)
                # original：每个输入干净图像通过模型的预测标签
                original = torch.max(self.model(v_image.to(device)), 1)[1]  # prediction of the nature image
                # perturbed_v_image：添加通用扰动后图像，像素限制在[0.0,1.0]之间
                perturbed_v_image = torch.clamp(v_image + uni_pert, 0.0, 1.0)  # predication of the perturbed image
                # current：添加通用扰动后图像通过模型的预测标签
                current = torch.max(self.model(perturbed_v_image.to(device)), 1)[1]
                # 计算总的输入数据数量和预测错误数量
                if original != current and current != label:
                    success += 1
                total += 1
            # 计算添加通用扰动后的错误率
            ratio = success / total
            print('\tcurrent fooling rate is {}/{}={}\n'.format(success, total, ratio))
        # 返回通用扰动
        return uni_pert

    # 产生UAP对抗样本
    def perturbation(self, xs, uni_pert, device):
        # xs：输入干净样本
        # uni_pert：通用扰动
        # adv_samples：对抗样本
        adv_samples = []
        # 对每一个输入添加UAP扰动
        for i in range(len(xs)):
            # 添加UAP扰动
            adv_image = xs[i: i + 1] + uni_pert
            # 将像素范围限制在[0.0,1.0]之间
            adv_image = np.clip(adv_image, 0.0, 1.0)
            # 放入adv_samples数组中
            adv_samples.extend(adv_image)
        # 返回UAP对抗样本
        return np.array(adv_samples)
