import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class ILLCAttack(Attack):
    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
        super(ILLCAttack, self).__init__(model)
        # model：模型
        # epsilon：扰动大小
        # eps_iter：迭代步长
        # num_steps：迭代次数
        self.model = model
        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
    # 用ILLC（迭代最小似然分类法）法产生对抗样本
    def perturbation(self, samples, ys_target, device):
        # samples：1000个干净样本（用于攻击）
        # ys_target：1000个干净样本的目标类标签
        # copy_samples：复制干净样本
        # var_ys_target：将目标标签封装，不能求导
        copy_samples = np.copy(samples)
        var_ys_target = tensor2variable(torch.from_numpy(ys_target), device)
        # 循环添加扰动
        for index in range(self.num_steps):
            # 对干净样本进行封装，可以对copy_samples进行求导
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            # 预测样本标签
            self.model.eval()
            preds = self.model(var_samples)
            # 计算损失函数
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, var_ys_target)
            # 对copy_samples求导并求符号函数转化为numpy形式
            loss.backward()
            gradient_sign = var_samples.grad.data.cpu().sign().numpy()
            # 根据步长求对抗样本
            copy_samples = copy_samples - self.epsilon_iter * gradient_sign
            # 将对抗样本的大小控制在[samples - self.epsilon,samples + self.epsilon]的范围内，epsilon为扰动大小
            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            # 将对抗样本的像素限制在[0.0,1.0]之间
            copy_samples = np.clip(copy_samples, 0.0, 1.0)
        return copy_samples
    # 分组产生对抗样本
    def batch_perturbation(self, xs, ys_target, batch_size, device):
        # xs：1000个干净样本（用于攻击）
        # ys_target：1000个干净样本的目标类标签
        # batch_size：产生对抗样本的分组大小
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"
        # adv_sample：对抗样本
        # number_batch：分组数量，向上取整
        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            # 分组产生对抗样本
            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
