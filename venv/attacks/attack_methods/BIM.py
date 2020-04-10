import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class BIMAttack(Attack):
    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
        # model：模型
        # epsilon：扰动大小
        # eps_iter：迭代步长
        # num_steps：迭代次数，默认为5
        super(BIMAttack, self).__init__(model)
        self.model = model
        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
    # 用BIM（基本迭代法）产生对抗样本
    def perturbation(self, samples, ys, device):
        # samples：干净样本（用于攻击）
        # ys：samples的真实标签
        # copy_samples：复制干净样本
        copy_samples = np.copy(samples)
        self.model.to(device)
        # 循环添加扰动
        for index in range(self.num_steps):
            # 对copy_samples进行封装，可以对copy_samples进行求导
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            # 对copy_samples的标签进行封装，不可以对其进行求导
            var_ys = tensor2variable(torch.LongTensor(ys), device=device)
            # preds：对var_samples的标签预测
            self.model.eval()
            preds = self.model(var_samples)
            # 计算损失函数
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, torch.max(var_ys, 1)[1])
            # 对var_samples进行链式法则求导，并求sign值转化为numpy数组
            loss.backward()
            gradient_sign = var_samples.grad.data.cpu().sign().numpy()
            # 计算对抗样本，epsilon_iter为迭代步长
            copy_samples = copy_samples + self.epsilon_iter * gradient_sign
            # 将对抗样本的大小控制在[samples - self.epsilon,samples + self.epsilon]的范围内，epsilon为扰动大小
            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            # 将对抗样本的像素限制在[0.0,1.0]之间
            copy_samples = np.clip(copy_samples, 0.0, 1.0)
        return copy_samples

    # 分组产生对抗样本
    def batch_perturbation(self, xs, ys, batch_size, device):
        # xs：干净样本（用于攻击）
        # ys：xs的真实标签
        # batch_size：分组大小
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
        # adv_sample：对抗样本
        adv_sample = []
        # number_batch：有多少个组，向上取整
        number_batch = int(math.ceil(len(xs) / batch_size))
        # 分组产生对抗样本
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
