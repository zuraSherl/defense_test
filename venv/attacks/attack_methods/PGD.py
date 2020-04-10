import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class PGDAttack(Attack):
    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
        # model：模型
        # epsilon：最大扰动
        # eps_iter：迭代步长
        # num_steps：迭代次数
        super(PGDAttack, self).__init__(model)
        self.model = model
        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
    # 添加PGD扰动
    def perturbation(self, samples, ys, device):
        # samples：干净样本
        # ys：真实标签，对应标签值[0,0,0,1,0,0...]
        # copy_samples：复制samples
        copy_samples = np.copy(samples)
        self.model.to(device)
        # randomly chosen starting points inside the L_\inf ball around the
        # 从均匀分布[-epsilon,epsilon)中采样与copy_samples叠加
        copy_samples = copy_samples + np.random.uniform(-self.epsilon, self.epsilon, copy_samples.shape).astype('float32')
        # 循环num_steps次
        for index in range(self.num_steps):
            # 对输入copy_samples进行封装
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            # 将ys标签值转化为longerTensor类型然后进行封装
            var_ys = tensor2variable(torch.LongTensor(ys), device=device)
            self.model.eval()
            # 经过网络返回预测的结果标签
            preds = self.model(var_samples)
            # 交叉熵损失函数
            loss_fun = torch.nn.CrossEntropyLoss()
            # 计算交叉熵损失，preds为预测值，后面一项为输出结果（每一行最大值的索引，输出的结果）
            loss = loss_fun(preds, torch.max(var_ys, 1)[1])
            # 对var_samples进行链式法则求导，并求sign值转化为numpy数组
            loss.backward()
            gradient_sign = var_samples.grad.data.cpu().sign().numpy()
            # 对抗样本=原来的样本+epsilon_iter*梯度
            copy_samples = copy_samples + self.epsilon_iter * gradient_sign
            # 将对抗样本的范围设置在[-epsilon,epsilon]中
            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            # 将对抗样本的范围设置在[0.0,1.0]中
            copy_samples = np.clip(copy_samples, 0.0, 1.0)
        return copy_samples
    # 分组产生PGD对抗样本
    def batch_perturbation(self, xs, ys, batch_size, device):
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        adv_sample = np.array(adv_sample)
        return adv_sample
