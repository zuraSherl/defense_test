import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class FGSMAttack(Attack):
    def __init__(self, model=None, epsilon=None):
        # model：模型
        # epsilon：扰动大小
        super(FGSMAttack, self).__init__(model)
        self.model = model
        self.epsilon = epsilon
    # 产生对抗性样本
    def perturbation(self, samples, ys, device):
        # smples：输入
        # ys：真实one-hot标签，对应标签值[0,0,0,1,0,0...]
        # epsilon：扰动大小
        copy_samples = np.copy(samples)
        # 对输入samples进行封装，设置为可求导
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
        # 对抗样本=原来的样本+epsilon*梯度
        adv_samples = copy_samples + self.epsilon * gradient_sign
        # 将对抗样本中的像素值剪切到0.0-1.0之间
        adv_samples = np.clip(adv_samples, 0.0, 1.0)
        return adv_samples

    # 对输入分组产生对抗性样本
    def batch_perturbation(self, xs, ys, batch_size, device):
        # xs：输入数据集
        # ys：对应的输出标签
        # batch_size：分组大小
        # 输入数据集的数量和标签的数量应该一致
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
        adv_sample = []
        # 将输入分割成number_batch个小组
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            # 每个组的起始位置和终点位置
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            # 对每个组分别计算梯度加入扰动
            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            # 所有的对抗样本放入adv_sample数组中
            adv_sample.extend(batch_adv_images)
        # 返回对抗样本
        return np.array(adv_sample)
