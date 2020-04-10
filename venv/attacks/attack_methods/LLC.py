import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class LLCAttack(Attack):
    def __init__(self, model=None, epsilon=None):
        super(LLCAttack, self).__init__(model)
        # model：模型
        # epsilon：扰动大小
        self.model = model
        self.epsilon = epsilon
    # 产生对抗样本
    def perturbation(self, samples, ys_target, device):
        # samples：1000个干净样本（用于攻击）
        # ys_target：samples目标类（最不可能的分类标签）
        # copy_samples：复制干净样本
        copy_samples = np.copy(samples)
        # 对copy_samples进行封装，可以对copy_samples进行求导
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        # 对ys_target进行封装，不能对ys_target进行求导
        var_ys_target = tensor2variable(torch.from_numpy(ys_target), device)
        # 对var_samples进行标签预测
        self.model.eval()
        preds = self.model(var_samples)
        # 求损失函数
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, var_ys_target)
        # 对copy_samples进行求导并求符号函数转为numpy的形式
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()
        # 求对抗样本
        adv_samples = copy_samples - self.epsilon * gradient_sign
        # 将对抗样本的像素限制在[0.0,1.0]之间
        adv_samples = np.clip(adv_samples, 0.0, 1.0)
        return adv_samples
    # 分组产生对抗样本
    def batch_perturbation(self, xs, ys_target, batch_size, device):
        # xs：1000个干净样本（用于攻击）
        # ys_target：xs目标类（最不可能的分类标签）
        # batch_size：分组大小
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"
        # adv_sample：对抗样本
        adv_sample = []
        # number_batch：分组数量，向上取整
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
