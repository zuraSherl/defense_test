import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class RFGSMAttack(Attack):
    def __init__(self, model=None, epsilon=None, alpha_ratio=None):
        # model：模型
        # epsilon：扰动大小
        # alpha_ratio：随机添加扰动的比例
        super(RFGSMAttack, self).__init__(model)
        self.model = model
        self.epsilon = epsilon
        self.alpha_ratio = alpha_ratio

    # 用RFGSM方法产生对抗样本
    def perturbation(self, samples, ys, device):
        # samples：1000个干净样本（用于攻击）
        # ys：1000个干净样本的真实标签
        # copy_samples：复制干净样本
        copy_samples = np.copy(samples)
        # 对干净样本做一个随机变换
        copy_samples = copy_samples + (self.alpha_ratio * self.epsilon * np.sign(np.random.randn(*copy_samples.shape)))
        # 变化后的样本像素控制在[0.0,1.0]之间
        copy_samples = np.clip(copy_samples, 0.0, 1.0).astype(np.float32)
        # eps：FGSM产生对抗样本的扰动大小
        eps = (1.0 - self.alpha_ratio) * self.epsilon
        # 将copy_samples进行封装，可以对copy_samples进行求导
        var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
        # 对ys进行封装，不能对ys_target进行求导
        var_ys = tensor2variable(torch.LongTensor(ys), device=device)
        # 样本标签预测
        self.model.eval()
        preds = self.model(var_samples)
        loss_fun = torch.nn.CrossEntropyLoss()
        loss = loss_fun(preds, torch.max(var_ys, 1)[1])
        # 对copy_samples求导并求符号函数转化为numpy的形式
        loss.backward()
        gradient_sign = var_samples.grad.data.cpu().sign().numpy()
        # 计算对抗样本
        adv_samples = copy_samples + eps * gradient_sign
        # 将对抗样本的像素控制在[0.0,1.0]之间
        adv_samples = np.clip(adv_samples, 0.0, 1.0)
        return adv_samples

    # 分组产生对抗样本
    def batch_perturbation(self, xs, ys, batch_size, device):
        # batch_size：产生对抗样本的分组大小
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
        # adv_sample：对抗样本
        # number_batch：分组数量，向上取整
        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            # RLLC产生对抗样本
            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
