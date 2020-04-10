import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class DeepFoolAttack(Attack):
    def __init__(self, model=None, overshoot=0.02, max_iters=50):
        # model：模型
        # overshoot：添加的扰动大小（计算的是达到分类边界的扰动，overshoot是需要额外添加的扰动使跨越扰动边界）
        # max_iters：最大迭代次数
        super(DeepFoolAttack, self).__init__(model=model)
        self.model = model
        self.overshoot = overshoot
        self.max_iterations = max_iters
    # 产生一个DeepFool对抗样本，返回扰动大小、迭代次数和对抗样本（可能是成功的对抗样本也可能不是）
    def perturbation_single(self, sample, device):
        # sample：一个干净样本
        assert sample.shape[0] == 1, 'only perturbing one sample'
        # copy_sample：复制sample
        copy_sample = np.copy(sample)
        # var_sample：对copy_sample进行封装，可以对其求导，并将其转化为float类型
        var_sample = tensor2variable(torch.from_numpy(copy_sample), device=device, requires_grad=True).float()
        self.model.eval()
        # prediction：干净样本经过模型预测的softmax标签
        prediction = self.model(var_sample)
        # original：干净样本经过模型预测的真实标签
        original = torch.max(prediction, 1)[1]
        # current：当前的标签（0~9）
        current = original
        # 将预测结果由tensor形式转为numpy形式并按照降序进行排列（里面的值为索引）
        I = np.argsort(prediction.data.cpu().numpy() * -1)
        # 构造和样本相同大小的全为0的数组
        perturbation_r_tot = np.zeros(copy_sample.shape, dtype=np.float32)
        # iteration：迭代次数
        iteration = 0
        # 如果当前的标签值等于原始标签值并且迭代次数小于最大迭代次数时进行循环（成功产生对抗样本或达到最大迭代次数时停止）
        while (original == current) and (iteration < self.max_iterations):
            # 清空梯度
            zero_gradients(var_sample)
            self.model.eval()
            # f_kx：将样本输入到模型中经过softmax预测的标签
            f_kx = self.model(var_sample)
            # current：var_sample对应的最大预测结果的索引
            current = torch.max(f_kx, 1)[1]
            # I[0, 0]：为原始样本预测的最大值的索引
            # f_kx[0, I[0, 0]]：为当前预测结果对应原始样本真实标签索引的softmax的值，保留中间参数
            # grad_original：计算var_sample预测为真实标签的梯度
            f_kx[0, I[0, 0]].backward(retain_graph=True)
            grad_original = np.copy(var_sample.grad.data.cpu().numpy())
            # closest_dist：与真实类的最小分类边界距离
            closest_dist = 1e10
            for k in range(1, 10):
                # 清空梯度
                zero_gradients(var_sample)
                # grad_current：计算var_sample经过网络预测为其他标签的梯度，保留中间参数
                f_kx[0, I[0, k]].backward(retain_graph=True)
                grad_current = var_sample.grad.data.cpu().numpy().copy()
                # w_k：当前计算的标签梯度减去真实标签梯度
                w_k = grad_current - grad_original
                # f_k：当前的k对应softmax预测值减去真实标签对应的softmax的值并转化为numpy的形式
                f_k = (f_kx[0, I[0, k]] - f_kx[0, I[0, 0]]).detach().data.cpu().numpy()
                # dist_k：计算其他类到真实分类边界的距离
                dist_k = np.abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-15)
                # closest_dist：更新与真实类的最小分类边界距离
                # closest_w：更新当前计算的标签梯度减去真实标签梯度
                if dist_k < closest_dist:
                    closest_dist = dist_k
                    closest_w = w_k
            # 计算最终扰动（距离乘距离方向）
            r_i = (closest_dist + 1e-4) * closest_w / np.linalg.norm(closest_w)
            # 为每个像素添加扰动，都为r_i
            perturbation_r_tot = perturbation_r_tot + r_i
            # 放大perturbation_r_tot扰动并叠加到干净样本上，每个像素的范围限制在[0.0,1.0]之间
            tmp_sample = np.clip((1 + self.overshoot) * perturbation_r_tot + sample, 0.0, 1.0)
            # 将tmp_sample进行封装
            var_sample = tensor2variable(torch.from_numpy(tmp_sample), device=device, requires_grad=True)
            # 循环次数加1
            iteration += 1
        # 循环结束，像对抗样本添加最终的扰动获得对抗样本
        adv = np.clip(sample + (1 + self.overshoot) * perturbation_r_tot, 0.0, 1.0)
        # 返回对抗样本，添加的扰动和迭代的次数
        return adv, perturbation_r_tot, iteration

    # 分组产生DeepFool对抗样本
    def perturbation(self, xs, device):
        # xs：干净样本
        print('\nThe DeepFool attack perturbs the samples one by one ......\n')
        # adv_samples：对抗样本数组
        adv_samples = []
        # 一个一个产生对抗样本
        for i in range(len(xs)):
            adv_image, _, _ = self.perturbation_single(sample=xs[i: i + 1], device=device)
            adv_samples.extend(adv_image)
        # 返回对抗样本数组
        return np.array(adv_samples)
