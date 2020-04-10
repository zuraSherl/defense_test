import math
import numpy as np
import torch

from attacks.attack_methods.attack_utils import tensor2variable
from attacks.attack_methods.attacks import Attack

class CW2Attack(Attack):
    def __init__(self, model=None, kappa=0, init_const=0.001, lr=0.02, binary_search_steps=5, max_iters=10000, lower_bound=0.0, upper_bound=1.0):
        # model：模型
        # kappa：用于控制置信度，一般设置为0
        # init_const：系数c的初始值
        # lr：求使目标函数最小的添加的扰动所使用的学习率
        # binary_search_steps：搜索c的循环次数
        # max_iters：最大迭代次数用于产生对抗样本
        # lower_bound：图像中像素的最小值
        # upper_bound：图像中像素的最大值
        super(CW2Attack, self).__init__(model=model)
        self.model = model
        self.kappa = kappa * 1.0
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iters
        self.binary_search_steps = binary_search_steps

    def perturbation(self, samples, ys_targets, batch_size, device):
        # samples：输入干净样本
        # ys_targets：目标标签
        # batch_size：分组大小=输入样本的数量
        assert len(samples) == batch_size, "the length of sample is not equal to the batch_size"
        # transform the samples [lower, upper] to [-1, 1] and then to the arctanh space
        # 将像素点x属于[lower_bound,upper_bound]变为x属于[-1,1]通过[x-(lower_bound+upper_bound)/2]/[(upper_bound-lower_bound)/2]
        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        # 然后将像素点转为arctanh的形式，arctanh(x)=0.5*ln(1+x/1-x)，求w的值，w=arctanh[(x-0.5)/0.5]
        # 将w用x表示
        arctanh_samples = np.arctanh((samples - mid_point) / half_range * 0.9999)
        # 将输入样本进行封装，可以对其进行求导
        var_samples = tensor2variable(torch.from_numpy(arctanh_samples), device=device, requires_grad=True)
        # set the lower and upper bound for searching 'c' const in the CW2 attack
        # 为每个输入样本生成一个对应的系数c的初始值
        const_origin = np.ones(shape=batch_size, dtype=float) * self.init_const
        # 设置系数c的最大值为10的10次方
        c_upper_bound = [1e10] * batch_size
        # 设置系数c的最小值为0
        c_lower_bound = np.zeros(batch_size)
        # convert targets to one hot encoder
        # 生成10*10，对角线为1其余元素为0的矩阵
        temp_one_hot_matrix = np.eye(10)
        # 目标标签one-hot存放位置
        targets_in_one_hot = []
        # 对每一个目标标签进行one-hot转换
        for i in range(batch_size):
            # 获取当前目标标签对应矩阵中的one-hot形式
            current_target = temp_one_hot_matrix[ys_targets[i]]
            # 将one-hot标签放入数组中
            targets_in_one_hot.append(current_target)
        # 将one-hot标签转化为tensor形式并进行封装
        targets_in_one_hot = tensor2variable(torch.FloatTensor(np.array(targets_in_one_hot)), device=device)
        # best_l2：最佳l2范数，初始化为一个很大的值
        best_l2 = [1e10] * batch_size
        # best_perturbation：添加的最佳扰动后的图像，与samples的大小相同，初始化为0
        best_perturbation = np.zeros(var_samples.size())
        # current_prediction_class：当前预测标签，初始化为-1
        current_prediction_class = [-1] * batch_size
        # 返回True或者False（是否成功找到对抗样本）
        def attack_achieved(pre_softmax, target_class):
            # pre_softmax：经softmax的预测值
            # target_class：目标标签（0~9）
            pre_softmax[target_class] -= self.kappa
            # 如果预测的标签与目标标签相同则返回True，否则返回Fasle
            return np.argmax(pre_softmax) == target_class
        self.model.eval()
        # Outer loop for linearly searching for c
        # 进行不同次的c值搜索，寻找最小l2且成功产生的对抗样本
        for search_for_c in range(self.binary_search_steps):
            # modifier：初始化与samples大小相同的，每个像素都为0.0
            modifier = torch.zeros(var_samples.size()).float()
            # 将modifier封装，默认可以求导（最后求得是使目标函数最小的扰动的值）
            modifier = tensor2variable(modifier, device=device, requires_grad=True)
            # 设置Adam优化器，参数为modifier，学习率为lr
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            # 将系数c转化为tensor并封装
            var_const = tensor2variable(torch.FloatTensor(const_origin), device=device)
            print("\tbinary search step {}:".format(search_for_c))
            # 迭代，寻找最小l2且成功产生的对抗样本
            for iteration_times in range(self.max_iterations):
                # inverse the transform tanh -> [0, 1]
                # 将arctanh(x)=0.5*ln[(1+x)/(1-x)]带入tanh(x)=[e(x)-e(-x)]/[e(x)+e(-x)]中得到tanh(x)的取值范围为[0,1]
                # 扰动图像大小（论文中的公式）：x+σ=0.5tanh(w)+0.5使得x+σ的范围为[0,1]
                # 将添加扰动后图像的大小控制在[0,1]之间
                perturbed_images = torch.tanh(var_samples + modifier) * half_range + mid_point
                # 预测添加扰动后的图像
                prediction = self.model(perturbed_images)
                # 计算添加扰动的各个像素的平方和（都转化为[0,1]范围再求添加扰动的平方）
                l2dist = torch.sum((perturbed_images - (torch.tanh(var_samples) * half_range + mid_point)) ** 2, [1, 2, 3])
                # 第一项为非目标标签对应softmax输出的最大值，第二项为对应目标标签预测的softmax值，第三项为k值控制置信度大小
                # constraint_loss为论文中的f(x')
                constraint_loss = torch.max((prediction - 1e10 * targets_in_one_hot).max(1)[0] - (prediction * targets_in_one_hot).sum(1),
                                            torch.ones(batch_size, device=device) * self.kappa * -1)
                # c*f(x')
                loss_f = var_const * constraint_loss
                # loss：目标函数=l2损失值+c*f(x')（求所有图像的总和）
                loss = l2dist.sum() + loss_f.sum()  # minimize |r| + c * loss_f(x+r,l)
                # 反向传播，每一次迭代会更新modifier的值
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # update the best l2 distance, current predication class as well as the corresponding adversarial example
                # 循环每一个图像
                for i, (dist, score, img) in enumerate(
                        zip(l2dist.data.cpu().numpy(), prediction.data.cpu().numpy(), perturbed_images.data.cpu().numpy())):
                    # 如果l2范数小于最佳l2范数且成功找到对抗样本
                    if dist < best_l2[i] and attack_achieved(score, ys_targets[i]):
                        # 更新l2
                        best_l2[i] = dist
                        # 更新当前预测标签（0~9）
                        current_prediction_class[i] = np.argmax(score)
                        # 更新添加扰动后的图像
                        best_perturbation[i] = img
            # update the best constant c for each sample in the batch
            # 循环每个图像，更新c当前值
            for i in range(batch_size):
                # 如果当前预测标签等于目标标签且不等于-1（即c值可以成功产生对抗样本，则减小c的值）
                if current_prediction_class[i] == ys_targets[i] and current_prediction_class[i] != -1:
                    # c的上界为原来的c的上界和初始值的最小值
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i])
                    # 如果小于10的10次方，则更新c的值为中间值
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                # 否则增大c的值
                else:
                    c_lower_bound[i] = max(c_lower_bound[i], const_origin[i])
                    # 如果上界小于1e10则将c的值调整到上下界的中间
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                    # 如果上界还是1e10则将c的值放大10倍
                    else:
                        const_origin[i] *= 10
        # 返回最终的添加了扰动的图像（数组形式）
        return np.array(best_perturbation)

    # 分批产生CW2对抗样本，返回数组形式
    def batch_perturbation(self, xs, ys_target, batch_size, device):
        # xs：输入样本
        # ys_target：目标标签
        # batch_size：分组大小
        assert len(xs) == len(ys_target), "The lengths of samples and its ys should be equal"
        adv_sample = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
            batch_adv_images = self.perturbation(xs[start:end], ys_target[start:end], batch_size, device)
            adv_sample.extend(batch_adv_images)
        return np.array(adv_sample)
