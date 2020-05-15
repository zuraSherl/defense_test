import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import skimage as sk

from attacks.attack_methods.attack_utils import tensor2variable
from Denfenses.DefenseMethods.defenses import Defense
from src.train_test import validation_evaluation

# 改变一张图像的曝光度
def brightness(x, c=0.15):
    # x：一张输入图像，shape为W*H*C
    # c：改变的像素值，默认为0.15
    # 将rgb图转化为hsv空间，shape为W*H*C，像素值为[0,1]
    x = sk.color.rgb2hsv(x)
    # 将第二个通道加上0.15，然后裁剪到[0,1]
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    # 将图像还原为rgb
    x = sk.color.hsv2rgb(x)
    # 将图像的范围裁剪到[0,1]，返回rgb图像
    return np.clip(x, 0, 1)

# 改变一组图像的曝光度
def batch_brightness(x, c=0.15):
    # x：一组图像
    # c：改变的像素值，默认为0.15
    # 返回改变曝光度后的一组图像
    return np.array([brightness(xi, c) for xi in x])

class RLFATDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练参数
        # kwargs：防御参数

        super(RLFATDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        # 将数据集名称转化为大写的形式
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # make sure to parse the parameters for the defense
        assert self._parsing_parameters(**kwargs)

        # get the training_parameters, the same as the settings of RawModels
        # num_epochs：获取训练次数
        # batch_size：获取分组大小
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']

        # 准备MNIST/CIFAR10的优化器
        if self.Dataset == "MNIST":
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                       momentum=training_parameters['momentum'],
                                       weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])


    # 封装防御参数
    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nparsing the user configuration for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs.get(key)))
        # attack_step_num：PGD迭代次数
        # step_size：每一步攻击的扰动步长
        # epsilon：最大扰动
        # splited_block：RBS划分的块数
        # k：损失函数的平衡系数
        self.attack_step_num = kwargs.get('attack_step_num')
        self.step_size = kwargs.get('step_size')
        self.epsilon = kwargs.get('epsilon')
        self.splited_block = kwargs.get('splited_block')
        self.k = kwargs.get('k')
        return True

    # 产生经过RBS变换的PGD对抗样本
    def rbs_pgd_generation(self, var_natural_images=None, var_natural_labels=None):
        # var_natural_images：干净样本
        # var_natural_labels：干净样本对应的标签
        self.model.eval()
        # natural_images：将干净样本转化为numpy的形式
        natural_images = var_natural_images.cpu().numpy()

        # copy_images：复制干净样本numpy形式
        copy_images = natural_images.copy()
        # 从[-epsilon,epsilon）的均匀分布中随机采样，形成初始的扰动叠加在干净样本上
        copy_images = copy_images + np.random.uniform(-self.epsilon, self.epsilon, copy_images.shape).astype('float32')

        # 进行迭代，求PGD对抗样本
        for i in range(self.attack_step_num):
            # 将初始的扰动样本由numpy形式转化为tensor形式
            var_copy_images = torch.from_numpy(copy_images).to(self.device)
            # 可以对其进行求导
            var_copy_images.requires_grad = True

            # 将其输入模型进行预测
            preds = self.model(var_copy_images)
            # 计算损失值
            loss = F.cross_entropy(preds, var_natural_labels)
            # 对输入求梯度
            gradient = torch.autograd.grad(loss, var_copy_images)[0]
            # 对梯度求符号函数并转为numpy形式
            gradient_sign = torch.sign(gradient).cpu().numpy()
            # 对样本添加一小步扰动
            copy_images = copy_images + self.step_size * gradient_sign

            # 将样本的扰动大小控制在[natural_images-epsilon,natural_images+epsilon]的范围之内
            copy_images = np.clip(copy_images, natural_images - self.epsilon, natural_images + self.epsilon)
            # 将扰动大小控制在[0.0,1.0]之间
            copy_images = np.clip(copy_images, 0.0, 1.0)

        # PGD对抗样本
        pgd_adv_images = torch.from_numpy(copy_images).to(self.device)

        # 对产生的PGD对抗样本进行RBS变换
        # 将tensor转化为numpy形式
        copy_images = copy_images.cpu().numpy()
        # 将PGD对抗样本的shape由N*C*W*H变为N*W*H*C
        copy_images = np.transpose(copy_images, (0, 2, 3, 1))
        clip_num = self.splited_block - 1
        # 获取PGD对抗样本的shape，为N*W*H*C
        _, w, h, c = copy_images.shape
        # 产生与x相同shape的全为0的矩阵
        result = np.zeros_like(copy_images, dtype=np.float32)
        # 为划分的块进行编号0~splited_block-1
        clip_index = list(range(clip_num + 1))
        # 返回[0,w)的一个随机数（为列表形式），如果值为0的话不会重新排序，保持原来的形状
        clip_width_points = np.random.randint(low=0, high=w, size=[clip_num], dtype=np.int32).tolist()
        # 将w插入随机数列表中，然后从小到大进行排序[clip_width_points,w]
        # clip_width_points代表将行分段为[0:clip_width_points],...,[clip_width_points:h]
        clip_width_points.append(w)
        clip_width_points.sort()
        # 同样产生一个[0,h)的随机数（为列表形式）
        clip_high_points = np.random.randint(low=0, high=h, size=[clip_num], dtype=np.int32).tolist()
        # 将h插入随机数列表中，然后从小到大进行排序[clip_high_points,h]
        # clip_high_points代表将列分段为[0:clip_high_points],...,[clip_high_points:h]
        clip_high_points.append(h)
        clip_high_points.sort()
        # 对clip_index进行随机排序，clip_index代表将图片分段重排的顺序（如果不进行重新排序，会保持原来的形状）
        random.shuffle(clip_index)
        # result当前填充的下标起始值，初始化为0
        now_index = 0
        # 打乱重排图像的行
        for ind, index in enumerate(clip_index):
            # ind：为clip_index的编号
            # index：为clip_index的值
            # 如果index为0
            if index == 0:
                # value为将第一段插入result的终止下标
                value = clip_width_points[index] + now_index
                result[:, now_index:value, :, :] = copy_images[:, 0:clip_width_points[index], :, :]
                now_index = value
            else:
                # value为将其他段插入result的终止下标
                value = clip_width_points[index] - clip_width_points[index - 1] + now_index
                result[:, now_index:value, :, :] = copy_images[:, clip_width_points[index - 1]:clip_width_points[index], :, :]
                now_index = value
        # 对clip_index进行随机排序，clip_index代表将图片分段重排的顺序
        random.shuffle(clip_index)
        # 将x更新为重排行后的result
        copy_images = np.copy(result)
        # result当前填充的下标起始值，初始化为0
        now_index = 0
        # 打乱重排图像的列
        for ind, index in enumerate(clip_index):
            if index == 0:
                value = clip_high_points[index] + now_index
                result[:, :, now_index:value, :] = copy_images[:, :, 0:clip_high_points[index], :]
                now_index = value
            else:
                value = clip_high_points[index] - clip_high_points[index - 1] + now_index
                result[:, :, now_index:value, :] = copy_images[:, :, clip_high_points[index - 1]:clip_high_points[index], :]
                now_index = value
        # 将result的shape有N*W*H*C变回为N*C*W*H，并转化为tensor形式
        result = torch.from_numpy(np.transpose(result, (0, 3, 1, 2))).float().to(self.device)
        # 返回PGD对抗样本和经过RBS变换的PGD对抗样本
        return pgd_adv_images,result

    # 一次完整的RLFAT对抗训练
    def train_one_epoch_with_rbs_pgd(self, train_loader, epoch):
        # train_loader：训练集
        # epoch：训练次数
        for index, (images, labels) in enumerate(train_loader):
            # nat_images：训练集
            # nat_labels：训练集对应标签
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            self.model.eval()
            # adv_images：产生一组PGD对抗样本和经过RBS变换的PGD对抗样本
            pgd_adv_images,adv_images = self.rbs_pgd_generation(var_natural_images=nat_images, var_natural_labels=nat_labels)
            self.model.train()

            # 预测PGD对抗样本标签
            logits_pgd_adv = self.model(pgd_adv_images)
            # 预测经过RBS变换的PGD对抗样本标签
            logits_adv = self.model(adv_images)
            # 计算经过RBS变换的PGD对抗样本损失
            loss_adv_RLFL = F.cross_entropy(logits_adv, nat_labels)
            # 计算特征转移损失
            # 求logits_pgd_adv与logits_adv的差值平方和
            logits = logits_adv - logits_pgd_adv
            logits = logits.mul(logits)
            sum_logits = torch.sum(logits)
            loss_adv_RLFT = (sum_logits * 1.0) / logits_adv.shape[0]
            # 总的损失函数
            loss = loss_adv_RLFL + self.k * loss_adv_RLFT
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 输出每一次训练的loss
            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \tloss_adv_RLFL={:.4f}, loss_adv_RLFT={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss_adv_RLFL, loss_adv_RLFT, loss), end=' ')

    # RLFAT防御，保存最佳模型，打印ϵ-邻域损失灵敏度和gamma损失灵敏度
    def defense(self, train_loader=None, validation_loader=None):
        # train_loader：训练集
        # validation_loader：验证集
        # best_val_acc：验证集最佳分类精度
        best_val_acc = None
        # 进行num_epochs次PGD对抗训练
        for epoch in range(self.num_epochs):
            # 进行一次完整的RLFAT对抗训练
            self.train_one_epoch_with_rbs_pgd(train_loader=train_loader, epoch=epoch)
            # val_acc：对验证集进行评估的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)

            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            # defense_enhanced_saver：对抗训练网络参数的存放位置为DefenseEnhancedModels/PAT/CIFAR10_PAT_enhanced.pt中
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            # 选取对验证集分类精度最高的模型参数进行保存
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))

            pgd_sense_val = 0
            gamma_sense_val = 0
            count = 0
            # 产生白盒攻击的PGD对抗样本
            for index, (images, labels) in enumerate(validation_loader):
                count = index
                nat_images = images.to(self.device)
                nat_labels = labels.to(self.device)
                self.model.eval()
                # natural_images：将干净样本转化为numpy的形式
                natural_images = nat_images.cpu().numpy()
                # copy_images：复制干净样本numpy形式
                copy_images = natural_images.copy()
                # 改变图像的曝光度
                gamma_copy_images = natural_images.copy()
                gamma_images = batch_brightness(gamma_copy_images, c=-0.15)
                gamma_images = torch.from_numpy(gamma_images).to(self.device)
                # 从[-epsilon,epsilon）的均匀分布中随机采样，形成初始的扰动叠加在干净样本上
                copy_images = copy_images + np.random.uniform(-self.epsilon, self.epsilon, copy_images.shape).astype('float32')
                # 进行迭代，求PGD对抗样本
                x_adv = []
                for i in range(self.attack_step_num):
                    # 将初始的扰动样本由numpy形式转化为tensor形式
                    var_copy_images = torch.from_numpy(copy_images).to(self.device)
                    # 可以对其进行求导
                    var_copy_images.requires_grad = True
                    # 将其输入模型进行预测
                    preds = self.model(var_copy_images)
                    # 计算损失值
                    loss = F.cross_entropy(preds, nat_labels)
                    # 对输入求梯度
                    gradient = torch.autograd.grad(loss, var_copy_images)[0]
                    # 对梯度求符号函数并转为numpy形式
                    gradient_sign = torch.sign(gradient).cpu().numpy()
                    # 对样本添加一小步扰动
                    copy_images = copy_images + self.step_size * gradient_sign
                    # 将样本的扰动大小控制在[natural_images-epsilon,natural_images+epsilon]的范围之内
                    copy_images = np.clip(copy_images, natural_images - self.epsilon, natural_images + self.epsilon)
                    # 将扰动大小控制在[0.0,1.0]之间
                    copy_images = np.clip(copy_images, 0.0, 1.0)
                # 一组PGD对抗样本
                pgd_adv_images = torch.from_numpy(copy_images).to(self.device)
                # 计算ϵ-邻域损失灵敏度
                self.model.eval()
                pgd_preds = self.model(pgd_adv_images)
                # 计算损失值
                pgd_sense_loss = F.cross_entropy(pgd_preds, nat_labels)
                # 对输入求梯度
                pgd_grad = torch.autograd.grad(pgd_sense_loss, pgd_adv_images)[0]
                # 计算一个分组ϵ-邻域损失灵敏度
                pgd_sense = torch.sum(pgd_grad.mul(pgd_grad))
                # 计算所有分组的损失灵敏度之和
                pgd_sense_val = pgd_sense_val + pgd_sense
                # 计算gamma损失灵敏度
                self.model.eval()
                gamma_preds = self.model(gamma_images)
                # 计算损失值
                gamma_sense_loss = F.cross_entropy(gamma_preds, nat_labels)
                # 对输入求梯度
                gamma_grad = torch.autograd.grad(gamma_sense_loss, gamma_images)[0]
                # 计算一个分组的gamma灵敏度
                gamma_sense = torch.sum(pgd_grad.mul(gamma_grad))
                # 计算所有分组的损失灵敏度之和
                gamma_sense_val = gamma_sense_val + gamma_sense

            # 打印ϵ-邻域损失灵敏度和gamma损失灵敏度
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                print('ϵ loss sense val:', pgd_sense_val / (count + 1))
                print('gamma loss sense val:', gamma_sense_val / (count + 1))
            #     x_adv.append(copy_images)
            # # 将多个分组拼接成一个
            # x_adv = np.concatenate(x_adv, axis=0)
            # # PGD对抗样本
            # pgd_adv_images = torch.from_numpy(x_adv).to(self.device)




