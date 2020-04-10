import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Denfenses.DefenseMethods.defenses import Defense
from src.train_mart_mnist import adjust_MNIST_learning_rate
from src.train_mart_cifar10 import adjust_CIFAR10_learning_rate
from src.train_test import validation_evaluation

class NEW_MART1Defense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练参数
        # kwargs：防御参数
        super(NEW_MART1Defense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        # 将数据集名称转化为大写的形式
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        assert self._parsing_parameters(**kwargs)
        # num_epochs：获取训练次数
        # batch_size：获取分组大小
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']
        # 准备MNIST/CIFAR10的优化器
        self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                       momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
    # 封装防御参数
    def _parsing_parameters(self, **kwargs):
        assert kwargs is not None, "the parameters should be specified"
        print("\nparsing the user configuration for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs.get(key)))
        # attack_step_num：PGD迭代次数
        # step_size：每一步攻击的扰动步长
        # epsilon：最大扰动
        # lamda：正则化项的系数
        self.attack_step_num = kwargs.get('attack_step_num')
        self.step_size = kwargs.get('step_size')
        self.epsilon = kwargs.get('epsilon')
        self.lamda = kwargs.get('lamda')
        return True

    # 产生PGD对抗样本
    def pgd_generation(self, var_natural_images=None, var_natural_labels=None):
        # var_natural_images：干净样本
        # var_natural_labels：干净样本对应的标签
        self.model.eval()
        # natural_images：将干净样本转化为numpy的形式
        natural_images = var_natural_images.cpu().numpy()
        # copy_images：复制干净样本numpy形式
        copy_images = natural_images.copy()
        # 从[-epsilon,epsilon）的均匀分布中随机采样，形成初始的扰动叠加在干净样本上
        copy_images = copy_images + np.random.uniform(-self.epsilon, self.epsilon, copy_images.shape).astype('float32')
        # 进行迭代
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
        # 返回最后copy_images的tensor形式，即为PGD对抗样本
        return torch.from_numpy(copy_images).to(self.device)

    # 一次完整的PGD对抗训练
    def train_one_epoch_with_pgd_and_nat(self, train_loader, epoch):
        # train_loader：训练集
        # epoch：训练次数
        for index, (images, labels) in enumerate(train_loader):
            # nat_images：训练集
            # nat_labels：训练集对应标签
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            self.model.eval()
            # adv_images：产生PGD对抗样本
            adv_images = self.pgd_generation(var_natural_images=nat_images, var_natural_labels=nat_labels)
            self.model.train()
            # 预测训练集标签
            logits_nat = self.model(nat_images)
            # 计算训练集损失
            # loss_nat = F.cross_entropy(logits_nat, nat_labels)
            # 预测PGD对抗样本标签
            logits_adv = self.model(adv_images)
            # 计算PGD对抗样本损失，MART损失函数
            # 第一个指标函数BCE
            loss_adv1 = F.cross_entropy(logits_adv, nat_labels)
            softm_adv = F.softmax(logits_adv, dim=1)
            inf = -float('inf')
            for i in range(logits_adv.shape[0]):
                logits_adv[i][nat_labels[i]] = inf
            _, predicted = torch.max(logits_adv.data, 1)
            predicted = predicted.numpy()
            one_hot_labels = []
            for label in predicted:
                one_hot_label = [0 for i in range(10)]
                one_hot_label[label] = 1
                one_hot_labels.append(one_hot_label)
            one_hot_labels = np.array(one_hot_labels)
            one_hot_labels = torch.from_numpy(one_hot_labels)
            log_one = torch.sum(torch.mul(one_hot_labels, softm_adv), dim=1)
            log_one = torch.ones(log_one.shape) - log_one
            log_likelihood = -torch.log(log_one)
            loss_adv11 = torch.mean(log_likelihood)
            loss_adv_bce = loss_adv1 + loss_adv11
            # 第二个指标函数KL
            softm_nat = F.softmax(logits_nat, dim=1)
            nat_div_adv = torch.div(softm_nat, softm_adv)
            log_nat_div_adv = torch.log(nat_div_adv)
            kl = torch.mul(softm_nat, log_nat_div_adv)
            KL_loss = torch.sum(kl, dim=1)
            # 第三个指标函数
            one_hot_nat_labels = []
            for nat_label in nat_labels:
                one_hot_nat_label = [0 for i in range(10)]
                one_hot_nat_label[nat_label] = 1
                one_hot_nat_labels.append(one_hot_nat_label)
            one_hot_nat_labels = np.array(one_hot_nat_labels)
            one_hot_nat_labels = torch.from_numpy(one_hot_nat_labels)
            pyi = torch.sum(torch.mul(softm_nat, one_hot_nat_labels), dim=1)
            loss3 = torch.ones(pyi.shape) - pyi
            # 第四个指标函数
            loss_nat1 = F.cross_entropy(logits_nat, nat_labels)
            for i in range(logits_nat.shape[0]):
                logits_nat[i][nat_labels[i]] = inf
            m, predicted_nat = torch.max(logits_nat.data, 1)
            predicted_nat = predicted_nat.numpy()
            one_hot_labels1 = []
            for label1 in predicted_nat:
                one_hot_label1 = [0 for i in range(10)]
                one_hot_label1[label1] = 1
                one_hot_labels1.append(one_hot_label1)
            one_hot_labels1 = np.array(one_hot_labels1)
            one_hot_labels1 = torch.from_numpy(one_hot_labels1)
            log_one1 = torch.sum(torch.mul(one_hot_labels1, softm_nat), dim=1)
            log_one1 = torch.ones(log_one1.shape) - log_one1
            log_likelihood1 = -torch.log(log_one1)
            loss_nat11 = torch.mean(log_likelihood1)
            loss_nat_bce = loss_nat1 + loss_nat11
            # 总的损失函数
            loss_KL_pyi = torch.mul(KL_loss,loss3)
            loss_KL_pyi = torch.mean(loss_KL_pyi)
            loss3 = torch.mean(loss3)
            loss = loss_adv_bce + self.lamda * (loss_KL_pyi + loss3 * loss_nat_bce)
            # 干净样本和对抗样本的总loss
            # loss = 0.5 * (loss_nat + loss_adv)
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 输出每一次训练的loss
            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss_nat, loss_adv, loss), end=' ')

    # PAT防御
    def defense(self, train_loader=None, validation_loader=None):
        # train_loader：训练集
        # validation_loader：验证集
        # best_val_acc：验证集最佳分类精度
        best_val_acc = None
        # 进行num_epochs次PGD对抗训练
        for epoch in range(self.num_epochs):
            # 进行一次完整PGD对抗训练
            self.train_one_epoch_with_pgd_and_nat(train_loader=train_loader, epoch=epoch)
            # val_acc：对验证集进行评估的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            # 进行学习率调整
            if self.Dataset == 'CIFAR10':
                adjust_CIFAR10_learning_rate(epoch=epoch, optimizer=self.optimizer)
            else:
                adjust_MNIST_learning_rate(epoch=epoch, optimizer=self.optimizer)
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
