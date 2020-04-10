import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_learning_rate
from src.train_test import validation_evaluation

class RAT6Defense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练模型的超参数
        super(RAT6Defense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        # Dataset：数据集名称（大写），必须为MNIST或CIFAR10
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        assert self._parsing_parameters(**kwargs)

        # num_epochs：训练次数
        # batch_size：分组大小
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']
        self.lr_schedule = lambda t:np.interp([t], [0, self.num_epochs * 2 // 5, self.num_epochs], [0, training_parameters['lr'], 0])[0]

        # 构造MNIST/CIFAR10的优化器
        # **************lr的值需要修改**************
        if self.Dataset == 'MNIST':
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                       momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])

    # 设置对抗训练的参数
    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))

        # epsilon：扰动范围
        # alpha：fgsm扰动大小
        # adv_ration：两种对抗样本所占损失函数的比例
        self.epsilon = kwargs['epsilon']
        self.alpha = kwargs['alpha']
        self.adv_ration = kwargs['adv_ration']
        return True

    # 用LLC方法产生对抗样本（每个llc对抗样本添加的扰动相同），返回对抗样本
    def random_llc_generation(self, var_natural_images=None):
        # var_natural_images：训练集
        # clone_var_natural_images：复制训练集

        # 产生随机数作为每个输入扰动的大小
        self.model.eval()
        clone_var_natural_images = var_natural_images.clone()
        # 损失函数对输入求梯度，并使用符号函数
        # 将输入的requires_grad设置为True，表示可以对输入进行求导
        clone_var_natural_images.requires_grad = True
        # 获取数据集的预测标签
        logits = self.model(clone_var_natural_images)
        # 获取每一行的最小值的下标（最不可能的分类标签）
        llc_labels = torch.min(logits, dim=1)[1]
        # 损失函数
        loss_llc = F.cross_entropy(logits, llc_labels)
        # 对损失函数的输入求导
        gradients_llc = torch.autograd.grad(loss_llc, clone_var_natural_images)[0]
        # 将输入的requires_grad设置为False，表示不能对输入进行求导
        clone_var_natural_images.requires_grad = False
        # 对梯度进行符号函数计算
        gradients_sign = torch.sign(gradients_llc)
        # 产生对抗样本,将对抗样本的每个像素都限制在[0,1]之间
        with torch.no_grad():
            ret_adv_images = var_natural_images - self.epsilon * gradients_sign
            ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)
        return ret_adv_images

    # 用RAT方法产生对抗样本（每个fgsm对抗样本添加的扰动相同），返回对抗样本
    def random_fgsm_generation(self, var_natural_images=None):
        # 产生随机数作为每个输入扰动的大小
        model = self.model.to(self.device)
        model.eval()
        # ****************添加的随机扰动*****************
        # 添加一个随机扰动，将扰动范围限制在[0,1]之间
        with torch.no_grad():
            random_distribution = torch.zeros_like(var_natural_images).uniform_(-self.epsilon, self.epsilon).to(
                self.device)
            new_images = torch.clamp(var_natural_images + random_distribution, min=0.0, max=1.0)
        # 损失函数对输入求梯度，并使用符号函数
        new_images.requires_grad = True
        # 获取数据集的预测标签
        logits = model(new_images)
        # 获取每一行的最小值的下标（最不可能的分类标签）
        rfgsm_labels = torch.max(logits, dim=1)[1]
        # 损失函数
        loss_rfgsm = F.cross_entropy(logits, rfgsm_labels)
        # 对损失函数的输入求导
        gradients_rfgsm = torch.autograd.grad(loss_rfgsm, new_images)[0]
        # 将输入的requires_grad设置为False，表示不能对输入进行求导
        new_images.requires_grad = False
        # 对梯度进行符号函数计算，产生对抗样本
        with torch.no_grad():
            ret_adv_images = new_images + self.alpha * torch.sign(gradients_rfgsm)
            ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)
        return ret_adv_images

    # 一次完整的对抗训练（llc对抗样本+rfgsm对抗样本）
    def train_one_epoch_with_adv(self, train_loader, epoch):
        # train_loader：训练集
        # epoch： 训练次数
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            # 采用循环学习率
            lr = self.lr_schedule(epoch + (i + 1) / len(train_loader))
            self.optimizer.param_groups[0].update(lr=lr)
            self.model.eval()
            # 为训练集产生对抗样本
            rfgsm_adv_images = self.random_rfgsm_generation(var_natural_images=nat_images)
            llc_adv_images = self.random_llc_generation(var_natural_images=nat_images)
            self.model.train()
            # rfgsm对抗样本的损失函数
            logits_rfgsm_adv = self.model(rfgsm_adv_images)
            loss_rfgsm_adv = F.cross_entropy(logits_rfgsm_adv, nat_labels)
            # llc对抗样本的损失函数
            logits_llc_adv = self.model(llc_adv_images)
            loss_llc_adv = F.cross_entropy(logits_llc_adv, nat_labels)  # loss on the generated adversarial images
            # 最终的损失函数：将两部分的loss按照权重相加再除以权重之和
            loss = (loss_rfgsm_adv + self.adv_ratio * loss_llc_adv) / (1.0 + self.adv_ratio)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, index, len(train_loader), loss_nat, loss_adv, loss), end=' ')

    def defense(self, train_loader=None, validation_loader=None):
        # train_loader：训练集
        # validation_loader：验证集
        # best_val_acc：验证集上的最佳分类精度
        best_val_acc = None
        # 进行epoch次训练
        for epoch in range(self.num_epochs):
            self.train_one_epoch_with_adv(train_loader=train_loader, epoch=epoch)
            # 计算每一次训练后在验证集上的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            # 将最佳模型参数保存到DefenseEnhancedModels/RAT/CIFAR10_RAT_enhanced.pt中或MNIST
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
