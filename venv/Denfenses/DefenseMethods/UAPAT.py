import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_learning_rate
from src.train_test import validation_evaluation
from attacks.attack_methods.UAP import UniversalAttack

class UAPATDefense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练模型的超参数
        super(UAPATDefense, self).__init__(model=model, defense_name=defense_name)
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

        # 构造MNIST/CIFAR10的优化器
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

        # fool_rate：UAP扰动要达到的错误率
        # epsilon：UAP添加的最大扰动值
        # max_iter_universal：最大的UAP迭代次数
        # overshoot：DeepFool添加的扰动大小
        # max_iter_deepfool：DeepFool的最大迭代次数
        self.fool_rate = kwargs['fool_rate']
        self.epsilon = kwargs['epsilon']
        self.max_iter_universal = kwargs['max_iter_universal']
        self.overshoot = kwargs['overshoot']
        self.max_iter_deepfool = kwargs['max_iter_deepfool']
        return True

    # 用UAP方法产生对抗样本，返回通用对抗扰动
    def uap_generation(self, dataset=None, validation=None):
        attacker = UniversalAttack(model=self.model, fooling_rate=self.fool_rate,
                                   max_iter_universal=self.max_iter_universal,
                                   epsilon=self.epsilon, overshoot=self.overshoot, max_iter_deepfool=self.max_iter_deepfool)
        universal_perturbation = attacker.universal_perturbation(dataset=dataset, validation=validation,
                                                                 device=self.device)
        universal_perturbation = universal_perturbation.cpu().numpy()
        return universal_perturbation

    # 一次完整的对抗训练
    def train_one_epoch_with_adv_and_nat(self, train_loader, uap_train_loader, uap_valid_loader, epoch):
        # train_loader：训练集
        # valid_loader：验证集
        # epoch： 训练次数
        universal_perturbation = self.uap_generation(dataset=uap_train_loader, validation=uap_valid_loader)
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            self.model.eval()
            # 为训练集产生对抗样本
            adv_images = nat_images + universal_perturbation
            self.model.train()
            # 干净样本的损失函数
            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)  # loss on natural images
            # 对抗样本的损失函数
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)  # loss on the generated adversarial images
            # 最终的损失函数：将两部分的loss按照权重相加再除以权重之和
            loss = 0.5 * (loss_nat + loss_adv)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, index, len(train_loader), loss_nat, loss_adv, loss), end=' ')

    def defense(self, train_loader=None, validation_loader=None, uap_train_loader=None, uap_validation_loader=None):
        # train_loader：训练集
        # validation_loader：验证集
        # best_val_acc：验证集上的最佳分类精度
        best_val_acc = None
        # 进行epoch次训练
        for epoch in range(self.num_epochs):
            self.train_one_epoch_with_adv_and_nat(train_loader=train_loader, uap_train_loader=uap_train_loader, uap_valid_loader=uap_validation_loader, epoch=epoch)
            # 计算每一次训练后在验证集上的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            # 如果是CIFAR10数据集，则调整模型参数
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer)
            # 将最佳模型参数保存到DefenseEnhancedModels/NAT/CIFAR10_NAT_enhanced.pt中或MNIST
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
