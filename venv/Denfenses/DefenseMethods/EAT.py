import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Denfenses.DefenseMethods.EAT_External_Models import MNIST_A, MNIST_B, MNIST_C, MNIST_D
from Denfenses.DefenseMethods.EAT_External_Models import CIFAR10_A, CIFAR10_B, CIFAR10_C, CIFAR10_D
from Denfenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_learning_rate
from src.train_test import testing_evaluation, train_one_epoch, validation_evaluation

class EATDefense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：原始模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练参数
        # kwargs：防御参数
        super(EATDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        self.training_parameters = training_parameters
        # Dataset：数据集大写形式，为MNIST或CIFAR10
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        assert self._parsing_parameters(**kwargs)
        # num_epochs：训练次数
        # batch_size：数据集分组大小
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']
        # 为MNIST/CIFAR10的对抗训练准备训练优化器
        if self.Dataset == "MNIST":
            self.optimizer_adv = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
                                           momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        else:
            self.optimizer_adv = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])
    # 封装防御参数
    def _parsing_parameters(self, **kwargs):
        assert kwargs is not None, "the parameters should be specified"
        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))
        # eps：添加随机扰动和FGSM扰动的总扰动大小
        # alpha：添加随机扰动的大小
        self.epsilon = kwargs['eps']
        self.alpha = kwargs['alpha']
        return True

    # 训练预先定义的模型
    def train_external_model_group(self, train_loader=None, validation_loader=None):
        # train_loader：训练集
        # validation_loader：验证集
        # model_group：MNIST/CIFAR10的网络，各有4个
        if self.Dataset == 'MNIST':
            model_group = [MNIST_A(), MNIST_B(), MNIST_C(), MNIST_D()]
        else:
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        model_group = [model.to(self.device) for model in model_group]

        # 一个一个训练MNIST/CIFAR10的网络
        for i in range(len(model_group)):
            # 为MNIST/CIFAR10准备优化器，MNIST网络的优化器都为SGD，CIFAR10的最后一个网络优化器为Adam，其余的都为SGD优化器
            if self.Dataset == "MNIST":
                optimizer_external = optim.SGD(model_group[i].parameters(), lr=self.training_parameters['learning_rate'],
                                               momentum=self.training_parameters['momentum'], weight_decay=self.training_parameters['decay'],
                                               nesterov=True)
            else:
                # 可进行调整
                if i == 3:
                    optimizer_external = optim.SGD(model_group[i].parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
                else:
                    optimizer_external = optim.Adam(model_group[i].parameters(), lr=self.training_parameters['lr'])
            print('\nwe are training the {}-th static external model ......'.format(i))
            # best_val_acc：验证集上的最佳分类精度
            best_val_acc = None
            # 对网络训练num_epochs次
            for index_epoch in range(self.num_epochs):
                # 一次完整的训练
                train_one_epoch(model=model_group[i], train_loader=train_loader, optimizer=optimizer_external, epoch=index_epoch,
                                device=self.device)
                # val_acc：每一次训练结束后，对验证集的分类精度
                val_acc = validation_evaluation(model=model_group[i], validation_loader=validation_loader, device=self.device)
                # 如果数据集为CIFAR10进行学习率调整
                if self.Dataset == 'CIFAR10':
                    adjust_learning_rate(epoch=index_epoch, optimizer=optimizer_external)
                assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
                # defense_external_saver：最佳模型参数保存位置为DefenseEnhancedModels/EAT/MNIST_EAT_0.pt中，或者为CIFAR10
                defense_external_saver = '../DefenseEnhancedModels/{}/{}_EAT_{}.pt'.format(self.defense_name, self.Dataset, str(i))
                # 计算每次训练对验证集的分类精度，将最佳的验证精度对应的模型参数进行保存
                if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                    if best_val_acc is not None:
                        os.remove(defense_external_saver)
                    best_val_acc = val_acc
                    model_group[i].save(name=defense_external_saver)
                else:
                    print('Train Epoch {:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(index_epoch, best_val_acc))

    # 加载预训练模型，返回加载的模型
    def load_external_model_group(self, model_dir='../DefenseEnhancedModels/EAT/', test_loader=None):
        # model_dir：预训练模型参数的保存位置为/DefenseEnhancedModels/EAT/
        # test_loader：测试集
        print("\n!!! Loading static external models ...")
        # Set up 4 static external models
        # model_group：获取MNIST/CIFAR10的4个模型
        if self.Dataset == 'MNIST':
            model_group = [MNIST_A(), MNIST_B(), MNIST_C(), MNIST_D()]
        else:
            model_group = [CIFAR10_A(), CIFAR10_B(), CIFAR10_C(), CIFAR10_D()]
        # 将这些模型加载到device上
        model_group = [model.to(self.device) for model in model_group]
        # 循环遍历模型
        for i in range(len(model_group)):
            print('loading the {}-th static external model'.format(i))
            # model_path：模型所在位置为DefenseEnhancedModels/EAT/MNIST_EAT_0.pt或者为CIFAR10
            model_path = '{}{}_EAT_{}.pt'.format(model_dir, self.Dataset, str(i))
            assert os.path.exists(model_path), "please train the external model first!!!"
            # 加载模型
            model_group[i].load(path=model_path, device=self.device)
            # 对测试集进行分类
            testing_evaluation(model=model_group[i], test_loader=test_loader, device=self.device)
        # 返回加载的模型
        return model_group

    # RFGSM产生对抗样本，首先添加随机扰动，然后用FGSM产生对抗样本返回
    def random_fgsm_generation(self, model=None, natural_images=None):
        # model：模型
        # natural_images：干净样本
        # attack_model：被攻击的模型
        attack_model = model.to(self.device)
        attack_model.eval()
        # 首先对干净样本添加随机扰动
        with torch.no_grad():
            # random_sign：产生和输入样本大小相同的一组服从(0,1)标准正态分布的随机数
            random_sign = torch.sign(torch.randn(*natural_images.size())).to(self.device)
            # new_images：将随机扰动*alpha（随机扰动的权重），然后添加到干净样本上并将像素范围控制在[0.0,1.0]之间
            new_images = torch.clamp(natural_images + self.alpha * random_sign, min=0.0, max=1.0)
        # 可以对new_images进行求导
        new_images.requires_grad = True
        # logits_attack：将new_images输入到模型中得到分类结果
        logits_attack = attack_model(new_images)
        # labels_attack：获取分类结果标签
        labels_attack = torch.max(logits_attack, dim=1)[1]
        # loss_attack：计算损失值（为了避免标签泄漏，将真实标签用预测标签进行替代）
        loss_attack = F.cross_entropy(logits_attack, labels_attack)
        # 计算new_images的梯度
        gradient = torch.autograd.grad(loss_attack, new_images)[0]
        # 将new_images设置为不可求导
        new_images.requires_grad = False
        # FGSM产生对抗样本
        with torch.no_grad():
            # 将梯度方向*(epsilon-alpha)作为添加的扰动加到new_images上
            xs_adv = new_images + (self.epsilon - self.alpha) * torch.sign(gradient)
            # 将像素的值限制在[0.0,1.0]之间，产生最终的对抗样本
            xs_adv = torch.clamp(xs_adv, min=0.0, max=1.0)
        # 返回产生的对抗样本
        return xs_adv

    # 一次完整的集成对抗训练，每一个训练批次中包含一半干净样本，一半预训练模型或者原始模型产生的对抗样本
    def train_one_epoch_with_adv_from_external_models(self, pre_trained_models=None, train_loader=None, epoch=None):
        # pre_trained_models：预训练模型
        # train_loader：训练集
        # epoch：训练的次数（第几次训练）
        for index, (images, labels) in enumerate(train_loader):
            # nat_images：获取干净训练集
            # nat_labels：获取干净训练集的标签
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            # idx：随机产生0,1,2,3,4五个数中的一个
            idx = np.random.randint(5)
            # 如果idx为0，则被攻击的模型为原始的MNIST/CIFAR10模型
            if idx == 0:
                attacking_model = self.model
            # 否则被攻击的模型为预训练的4个模型中的一个
            else:
                attacking_model = pre_trained_models[idx - 1]
            # adv_images：利用RFGSM方法攻击attacking_model模型产生对抗样本
            adv_images = self.random_fgsm_generation(model=attacking_model, natural_images=nat_images)
            # 在原始模型中进行训练
            self.model.train()
            # logits_nat：干净样本经过模型的预测结果
            logits_nat = self.model(nat_images)
            # loss_nat：计算干净样本的loss
            loss_nat = F.cross_entropy(logits_nat, nat_labels)
            # logits_adv：对抗样本经过模型的预测结果
            logits_adv = self.model(adv_images)
            # loss_adv：计算对抗样本的loss
            loss_adv = F.cross_entropy(logits_adv, nat_labels)
            # loss：干净样本和对抗样本的总loss，权重各占0.5
            loss = 0.5 * (loss_nat + loss_adv)
            # 反向传播
            self.optimizer_adv.zero_grad()
            loss.backward()
            self.optimizer_adv.step()
            # 输出loss
            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \tloss_nat={:.4f}, loss_adv={:.4f}, total_loss={:.4f} ===> '. \
                  format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss_nat, loss_adv, loss), end=' ')

    # 集成对抗训练防御，保存最佳模型
    def defense(self, pre_trained_models=None, train_loader=None, validation_loader=None):
        # pre_trained_models：预训练模型
        # train_loader：训练集
        # validation_loader：验证集
        # best_val_acc：最佳测试集分类精度
        best_val_acc = None
        # 进行num_epochs次集成对抗训练
        for epoch in range(self.num_epochs):
            # training the model with natural examples and corresponding adversarial examples from external models
            # 一次完整的集成对抗训练
            self.train_one_epoch_with_adv_from_external_models(pre_trained_models=pre_trained_models, train_loader=train_loader, epoch=epoch)
            # 计算训练后的模型对验证集的分类精度
            val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            # 如果数据集为CIFAR10则进行学习率的调整
            if self.Dataset == 'CIFAR10':
                adjust_learning_rate(epoch=epoch, optimizer=self.optimizer_adv)
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            # defense_enhanced_saver：集成对抗训练防御模型参数的保存位置在DefenseEnhancedModels/EAT/MNIST_EAT_enhanced.pt中或者CIFAR10
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            # 选择验证集上分类精度最高的模型进行保存
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(defense_enhanced_saver)
                best_val_acc = val_acc
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch {:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))

