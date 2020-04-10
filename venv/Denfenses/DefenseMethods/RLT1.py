import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_learning_rate
from src.train_test import validation_evaluation

class RLT1Defense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练模型的超参数
        super(RLT1Defense, self).__init__(model=model, defense_name=defense_name)
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

        # eps_mu：正态分布的均值
        # eps_sigma：正太分布的方差
        # clip_eps_min：添加扰动的最小值
        # clip_eps_max：添加扰动的最大值
        # epsilon：添加随机扰动的范围
        self.eps_mu = kwargs['mu']
        self.eps_sigma = kwargs['sigma']
        self.clip_eps_min = kwargs['eps_min']
        self.clip_eps_max = kwargs['eps_max']
        self.epsilon = kwargs['epsilon']
        return True

    # 用LLC方法产生对抗样本（每个样本的扰动不同随机产生，所以不能使用攻击里面的对抗性样本），返回对抗样本
    def random_llc_generation(self, var_natural_images=None):
        # var_natural_images：训练集
        # clone_var_natural_images：复制训练集
        # 产生随机数作为每个输入扰动的大小
        self.model.eval()
        clone_var_natural_images = var_natural_images.clone()
        # 随机产生var_natural_images.size(0)个均值为eps_mu，方差为eps_sigma的随机数/255.0
        random_eps = np.random.normal(loc=self.eps_mu, scale=self.eps_sigma, size=[var_natural_images.size(0)]) / 255.0
        # 将产生的随机数大小控制在[clip_eps_min，clip_eps_max]范围内
        random_eps = np.clip(np.abs(random_eps), self.clip_eps_min, self.clip_eps_max)
        # ****************添加的随机扰动*****************
        # 添加一个随机扰动，将扰动范围限制在[0,1]之间
        with torch.no_grad():
            random_distribution = torch.zeros_like(var_natural_images).uniform_(-self.epsilon, self.epsilon).to(
                self.device)
            clone_var_natural_images = torch.clamp(clone_var_natural_images + random_distribution, min=0.0, max=1.0)
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
        # 将随机扰动转化为numpy数组形式
        var_random_eps = torch.from_numpy(random_eps).float().to(self.device)

        # 产生对抗样本
        with torch.no_grad():
            list_var_adv_images = []
            for i in range(var_natural_images.size(0)):
                # 对每个输入求LLC对抗样本
                var_adv_image = clone_var_natural_images[i] - var_random_eps[i] * gradients_sign[i]
                # 将对抗样本的每个像素都限制在[0,1]之间
                var_adv_image = torch.clamp(var_adv_image, min=0.0, max=1.0)
                list_var_adv_images.append(var_adv_image)
            # 将对抗样本转化为二维的tensor
            ret_adv_images = torch.stack(list_var_adv_images)
        # 将对抗样本的每个像素都限制在[0,1]之间
        ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)

        return ret_adv_images

    # 一次完整的对抗训练
    def train_one_epoch_with_adv_and_nat(self, train_loader, epoch):
        # train_loader：训练集
        # epoch： 训练次数
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)
            self.model.eval()
            # 为训练集产生对抗样本
            adv_images = self.random_llc_generation(var_natural_images=nat_images)
            self.model.train()

            # 对抗样本的损失函数
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)  # loss on the generated adversarial images
            # 最终的损失函数：将两部分的loss按照权重相加再除以权重之和
            loss = loss_adv
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
            self.train_one_epoch_with_adv_and_nat(train_loader=train_loader, epoch=epoch)
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
