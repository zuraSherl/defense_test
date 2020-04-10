import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Defenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_learning_rate
from src.train_test import validation_evaluation

class RAT9Defense(Defense):
    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练模型的超参数
        super(RAT9Defense, self).__init__(model=model, defense_name=defense_name)
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
        # adv_ratio：对抗样本在损失函数中的比例
        # mean：正态分布的均值
        # std：正态分布的标准差
        # eps_mu：产生扰动的均值
        # eps_sigma：产生扰动的标准差
        # clip_eps_min：扰动的最小值
        # clip_eps_max：扰动的最大值
        self.epsilon = kwargs['epsilon']
        self.adv_ratio = kwargs['adv_ratio']
        self.mean = kwargs['mean']
        self.std = kwargs['std']
        self.eps_mu = kwargs['eps_mu']
        self.eps_sigma = kwargs['eps_sigma']
        self.clip_eps_min = kwargs['clip_eps_min']
        self.clip_eps_max = kwargs['clip_eps_max']
        return True

    # 用RAT方法产生对抗样本（每个fgsm对抗样本的扰动值是随机的），返回对抗样本
    def random_fgsm_generation(self, var_natural_images=None):
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
        clone_var_natural_images.requires_grad = True
        # 获取数据集的预测标签
        logits = self.model(clone_var_natural_images)
        rfgsm_labels = torch.max(logits, dim=1)[1]
        # 损失函数
        loss_rfgsm = F.cross_entropy(logits, rfgsm_labels)
        # 对损失函数的输入求导
        gradients_rfgsm = torch.autograd.grad(loss_rfgsm, clone_var_natural_images)[0]
        # 将输入的requires_grad设置为False，表示不能对输入进行求导
        clone_var_natural_images.requires_grad = False
        # 对梯度进行符号函数计算
        gradients_sign = torch.sign(gradients_rfgsm)
        # 将随机扰动转化为numpy数组形式
        var_random_eps = torch.from_numpy(random_eps).float().to(self.device)
        # 产生对抗样本
        with torch.no_grad():
            list_var_adv_images = []
            for i in range(var_natural_images.size(0)):
                # 对每个输入求fgsm对抗样本
                var_adv_image = clone_var_natural_images[i] + var_random_eps[i] * gradients_sign[i]
                # 将对抗样本的每个像素都限制在[0,1]之间
                var_adv_image = torch.clamp(var_adv_image, min=0.0, max=1.0)
                list_var_adv_images.append(var_adv_image)
            # 将对抗样本转化为二维的tensor
            ret_adv_images = torch.stack(list_var_adv_images)
        # 将对抗样本的每个像素都限制在[0,1]之间
        ret_adv_images = torch.clamp(ret_adv_images, min=0.0, max=1.0)
        return ret_adv_images

    # 一次完整的对抗训练（干净样本添加高斯噪声+rfgsm对抗样本）
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
            adv_images = self.random_rfgsm_generation(var_natural_images=nat_images)
            self.model.train()
            # 干净样本添加高斯噪声后的损失函数
            random_space = torch.Tensor(*nat_images.size()).float()
            mean = torch.tensor([self.mean])
            std = torch.tensor([self.std])
            mean = mean.expand(random_space.shape[2], random_space.shape[3], random_space.shape[0],
                               random_space.shape[1]).permute(2, 3, 0, 1)
            std = std.expand(random_space.shape[2], random_space.shape[3], random_space.shape[0],
                             random_space.shape[1]).permute(2, 3, 0, 1)
            random_space = torch.normal(mean, std)
            nat_images = torch.clamp(nat_images + random_space, min=0.0, max=1.0)
            logits_nat = self.model(nat_images)
            loss_nat = F.cross_entropy(logits_nat, nat_labels)  # loss on natural images
            # 对抗样本的损失函数
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)  # loss on the generated adversarial images
            # 最终的损失函数：将两部分的loss按照权重相加再除以权重之和
            loss = (loss_nat + self.adv_ratio * loss_adv) / (1.0 + self.adv_ratio)
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
