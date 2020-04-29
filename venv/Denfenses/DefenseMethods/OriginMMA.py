import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
from collections import OrderedDict

import os
import time
import copy
import numpy as np
import torch.optim as optim
from Denfenses.DefenseMethods.defenses import Defense
from src.train_cifar10 import adjust_mma_learning_rate

# 为训练集/测试集的分组进行编号(0~len(loader.targets)-1)
def add_indexes_to_loader(loader):
    # loader：已经分组的训练集/测试集
    dataset = loader.dataset
    # 如果dataset是存在于子数据集SubSet中
    while isinstance(dataset, Subset):  # XXX: there might be multiple layers
        dataset = dataset.dataset
    # 如果为训练集
    if dataset.train:
        # XXX: if statements for old torchvision
        # 如果train_labels在dataset字典中，令targets为训练集标签，而dataset.train_labels为编号
        if "train_labels" in dataset.__dict__:
            targets = dataset.train_labels
            dataset.train_labels = torch.arange(len(targets))
        # 否则令targets为训练集标签，而dataset.targets为编号
        else:
            targets = dataset.targets
            dataset.targets = torch.arange(len(targets))
    # 如果为测试集与训练集相同处理方式
    else:
        # XXX: if statements for old torchvision
        if "test_labels" in dataset.__dict__:
            targets = dataset.test_labels
            dataset.test_labels = torch.arange(len(targets))
        else:
            targets = dataset.targets
            dataset.targets = torch.arange(len(targets))
    # 将数据集的标签设置为targets，获取某个标签则为targets[dataset.targets]/targets[dataset.train_labels]
    loader.targets = torch.tensor(targets)

# 返回batch_tensor=vector的对应值*batch_tensor的对应值
def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

# 返回tensor = float_or_vector * tensor
def batch_multiply(float_or_vector, tensor):
    # 如果float_or_vector是tensor类型
    if isinstance(float_or_vector, torch.Tensor):
        # 检查float_or_vector和tensor的大小是否相同
        assert len(float_or_vector) == len(tensor)
        # 如果相同则对应相乘
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    # 如果float_or_vector是float类型
    elif isinstance(float_or_vector, float):
        # 则float_or_vector直接乘tensor
        tensor *= float_or_vector
    # 如果float_or_vector为其他类型，则报错
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    # 返回更新后的tensor
    return tensor

# 裁剪输入input到[min,max]范围内
def clamp(input, min=None, max=None):
    # input：输入
    # min：最小值
    # max：最大值
    # 输入的维数N*C*H*W的话就为4
    ndim = input.ndimension()
    # 如果min为None则结束条件
    if min is None:
        pass
    # 如果min的类型是float或int
    elif isinstance(min, (float, int)):
        # 将input的最小值裁剪到min
        input = torch.clamp(input, min=min)
    # 如果min的类型是tensor
    elif isinstance(min, torch.Tensor):
        # 如果min的shape为C*H*W，input的shape为N*C*H*W
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            # 将input中每个像素的最小值裁剪到min
            input = torch.max(input, min.view(1, *min.shape))
        # 否则
        else:
            # 检查min的shape是否和input的shape相同
            assert min.shape == input.shape
            # 若相同则将input中每个像素的最小值裁剪到min
            input = torch.max(input, min)
    # 如果min为其他类型，则报错
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    # 如果max为None则结束条件
    if max is None:
        pass
    # 如果max的类型是float或int
    elif isinstance(max, (float, int)):
        # 将input的最大值裁剪到min
        input = torch.clamp(input, max=max)
    # 如果max的类型是tensor
    elif isinstance(max, torch.Tensor):
        # 如果max的shape为C*H*W，input的shape为N*C*H*W
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            # 将input中每个像素的最da值裁剪到max
            input = torch.min(input, max.view(1, *max.shape))
        # 否则
        else:
            # 检查max的shape是否和input的shape相同
            assert max.shape == input.shape
            # 若相同则将input中每个像素的最大值裁剪到max
            input = torch.min(input, max)
    # 如果max为其他类型，则报错
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    # 返回裁剪后的input
    return input

# 返回logit输出对应的标签
def predict_from_logits(logits, dim=1):
    # logits：网络的logit输出
    # 返回logits输出最大值对应的标签
    return logits.max(dim=dim, keepdim=False)[1]

# 返回预测的正确率
def get_accuracy(pred, target):
    # pred：预测标签
    # target：真实标签
    # 返回预测的正确率
    return pred.eq(target).float().mean().item()

# 损失函数
# 损失的不同类型（原损失、平均损失和损失求和）
def _reduce_loss(loss, reduction):
    # 如果reduction为none，则返回loss本身
    if reduction == 'none':
        return loss
    # 如果reduction为elementwise_mean，则返回loss的平均值
    elif reduction == 'elementwise_mean':
        return loss.mean()
    # 如果reduction为sum，则返回loss的和
    elif reduction == 'sum':
        return loss.sum()
    # 否则报错
    else:
        raise ValueError(reduction + " is not valid")

# LM损失（非softmax损失）
def elementwise_margin(logits, label):
    # logits：logit输出
    # label：真实标签
    # 多少个图像的logits输出
    batch_size = logits.size(0)
    # 获取logit输出最大的两个值和对应的下标
    topval, topidx = logits.topk(2, dim=1)
    # 如果top1输出等于真实标签，则取top2的值(获取最近的误分类的logits)，否则取top1的值
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    # 返回maxelse-真实标签对应的logits值（maxj≠ylogits(j)-logits(y)）
    return maxelse - logits[torch.arange(batch_size), label]

# 论文中的LM损失（LM softmax损失）
def logit_margin_loss(input, target, reduction='elementwise_mean', offset=0.):
    # input：预测的logits输出
    # target：真实标签
    # reduction：loss类型，这里为elementwise_mean，求loss平均值
    # offset：偏差
    # 获取最小边距
    loss = elementwise_margin(input, target)
    # 返回logit损失+偏差
    return _reduce_loss(loss, reduction) + offset

# 论文中的SLM损失，用于替代LM损失
def soft_logit_margin_loss(logits, targets, reduction='elementwise_mean', offset=0.):
    # logits：预测的logits输出
    # targets：真实标签
    # reduction：loss类型，这里为elementwise_mean，求loss平均值
    # offset：偏差
    # 输入图像个数
    batch_size = logits.size(0)
    # 分类数
    num_class = logits.size(1)
    # 与logits形状相同的全为1的tensor
    mask = torch.ones_like(logits).byte()
    # TODO: need to cover different versions of torch
    # mask = torch.ones_like(logits).bool()
    # 令真实标签对应位置的mask中的值为0
    mask[torch.arange(batch_size), targets] = 0
    # 真实标签对应的logits值
    logits_true_label = logits[torch.arange(batch_size), targets]
    # 去除真实标签对应的logits值后的新的logits
    logits_other_label = logits[mask].reshape(batch_size, num_class - 1)
    # 损失=logsumexp非真实标签减去真实标签对应的logits
    loss = torch.logsumexp(logits_other_label, dim=1) - logits_true_label
    # 返回SLM损失+偏差值
    return _reduce_loss(loss, reduction) + offset

# CW损失
def cw_loss(input, target, reduction='elementwise_mean'):
    # input：预测的logits输出
    # target：真实标签
    # reduction：loss类型，这里为elementwise_mean，求loss平均值
    # 将最小边距的平均损失+50，最小值裁剪为0.0
    loss = clamp(elementwise_margin(input, target) + 50, 0.)
    # 返回cw损失
    return _reduce_loss(loss, reduction)

# LM损失
class LogitMarginLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean', offset=0.):
        super(LogitMarginLoss, self).__init__(size_average, reduce, reduction)
        # offset：偏差
        self.offset = offset
    # 返回LM损失
    def forward(self, input, target):
        return logit_margin_loss(input, target, reduction=self.reduction, offset=self.offset)

# SLM损失
class SoftLogitMarginLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean', offset=0.):
        super(SoftLogitMarginLoss, self).__init__(size_average, reduce, reduction)
        # offset：偏差
        self.offset = offset
    # 返回SLM损失
    def forward(self, logits, targets):
        return soft_logit_margin_loss(logits, targets, reduction=self.reduction, offset=self.offset)

# CW损失
class CWLoss(_Loss):
    # TODO: combine with the CWLoss in advertorch.utils
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(CWLoss, self).__init__(size_average, reduce, reduction)
    # 返回CW损失
    def forward(self, input, target):
        return cw_loss(input, target, reduction=self.reduction)

# 获取损失函数
def get_loss_fn(name, reduction):
    # 交叉熵损失函数
    if name == "xent":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    # SLM损失函数
    elif name == "slm":
        from advertorch.loss import SoftLogitMarginLoss
        loss_fn = SoftLogitMarginLoss(reduction=reduction)
    # LM损失函数
    elif name == "lm":
        from advertorch.loss import LogitMarginLoss
        loss_fn = LogitMarginLoss(reduction=reduction)
    # CW损失函数
    elif name == "cw":
        from advertorch.loss import CWLoss
        loss_fn = CWLoss(reduction=reduction)
    # 否则报错
    else:
        raise NotImplementedError("loss_fn={}".format(name))
    # 返回损失函数
    return loss_fn

# 返回损失和
def get_sum_loss_fn(name):
    return get_loss_fn(name, "sum")

# 返回平均损失
def get_mean_loss_fn(name):
    return get_loss_fn(name, "elementwise_mean")

# 直接返回损失
def get_none_loss_fn(name):
    return get_loss_fn(name, "none")

# 搜索函数，返回ANPGD最佳扰动
def bisection_search(cur_eps, ptb, model, data, label, fn_margin, margin_init,maxeps, num_steps,
        cur_min=None, clip_min=0., clip_max=1.):
    # cur_eps：pgd攻击的扰动范围
    # ptb：pgd攻击的扰动方向
    # model：模型
    # data：干净数据集
    # label：干净数据集对应标签
    # fn_margin：损失函数（默认为slm损失函数，返回原损失）
    # margin_init：pgd对抗样本的损失（默认为slm损失函数，返回原损失）
    # maxeps：最大攻击扰动，为论文中的1.05*dmax(形状与y相同)
    # num_steps：搜索次数
    # cur_min：当前扰动的最小值，默认为None
    # clip_min：裁剪的最小值，默认为0.0
    # clip_max：裁剪的最大值，默认为1.0
    # 检查当前扰动是否小于最大扰动
    assert torch.all(cur_eps <= maxeps)
    # 对抗样本的损失（刚开始为pgd对抗样本的损失，形状与y相同）
    margin = margin_init
    # 初始化当前最小扰动的值为0，shape与y相同
    if cur_min is None:
        cur_min = torch.zeros_like(margin)
    # 最大扰动值与maxeps相同，shape与y相同
    cur_max = maxeps.clone().detach()

    # 进行num_steps次搜索
    for ii in range(num_steps):
        # margin < 0代表正确分类,cur_min=max[cur_eps,cur_min],cur_max=min[maxeps,cur_max]
        # 与论文中的[cur_eps,maxeps]相同
        # margin >= 0代表错误分类,cur_min=max[0,cur_min],cur_max=min[cur_eps,cur_max]
        # 与论文中的[0,cur_eps]相同
        cur_min = torch.max((margin < 0).float() * cur_eps, cur_min)
        cur_max = torch.min(((margin < 0).float() * maxeps + (margin >= 0).float() * cur_eps), cur_max)
        # 计算当前添加的扰动大小，更新cur_eps，若正确分类则增大cur_eps的值，错误分类则减小cur_eps的值
        cur_eps = (cur_min + cur_max) / 2
        # 计算添加扰动后的对抗样本的slm损失，更新margin
        margin = fn_margin(model(clamp(data + batch_multiply(cur_eps, ptb), min=clip_min, max=clip_max)), label)
    # 检查当前添加的扰动是否小于最大扰动dmax
    assert torch.all(cur_eps <= maxeps)
    # 返回最佳扰动
    return cur_eps

# 复制梯度
class GradCloner(object):
    def __init__(self, model, optimizer):
        # model：模型
        # optimizer：优化器
        self.model = model
        self.optimizer = optimizer
        # clone_model：复制模型
        self.clone_model = copy.deepcopy(model)
        # clone_optimizer：设置复制优化器为SGD，lr=0.0
        self.clone_optimizer = optim.SGD(self.clone_model.parameters(), lr=0.)

    # 复制和清空梯度
    def copy_and_clear_grad(self):
        # 初始化复制优化器的梯度为0
        self.clone_optimizer.zero_grad()
        # 将复制优化器的梯度赋值为原始优化器的梯度
        for (pname, pvalue), (cname, cvalue) in zip(self.model.named_parameters(), self.clone_model.named_parameters()):
            cvalue.grad = pvalue.grad.clone()
        # 将原始优化器的梯度清空
        self.optimizer.zero_grad()

    # 更新原始优化器的梯度为组合梯度
    def combine_grad(self, alpha=1, beta=1):
        # alpha：原始优化器梯度的系数，默认为1
        # beta：复制优化器梯度的系数，默认为1
        for (pname, pvalue), (cname, cvalue) in zip(self.model.named_parameters(), self.clone_model.named_parameters()):
            # 原始优化器梯度 = alpha*原始优化器的梯度 + beta*复制优化器的梯度
            pvalue.grad.data = \
                alpha * pvalue.grad.data + beta * cvalue.grad.data

# 计算平均精度
class AverageMeter(object):
    """Computes and stores the average and current value"""

    # copied from https://github.com/pytorch/examples

    # 初始化
    def __init__(self):
        self.reset()
    # 重置
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 更新，求添加val后的平均值
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 初始化epoch损失、epoch精度、disp损失和disp精度
def init_loss_acc_meter():
    meter = {}
    meter["epoch_loss"] = AverageMeter()
    meter["epoch_acc"] = AverageMeter()
    meter["disp_loss"] = AverageMeter()
    meter["disp_acc"] = AverageMeter()
    return meter

# 初始化epoch扰动和disp扰动
def init_eps_meter():
    meter = {}
    meter["epoch_eps"] = AverageMeter()
    meter["disp_eps"] = AverageMeter()
    return meter

# 重置epoch损失和epoch精度
def reset_epoch_loss_acc_meter(meter):
    meter["epoch_loss"] = AverageMeter()
    meter["epoch_acc"] = AverageMeter()

# 重置disp损失和disp精度
def reset_disp_loss_acc_meter(meter):
    meter["disp_loss"] = AverageMeter()
    meter["disp_acc"] = AverageMeter()

# 更新epoch损失、epoch精度、disp损失和disp精度
def update_loss_acc_meter(meter, loss, acc, num):
    # loss：新添加的损失
    # acc：新增加的精度
    # num：新增的个数
    meter["epoch_loss"].update(loss, num)
    meter["epoch_acc"].update(acc, num)
    meter["disp_loss"].update(loss, num)
    meter["disp_acc"].update(acc, num)

# 更新epoch_eps和disp_eps
def update_eps_meter(meter, eps, num):
    # eps：新添加的扰动
    # num：新增的个数
    meter["epoch_eps"].update(eps, num)
    meter["disp_eps"].update(eps, num)

# MMA防御
class OriginMMADefense(Defense):
    def __init__(self, loader, dataname="train", verbose=True, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        # model：模型
        # defense_name：防御名称
        # dataset：数据集名称
        # training_parameters：训练模型的超参数
        # loader：默认为训练集
        # epochs：训练次数
        # dataname：默认为train
        # verbose：默认为True
        # dct_eps：记录当前扰动
        # dct_eps_record：记录每个epoch的扰动
        super(OriginMMADefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device
        self.dataname = dataname
        self.varbose = verbose
        self.loader = loader
        self.epochs = 0
        self.dct_eps = {}
        self.dct_eps_record = {}
        self.dct_eps_test = {}
        self.dct_eps_record_test = {}
        # 初始化meters、meters_test
        self.init_meters()
        self.init_meters_test()
        # 加载到设备上
        self.loader.targets = self.loader.targets.to(self.device)
        # self.model.to(self.device)

        # add_clean_loss：是否添加正确分类样本的损失，大于0为True，小于等于0为False
        self.add_clean_loss = self.clean_loss_coeff > 0
        # 如果要添加正确分类的损失，则将模型和优化器传入GradCloner中
        if self.add_clean_loss:
            self.grad_cloner = GradCloner(self.model, self.optimizer)
        # 交叉熵损失的原损失
        self.margin_loss_fn1 =  get_none_loss_fn(self.margin_loss_fn)
        # 交叉损失的平均损失（为loss_fn）
        self.loss_fn = get_mean_loss_fn(self.clean_loss_fn)
        # SLM损失的原损失
        self.search_loss_fn1 = get_none_loss_fn(self.search_loss_fn)
        # SLM损失和
        self.train_adv_loss_fn = get_sum_loss_fn(self.attack_loss_fn)
        # 添加的最大扰动
        self.maxeps = self.hinge_maxeps * 1.05
        # 初始化PGD扰动大小
        self.mineps = self.attack_mineps

        self.adv_meter = init_loss_acc_meter()
        # 将adv放入meters字典中
        self.meters_test["adv"] = self.adv_meter


        # Dataset：数据集名称（大写），必须为MNIST或CIFAR10
        self.Dataset = dataset.upper()
        assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"
        assert self._parsing_parameters(**kwargs)

        # num_epochs：训练次数
        # batch_size：分组大小
        self.num_epochs = training_parameters['num_epochs']
        self.batch_size = training_parameters['batch_size']

        # 构造MNIST/CIFAR10的优化器
        # if self.Dataset == 'MNIST':
        #     self.optimizer = optim.SGD(self.model.parameters(), lr=training_parameters['learning_rate'],
        #                                momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'], nesterov=True)
        # else:
        #     self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['lr'])
        if self.Dataset == 'MNIST':
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_parameters['learning_rate'])
        else:
            self.optimizer =  optim.SGD(self.model.parameters(), lr=training_parameters['lr'],
                                        momentum=training_parameters['momentum'], weight_decay=training_parameters['decay'])

    # 设置对抗训练的参数
    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nUser configurations for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs[key]))

        # 产生PGD对抗样本的参数(用于训练)
        # 由nb_iter（迭代次数）、attack_mineps（扰动范围）和eps_iter_scale计算pgd扰动步长
        self.nb_iter = kwargs['nb_iter']
        self.attack_mineps = kwargs['attack_mineps']
        self.eps_iter_scale = kwargs['eps_iter_scale']
        # 产生PGD对抗样本的参数(用于测试)
        self.test_eps = kwargs['test_eps']
        self.test_eps_iter = kwargs['test_eps_iter']
        # 损失函数
        self.clean_loss_fn = kwargs['clean_loss_fn']
        self.margin_loss_fn = kwargs['margin_loss_fn']
        self.attack_loss_fn = kwargs['attack_loss_fn']
        self.search_loss_fn = kwargs['search_loss_fn']
        # ANPGD添加的最大扰动，为论文中的dmax
        self.hinge_maxeps = kwargs['hinge_maxeps']
        # ANPGD的搜索次数
        self.num_search_steps = kwargs['num_search_steps']
        # 损失函数的系数
        self.clean_loss_coeff = kwargs['clean_loss_coeff']
        # 其他参数
        self.disp_interval = kwargs['disp_interval']
        return True

    # 产生PGD对抗样本
    def pgd_generation(self, var_natural_images=None, var_natural_labels=None, type='train'):
        # var_natural_images：干净样本
        # var_natural_labels：干净样本对应的标签
        # pgd攻击的迭代步长
        if type == 'train':
            pgd_eps_iter = self.eps_iter_scale * self.attack_mineps / self.nb_iter
            eps = self.attack_mineps
        if type == 'test':
            pgd_eps_iter = self.test_eps_iter
            eps = self.test_eps
        self.model.eval()
        natural_images = var_natural_images.cpu().numpy()
        # copy_images：复制干净样本numpy形式
        copy_images = natural_images.copy()
        # 从[-epsilon,epsilon）的均匀分布中随机采样，形成初始的扰动叠加在干净样本上
        copy_images = copy_images + np.random.uniform(-eps, eps, copy_images.shape).astype('float32')
        # 进行迭代
        for i in range(self.nb_iter):
            # 将初始的扰动样本由numpy形式转化为tensor形式
            var_copy_images = torch.from_numpy(copy_images).to(self.device)
            # 可以对其进行求导
            var_copy_images.requires_grad = True
            # 将其输入模型进行预测
            preds = self.model(var_copy_images)
            # 计算损失值(采用SLM损失和)
            loss = self.train_adv_loss_fn(preds, var_natural_labels)
            # loss = F.cross_entropy(preds, var_natural_labels)
            # 对输入求梯度
            gradient = torch.autograd.grad(loss, var_copy_images)[0]
            # 对梯度求符号函数并转为numpy形式
            gradient_sign = torch.sign(gradient).cpu().numpy()
            # 对样本添加一小步扰动
            copy_images = copy_images + pgd_eps_iter * gradient_sign
            # 将样本的扰动大小控制在[natural_images-epsilon,natural_images+epsilon]的范围之内
            copy_images = np.clip(copy_images, natural_images - eps, natural_images + eps)
            # 将扰动大小控制在[0.0,1.0]之间
            copy_images = np.clip(copy_images, 0.0, 1.0)
        # 返回最后copy_images的tensor形式，即为PGD对抗样本
        return torch.from_numpy(copy_images).to(self.device)

    # 产生ANPGD对抗样本，用于对抗训练
    def anpgd_generation(self, prev_eps, var_natural_images=None, var_natural_labels=None):
        # 产生PGD对抗样本
        pgd_adv = self.pgd_generation(self, var_natural_images, var_natural_labels, type='train')
        # 计算PGD对抗样本的方向
        unitptb = batch_multiply(1. / (prev_eps + 1e-12), (pgd_adv - var_natural_images))
        self.model.eval()
        # 计算SLM损失
        logit_margin = self.search_loss_fn(self.model(pgd_adv), var_natural_labels)
        # 最大扰动dmax与var_natural_labels的shape相同
        maxeps = self.maxeps * torch.ones_like(var_natural_labels).float()
        # 产生最佳扰动
        curr_eps = bisection_search(prev_eps, unitptb, self.model, var_natural_images, var_natural_labels, self.search_loss_fn1,
                                    logit_margin, maxeps, self.num_search_steps)
        # 产生ANPGD对抗样本
        pgd_adv = var_natural_images + batch_multiply(curr_eps, unitptb)
        return pgd_adv, curr_eps

    # 产生ANPGD对抗样本，用于白盒攻击测试
    def anpgd_generation_test(self, var_natural_images=None, var_natural_labels=None):
        # 产生PGD对抗样本
        pgd_adv = self.pgd_generation(self, var_natural_images, var_natural_labels, type='test')
        # 计算PGD对抗样本扰动方向
        unitptb = batch_multiply(1. / (self.test_eps + 1e-12), (pgd_adv - var_natural_images))
        # 计算LM损失
        self.model.eval()
        logit_margin = elementwise_margin(self.model(pgd_adv), var_natural_labels)
        # 产生与var_natural_labels大小相同全为1的tensor
        ones = torch.ones_like(var_natural_labels).float()
        # 最大扰动dmax与var_natural_labels的shape相同
        maxeps = self.maxeps * ones
        # 计算最佳扰动，当前扰动设置为最大扰动的一半，loss设置为LMloss
        curr_eps = bisection_search(maxeps * 0.5, unitptb, self.model, var_natural_images, var_natural_labels,
                                    elementwise_margin, logit_margin, maxeps, self.num_search_steps)
        # 返回PGD对抗样本和最佳扰动
        return pgd_adv, curr_eps

    # 获取正确分类样本的扰动(初始化为mineps)
    def get_eps(self, idx, data):
        # idx：data中正确分类的下标值
        # data：输入图像
        lst_eps = []
        # 遍历idx
        for ii in idx:
            ii = ii.item()
            # lst_eps添加max[攻击中的最小eps,dct_eps设置默认值的最小eps]
            lst_eps.append(max(self.mineps, self.dct_eps.setdefault(ii, self.mineps)))
        # 返回lst_eps的tensor形式
        return data.new_tensor(lst_eps)

    # 更新dct_eps为当前扰动值，将每个epoch的扰动值记录在dct_eps_record中
    def update_eps(self, eps, idx):
        # eps：当前扰动值（错误分类为0，正确分类为ANPGD扰动）
        # idx：一个分组data的下标值
        # 遍历idx
        for jj, ii in enumerate(idx):
            # ii为idx的值
            ii = ii.item()
            # 设置当前扰动的值为eps[jj]
            curr_epsval = eps[jj].item()
            # 如果ii没有在dct_eps_record中，令dct_eps_record[ii]为[]
            if ii not in self.dct_eps_record:
                self.dct_eps_record[ii] = []
            # 将dct_eps_record[ii]插入当前扰动值，epochs
            self.dct_eps_record[ii].append((curr_epsval, self.epochs))
            # 更新dct_eps[ii]的值为当前扰动值
            self.dct_eps[ii] = curr_epsval

    # 更新dct_eps为当前扰动值，将每个epoch的扰动值记录在dct_eps_record中
    def update_eps_test(self, eps, idx):
        # eps：当前扰动值（错误分类为0，正确分类为ANPGD扰动）
        # idx：一个分组data的下标值
        # 遍历idx
        for jj, ii in enumerate(idx):
            # ii为idx的值
            ii = ii.item()
            # 设置当前扰动的值为eps[jj]
            curr_epsval = eps[jj].item()
            # 如果ii没有在dct_eps_record中，令dct_eps_record[ii]为[]
            if ii not in self.dct_eps_record_test:
                self.dct_eps_record_test[ii] = []
            # 将dct_eps_record[ii]插入当前扰动值，epochs
            self.dct_eps_record_test[ii].append((curr_epsval, self.epochs - 1))
            # 更新dct_eps[ii]的值为当前扰动值
            self.dct_eps_test[ii] = curr_epsval

    # 一个batch的MMA对抗训练
    def train_one_batch(self, data, idx, target):
        # data：一个batch的训练集
        # target：data对应的真实标签
        # idx：data中每张图像的编号
        self.model.eval()
        # 经过模型的训练集的logit输出
        clnoutput = self.model(data)
        # 对干净训练集求平均交叉熵损失
        clnloss = self.loss_fn(clnoutput, target)
        # 是否添加净损失
        if self.add_clean_loss:
            # 清空优化器梯度
            self.optimizer.zero_grad()
            # 反向传播
            clnloss.backward()
            # 复制原始优化器梯度，再清空原始优化器梯度
            self.grad_cloner.copy_and_clear_grad()
        # 对干净训练集计算SLM原损失
        search_loss = self.search_loss_fn1(clnoutput, target)
        # 正确分类的图像slm损失小于0
        cln_correct = (search_loss < 0)
        # 错误分类的图像的slm损失大于等于0
        cln_wrong = (search_loss >= 0)
        # 获取正确分类的图像
        data_correct = data[cln_correct]
        # 获取正确分类的标签
        target_correct = target[cln_correct]
        # 获取正确分类的图像在data中的下标
        idx_correct = idx[cln_correct]
        # 计算所有分类正确的data数量
        num_correct = cln_correct.sum().item()
        # 计算所有错误分类的数量
        num_wrong = cln_wrong.sum().item()
        # 初始化当前的扰动为0，shape与len(data)相同
        curr_eps = data.new_zeros(len(data))

        # 如果正确分类的图片数目大于0
        if num_correct > 0:
            # 获取分类正确图片的PGD初始扰动范围
            prev_eps = self.get_eps(idx_correct, data)
            # 利用ANPGD，返回ANPGD对抗样本和最佳扰动
            advdata_correct, curr_eps_correct = self.anpgd_generation(data_correct, target_correct, prev_eps)
            # 用ANPGD对抗样本替换正确分类的样本，更新data
            data[cln_correct] = advdata_correct
            # 更新正确分类样本添加的扰动，错误分类的保持为0
            curr_eps[cln_correct] = curr_eps_correct
        # 模型预测输出（正确分类的添加ANPGD扰动后的分类结果，错误分类则为原样本的分类结果）
        mmaoutput = self.model(data)
        # 如果没有正确分类的样本，则令正确分类样本的交叉损失熵为0
        if num_correct == 0:
            marginloss = mmaoutput.new_zeros(size=(1,))
        # 否则求正确分类样本添加ANPGD扰动后的交叉熵损失
        else:
            marginloss = margin_loss_fn(mmaoutput[cln_correct], target[cln_correct])
        # 如果错误分类的数量为0，则错误分类的loss为0
        if num_wrong == 0:
            clsloss = 0.
        # 否则为错误分类样本的交叉熵损失
        else:
            clsloss = clean_loss_fn(mmaoutput[cln_wrong], target[cln_wrong])
        # 如果有正确分类的样本，仅取小于dmax的扰动的交叉熵损失
        if num_correct > 0:
            marginloss = marginloss[self.hinge_maxeps > curr_eps_correct]
        # 求平均MMA损失
        mmaloss = (marginloss.sum() + clsloss * num_wrong) / len(data)
        # 清空梯度
        self.optimizer.zero_grad()
        # 反向传播
        mmaloss.backward()
        # 组合梯度，1/3净损失+2/3MMA损失
        if self.add_clean_loss:
            self.grad_cloner.combine_grad(1 - self.clean_loss_coeff, self.clean_loss_coeff)
        # 梯度更新
        self.optimizer.step()
        # 更新dct_eps和dct_eps_record
        self.update_eps(curr_eps, idx)
        # 返回经过模型一个batch的logit输出、训练集的平均交叉熵损失和当前扰动（0/ANPGD扰动）
        return clnoutput, clnloss, curr_eps

    # 初始化meters（OrderedDict会保留字典的添加顺序）
    # cln包括epoch_loss、epoch_acc、disp_loss和disp_acc，每一个epoch训练开始时都会被清空
    # disp_loss和disp_acc，在batch达到disp_interval时就会被清空
    # eps包括，epoch_eps和distp_eps，计算的是所有epoch中添加的平均扰动，不会被清空
    def init_meters(self):
        # 初始化一个字典
        self.meters = OrderedDict()
        # 初始化cln_meter，包括epoch_loss、epoch_acc、disp_loss和disp_acc
        self.cln_meter = init_loss_acc_meter()
        # 将cln_meter加入meters字典中
        self.meters["cln"] = self.cln_meter
        # 初始化eps_meter，包括epoch_eps和disp_eps
        self.eps_meter = init_eps_meter()
        # 将eps_meter加入meters字典中
        self.meters["eps"] = self.eps_meter

    # 初始化meters_test（OrderedDict会保留字典的添加顺序）
    def init_meters_test(self):
        # 初始化一个字典
        self.meters_test = OrderedDict()
        # 初始化cln_meter，包括epoch_loss、epoch_acc、disp_loss和disp_acc
        self.cln_meter_test = init_loss_acc_meter()
        # 将cln_meter加入meters字典中
        self.meters_test["cln"] = self.cln_meter_test
        # 初始化eps_meter，包括epoch_eps和disp_eps
        self.eps_meter_test = init_eps_meter()
        # 将eps_meter加入meters字典中
        self.meters_test["eps"] = self.eps_meter_test

    # 重置meters中的epoch_loss和epoch_acc
    def reset_epoch_meters(self):
        for key in self.meters:
            reset_epoch_loss_acc_meter(self.meters[key])

    # 重置meters_test中的epoch_loss和epoch_acc
    def reset_epoch_meters_test(self):
        for key in self.meters_test:
            reset_epoch_loss_acc_meter(self.meters_test[key])

    # 重置meters中的disp_loss和disp_acc
    def reset_disp_meters(self):
        for key in self.meters:
            reset_disp_loss_acc_meter(self.meters[key])

    # 重置meters_test中的disp_loss和disp_acc
    def reset_disp_meters_test(self):
        for key in self.meters_test:
            reset_disp_loss_acc_meter(self.meters_test[key])

    # 预测并更新epoch_loss、epoch_acc、disp_loss和disp_acc，返回损失值和精度
    def predict_then_update_loss_acc_meter(self, meter, data, target):
        # data：输入图像
        # target：真实标签
        # meter：字典
        with torch.no_grad():
            # 经过模型预测输出
            output = self.model(data)
        # 计算预测正确的精度
        acc = get_accuracy(predict_from_logits(output), target)
        # 计算损失值（默认为平均交叉熵损失）
        loss = self.loss_fn(output, target).item()
        # 更新epoch_loss、epoch_acc、disp_loss和disp_acc
        update_loss_acc_meter(meter, loss, acc, len(data))
        # 返回损失值和精度
        return loss, acc

    def print_disp_meters(self, batch_idx=None):
        # batch_id：分组的序号，默认为None
        if not self.verbose:
            return
        # 如果batch_id不为None，打印训练一个epoch的batch进度
        if batch_idx is not None:
            disp_str = "Epoch: {} ({:.0f}%)".format(self.epochs,100. * (batch_idx + 1) / len(self.loader),)
        else:
            disp_str = ""
        # 遍历meters
        for key in self.meters:
            meter = self.meters[key]
            # 如果key为eps，打印disp_eps的平均值
            if key == "eps":
                disp_str += "\tavgeps: {:.4f}".format(meter["disp_eps"].avg)
            # 如果key为cln或adv或mix，打印disp_loss和disp_acc的平均值
            elif key in ["cln", "adv", "mix"]:
                disp_str += "\t{}loss: {:.4f}, {}acc: {:.2f}%".format(key, meter["disp_loss"].avg, key,
                                                                      100 * meter["disp_acc"].avg)
            # 否则提示错误
            else:
                raise ValueError("key=".format(key))
        # 打印disp_str
        print(disp_str)

    def print_disp_meters_test(self, batch_idx=None):
        # batch_id：分组的序号，默认为None
        if not self.verbose:
            return
        # 如果batch_id不为None，打印训练一个epoch的batch进度
        if batch_idx is not None:
            disp_str = "Epoch: {} ({:.0f}%)".format(self.epochs,100. * (batch_idx + 1) / len(self.loader),)
        else:
            disp_str = ""
        # 遍历meters
        for key in self.meters_test:
            meter = self.meters_test[key]
            # 如果key为eps，打印disp_eps的平均值
            if key == "eps":
                disp_str += "\tavgeps: {:.4f}".format(meter["disp_eps"].avg)
            # 如果key为cln或adv或mix，打印disp_loss和disp_acc的平均值
            elif key in ["cln", "adv", "mix"]:
                disp_str += "\t{}loss: {:.4f}, {}acc: {:.2f}%".format(key, meter["disp_loss"].avg, key,
                                                                      100 * meter["disp_acc"].avg)
            # 否则提示错误
            else:
                raise ValueError("key=".format(key))
        # 打印disp_str
        print(disp_str)

    # 根据dct_eps所在扰动分段位置更新hist
    def disp_eps_hist(self, bins=10):
        # bins：分段数量，默认为10
        # interval：把[0,adversary.maxeps]均分为10段
        interval = self.maxeps / bins
        hist_str = []
        hist = []
        thresholds = []
        for ii in range(bins):
            # 将[0,dmax]等距分为bins=10段
            thresholds.append((ii * interval, (ii + 1) * interval))
            # 将分段用'~ to ~:'的形式存储在hist_str中
            hist_str.append("{:.2f} to {:.2f}:".format(thresholds[-1][0], thresholds[-1][1]))
            # 添加bins=10个0
            hist.append(0)

        # 遍历eps字典，寻找dct_eps[key]所在的分段位置
        for key in self.dct_eps:
            # 初始化assigned为False
            assigned = False
            # 遍历每一个分段，如果dct_eps[key]在某个分段中，hist对应位置+1，然后跳出循环
            for ii in range(bins):
                if thresholds[ii][0] <= self.dct_eps[key] < thresholds[ii][1]:
                    hist[ii] += 1
                    assigned = True
                    break
            # 如果assigned为False并且dct_eps[key]接近maxeps，则最后一个hist+1，进行下一次循环
            if not assigned and np.allclose(self.dct_eps[key], self.maxeps):
                hist[-1] += 1
                assigned = True
            # 如果仅仅assigned为False，dct_eps[key]远离maxeps，则提示错误
            if not assigned:
                raise ValueError("Should not reach here, eps={}, maxeps={}".format(self.dct_eps[key], self.maxeps))

        # 输出hist_str和hist(在哪一段上有多少个dct_eps的value在其中)
        for hstr, h in zip(hist_str, hist):
            print(hstr, h)

    # 根据dct_eps所在扰动分段位置更新hist
    def disp_eps_hist_test(self, bins=10):
        # bins：分段数量，默认为10
        # interval：把[0,adversary.maxeps]均分为10段
        interval = self.maxeps / bins
        hist_str = []
        hist = []
        thresholds = []
        for ii in range(bins):
            # 将[0,dmax]等距分为bins=10段
            thresholds.append((ii * interval, (ii + 1) * interval))
            # 将分段用'~ to ~:'的形式存储在hist_str中
            hist_str.append("{:.2f} to {:.2f}:".format(thresholds[-1][0], thresholds[-1][1]))
            # 添加bins=10个0
            hist.append(0)

        # 遍历eps字典，寻找dct_eps[key]所在的分段位置
        for key in self.dct_eps_test:
            # 初始化assigned为False
            assigned = False
            # 遍历每一个分段，如果dct_eps[key]在某个分段中，hist对应位置+1，然后跳出循环
            for ii in range(bins):
                if thresholds[ii][0] <= self.dct_eps_test[key] < thresholds[ii][1]:
                    hist[ii] += 1
                    assigned = True
                    break
            # 如果assigned为False并且dct_eps[key]接近maxeps，则最后一个hist+1，进行下一次循环
            if not assigned and np.allclose(self.dct_eps_test[key], self.maxeps):
                hist[-1] += 1
                assigned = True
            # 如果仅仅assigned为False，dct_eps[key]远离maxeps，则提示错误
            if not assigned:
                raise ValueError("Should not reach here, eps={}, maxeps={}".format(self.dct_eps_test[key], self.maxeps))

        # 输出hist_str和hist(在哪一段上有多少个dct_eps的value在其中)
        for hstr, h in zip(hist_str, hist):
            print(hstr, h)

    # 一个epoch的MMA对抗训练
    def train_one_epoch(self):
        # 开始训练的时间点
        _bgn_epoch = time.time()
        # 打印epochs数
        if self.verbose:
            print("Training epoch {}".format(self.epochs))
        # 准备训练
        self.model.train()
        # 将模型加载到设备上
        self.model.to(self.device)
        # 重置meters中所有的epoch_loss和epoch_acc
        self.reset_epoch_meters()
        # 重置meters中所有的disp_loss和disp_acc
        self.reset_disp_meters()
        # 一个epoch的训练时间，初始化为0.0
        _train_time = 0.
        # 遍历训练集
        for batch_idx, (data, idx) in enumerate(self.loader):
            # batch_idx：每个batch的id
            # data：每个batch的图像
            # idx：data的id
            # target：真实标签
            data, idx = data.to(self.device), idx.to(self.device)
            target = self.loader.targets[idx]
            # 开始训练的时间点
            _bgn_train = time.time()
            # 一个batch的MMA训练，返回干净样本的logit输出、训练集的平均交叉熵损失和当前扰动（0/ANPGD扰动）
            clnoutput, clnloss, eps = self.train_one_batch(data, idx, target)
            # 每个batch的训练时间叠加
            _train_time = _train_time + (time.time() - _bgn_train)
            # 计算干净样本经过模型的分类精度
            clnacc = get_accuracy(predict_from_logits(clnoutput), target)
            # 更新cln中的epoch_loss、epoch_acc、disp_loss和disp_acc
            update_loss_acc_meter(self.cln_meter, clnloss.item(), clnacc, len(data))
            # 更新eps中的epoch_eps和disp_eps
            update_eps_meter(self.eps_meter, eps.mean().item(), len(data))
            # 如果batch_idx可以整除disp_interval
            if self.disp_interval is not None and \
                    batch_idx % self.disp_interval == 0:
                # 打印训练进度、disp_eps/disp_acc(每个batch的)/disp_loss(每个batch的)的平均值
                self.print_disp_meters(batch_idx)
                # 重置所有的disp_loss和disp_acc
                self.reset_disp_meters()
        # 一个epoch结束后，打印disp_eps、disp_loss和disp_acc
        self.print_disp_meters(batch_idx)
        # 输出hist_str和hist
        self.disp_eps_hist()
        # 训练的epoch数加1
        self.epochs += 1
        # 打印训练一个epoch的时间
        print("total epoch time", time.time() - _bgn_epoch)
        # 打印总的训练时间
        print("training total time", _train_time)

    # MMA白盒攻击测试
    def test_one_epoch(self, dataname, loader):
        print("Evaluating on {}, epoch {}".format(self.dataname, self.epochs))
        self.model.eval()
        self.model.to(self.device)
        # 重置meters_test中所有的epoch_loss、epoch_acc
        self.reset_epoch_meters_test()
        # 重置meters_test中所有的disp_loss、disp_acc
        self.reset_disp_meters_test()
        # 遍历验证集/测试集
        for data, idx in loader:
            data, idx = data.to(self.device), idx.to(self.device)
            target = loader.targets[idx]
            # 获取PGD对抗样本和ANPGD扰动
            advdata, curr_eps = self.anpgd_generation_test(data, target)
            # 更新epoch_eps、disp_eps为curr_eps的平均值
            update_eps_meter(self.eps_meter_test, curr_eps.mean().item(), len(data))
            # 更新dct_eps_record_test和dic_eps_test为当前验证集/测试集添加的扰动
            self.update_eps_test(curr_eps, idx)
            # 预测并cln中的更新epoch_loss、epoch_acc、disp_loss和disp_acc，返回损失值和精度（干净样本）
            clnloss, clnacc = self.predict_then_update_loss_acc_meter(self.cln_meter_test, data, target)
            # 预测并更新adv中epoch损失、epoch精度、disp损失和disp精度，返回损失值和精度（PGD对抗样本）
            advloss, advacc = self.predict_then_update_loss_acc_meter(self.adv_meter, advdata, target)
        # 打印disp_eps、disp_loss和disp_acc
        self.print_disp_meters_test()
        # 输出hist_str和hist
        self.disp_eps_hist_test()
        # 返回干净验证集/测试集的平均精度、白盒攻击PGD对抗样本的平均精度、需要添加的平均ANPGD扰动
        return (self.meters_test['cln']['epoch_acc'].avg, self.meters_test['adv']['epoch_acc'].avg,
                np.array(list(self.dct_eps_test.values())).mean())

    def test_defense(self, validation_loader=None):
        # 初始化最大ANPGD平均扰动为0.0
        best_avgeps = 0.
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
            # 验证集的平均精度、白盒攻击PGD对抗样本的平均精度、需要添加的平均ANPGD扰动
            val_clnacc, val_advacc, val_avgeps = self.test_one_epoch(dataname='valid', loader=validation_loader)
            # 如果是CIFAR10数据集，则调整模型参数
            if self.Dataset == 'CIFAR10':
                adjust_mma_learning_rate(epoch=epoch, optimizer=self.optimizer)
            # 将最佳模型参数保存到DefenseEnhancedModels/OriginMMA/CIFAR10_OriginMMA_enhanced.pt中或MNIST
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
            if val_avgeps > best_avgeps:
                # 更新平均ANPGD扰动（越大越好）
                best_avgeps = val_avgeps
                if best_avgeps != 0.:
                    os.remove(defense_enhanced_saver)
                self.model.save(name=defense_enhanced_saver)
            else:
                print('Train Epoch{:>3}: Average correct ANPGD perturbation did not improve from {:.4f}\n'.format(epoch, best_avgeps))







    # # 保存在验证集上最好分类精度的MMA对抗训练模型
    # def defense(self, validation_loader=None):
    #     # train_loader：训练集
    #     # validation_loader：验证集
    #     # best_val_acc：验证集上的最佳分类精度
    #     best_val_acc = None
    #     for epoch in range(self.num_epochs):
    #         self.train_one_epoch()
    #         # 计算验证集精度
    #         val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
    #         # 调整学习率
    #         if self.Dataset == 'CIFAR10':
    #             adjust_mma_learning_rate(epoch=epoch, optimizer=self.optimizer)
    #         # 将最佳模型参数保存到DefenseEnhancedModels/OriginMMA/CIFAR10_OriginMMA_enhanced.pt中或MNIST
    #         assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
    #         defense_enhanced_saver = '../DefenseEnhancedModels/{}/{}_{}_enhanced.pt'.format(self.defense_name, self.Dataset, self.defense_name)
    #         # 将最好的模型参数进行保存
    #         if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
    #             if best_val_acc is not None:
    #                 os.remove(defense_enhanced_saver)
    #             best_val_acc = val_acc
    #             self.model.save(name=defense_enhanced_saver)
    #         else:
    #             print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))



