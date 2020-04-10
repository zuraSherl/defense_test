import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# 对x进行封装，requires_grad为True时对x求导
def tensor2variable(x=None, device=None, requires_grad=False):
    x = x.to(device)
    return Variable(x, requires_grad=requires_grad)

# 对samples进行封装，并传入模型返回预测结果标签
def predict(model=None, samples=None, device=None):
    # samples：输入
    # model：模型
    model.eval()
    model = model.to(device)
    # 操作copy_samples并不能改变samples
    copy_samples = np.copy(samples)
    # numpy中的ndarray转化成pytorch中的tensor：torch.from_numpy()
    var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
    # 将var_samples传入模型中得到预测的结果
    predictions = model(var_samples.float())
    return predictions