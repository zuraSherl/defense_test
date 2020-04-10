import argparse
import copy
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

# ****************引入需要训练的模型****************
from src.mnist_model import MNIST_CNN
from src.train_test import train_one_epoch, testing_evaluation, validation_evaluation
from src.get_datasets import get_mnist_train_validate_loader, get_mnist_test_loader

# 训练普通的CNN模型并保存
# **************设置训练模型的参数**************
MNIST_Training_Parameters = {
    'num_epochs': 50,
    'batch_size': 100,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'decay': 2e-4
}

# 调整学习率
def adjust_MNIST_learning_rate(epoch, optimizer):
    # 最小的学习率
    minimum_learning_rate = 0.5e-6
    # optimizer.param_groups：优化器对象的超参数
    for param_group in optimizer.param_groups:
        lr_temp = param_group["lr"]
        # 如果epoch的次数达到一定值时，进行学习率的调整
        if epoch == 20 or epoch == 40:
            lr_temp = lr_temp * 0.1
        # 将调整后的学习率写入到优化器中
        param_group["lr"] = max(lr_temp, minimum_learning_rate)
        print('The **learning rate** of the {} epoch is {}'.format(epoch, param_group["lr"]))

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 下载mnist训练集，划分为训练集和测试集（然后进行分组），保存到MNIST文件夹下面
    train_loader, valid_loader = get_mnist_train_validate_loader(dir_name='../data/MNIST', batch_size=MNIST_Training_Parameters['batch_size'],valid_size=0.1, shuffle=True)
    # 下载minst测试集（然后进行分组），保存到MNIST文件夹下面
    test_loader = get_mnist_test_loader(dir_name='../data/MNIST', batch_size=MNIST_Training_Parameters['batch_size'])
    # 设置模型
    # **************引入的模型名称**************
    mnist_model = MNIST_CNN().to(device)
    # 设置优化器
    optimizer = optim.SGD(mnist_model.parameters(), lr=MNIST_Training_Parameters['learning_rate'],
                          momentum=MNIST_Training_Parameters['momentum'], weight_decay=MNIST_Training_Parameters['decay'], nesterov=True)
    # 训练
    # 最好的验证集精度
    best_val_acc = None
    # 训练模型参数保存路径：/MNIST/model/MNIST_raw.pt
    # **************不同模型需要修改名称**************
    model_saver = '../data/MNIST/model/MART_MNIST_' + 'raw' + '.pt'
    # 进行epoch次循环训练
    for epoch in range(MNIST_Training_Parameters['num_epochs']):
        # 一次epoch训练
        train_one_epoch(model=mnist_model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, device=device)
        # 验证集的精度
        val_acc = validation_evaluation(model=mnist_model, validation_loader=valid_loader, device=device)
        adjust_MNIST_learning_rate(optimizer=optimizer, epoch=epoch)
        # 每一次epoch后验证集的精度大于最好的精度时（移除模型保存路径），或者best_val_acc为None时，更新最佳精度，然后将模型参数重新写入保存路径中
        if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
            if best_val_acc is not None:
                os.remove(model_saver)
            best_val_acc = val_acc
            mnist_model.save(name=model_saver)
        # 否则提示精度未发生提高
        else:
            print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch, best_val_acc))
    # 测试
    # 复制mnist_model
    final_model = copy.deepcopy(mnist_model)
    # 加载final_model
    final_model.load(path=model_saver, device=device)
    # 计算模型在测试集上面的精度并输出
    accuracy = testing_evaluation(model=final_model, test_loader=test_loader, device=device)
    # 打印模型在测试集上的精度
    print('Finally, the ACCURACY of saved model [{}] on testing dataset is {:.2f}%\n'.format(final_model.model_name, accuracy * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Testing the model for MNIST')
    # gpu_index：默认值为0
    # seed：默认值为100
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    arguments = parser.parse_args()
    main(arguments)