import torch
import torch.nn.functional as F

# 一个epoch的正常训练
def train_one_epoch(model, train_loader, optimizer, epoch, device):
    # model：模型
    # train_loader：训练集
    # optimizer：优化器
    # epoch：训练次数
    # 进行训练
    model.train()
    # 循环训练每一个epoch
    for index, (images, labels) in enumerate(train_loader):
        # 将图像和标签加载到设备上
        images = images.to(device)
        labels = labels.to(device)
        # 图像经过模型预测
        outputs = model(images)
        # 计算交叉损失
        loss = F.cross_entropy(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印训练进度
        print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
              format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')

# 计算模型对干净验证集的精度
def validation_evaluation(model, validation_loader, device):
    # model：模型
    # validation_loader：验证集
    model = model.to(device)
    model.eval()
    total = 0.0
    correct = 0.0
    # 不需要计算梯度，也不会进行反向传播
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # predicted：返回输出最大值的索引
            _, predicted = torch.max(outputs.data, 1)
            # labels.size(0)：返回行数，求总的标签数目
            total = total + labels.size(0)
            # 对预测正确的标签进行求和
            correct = correct + (predicted == labels).sum().item()
        # 计算准确率
        ratio = correct / total
    print('validation dataset accuracy is {:.4f}'.format(ratio))
    return ratio

# 计算模型对干净测试集的精度
def testing_evaluation(model, test_loader, device):
    # model：模型
    # test_loader：测试集
    print('\n#####################################')
    print('#### The {} model is evaluated on the testing dataset loader ...... '.format(model.model_name))
    model = model.to(device)
    model.eval()
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('#### Accuracy of the loaded model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%'.format(correct, total, ratio * 100))
    print('#####################################\n')
    return ratio


