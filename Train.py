import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

from evaluate_classify.Evaluate_classify import Confusion_metrics, Confusion_matrixs
from model.CNN import CNN
from model.CNN_kan import KANC_MLP
from model.MY_MLP import Net
import sys
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from model.efficient_Kan import KAN
from model import Mymodel

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
# 训练集的数据集
train_data = torchvision.datasets.FashionMNIST(root='processed/FashionMNIST',
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())
# 测试集的数据集
test_data = torchvision.datasets.FashionMNIST(root='processed/FashionMNIST',
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())
if __name__ == "__main__":
    # 创建两个Dataloader, 包尺寸为64
    # 训练用的Dataloader
    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # 明确使用cuda进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 测试用的Dataloader*
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # 实例化模型
    # model = KAN([28 * 28, 280, 64, 10])  # KAN 网络
    # model = Net()  # MLP 网络
    # model = KANC_MLP() # KAN 卷积
    model = CNN()  # 卷积神经网络 CNN
    model.to(device)
    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 学习逐渐减小
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # 迭代次数
    epoch = 100
    accuracy_train_list = []
    accuracy_test_list = []
    loss_train_list = []
    loss_test_list = []
    for epoch in range(100):  # 可以定义训练次数
        loss_train__total =0
        accuracy_total = 0
        with tqdm(train_dataloader) as pbar:
            for i, (imgs, labels) in enumerate(pbar):
                model.train()
                # imgs, labels = imgs.view(imgs.size(0),-1).to(device), labels.to(device)
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = loss_func(output, labels)
                loss_train__total += loss.item()
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                accuracy_total += (output.argmax(dim=1) == labels).sum()
                pbar.set_postfix(train_epoch=epoch + 1,
                                 loss=loss.item(),
                                 accuracy=accuracy.item(),
                                 lr=optimizer.param_groups[0]['lr'],
                                 accuracy_train_global=(accuracy_total / ((i + 1) * batch_size)).item()
                                 )
        loss_train_list.append((loss_train__total/(i+1)))
        # 获取每次训练的数据用于画图
        accuracy_train_list.append((accuracy_total / ((i + 1) * batch_size)).item())
        loss_test_total = 0
        accuracy_total = 0
        pred = []
        y_true = []
        with tqdm(test_dataloader) as pbar:
            for j, (imgs, labels) in enumerate(pbar):
                model.eval()
                # imgs, labels = imgs.view(imgs.size(0), -1).to(device), labels.to(device)
                # .view(-1, 28 * 28)
                imgs, labels = imgs.to(device), labels.to(device)
                y_true.append(labels)
                output = model(imgs)
                loss = loss_func(output, labels)
                loss_test_total += loss.item()
                pred.append(output.argmax(1).cpu())
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                accuracy_total += (output.argmax(dim=1) == labels).sum()

                pbar.set_postfix(test_epoch=epoch + 1,
                                 accuracy=accuracy.item(),
                                 accuracy_test_global=(accuracy_total / ((j + 1) * batch_size)).item()
                                 )
            Confusion_metrics(y_true, pred)
            Confusion_matrixs(y_true, pred)
            # 如果测试的准确率超过了90%，则保存模型
            # 每一个模型对应一个pth，和，txt文档 需要使用的话分别调用每个模型的训练批次和查看txt结果。
            if (accuracy_total / ((j + 1) * batch_size)).item() >= 0:
                # torch.save(model, 'whole_model.pth')
                # # 加载整个模型
                # loaded_model = torch.load('whole_model.pth')
                torch.save(model, 'log/CNN_model.pth')
                # 将结果写入到一个txt文档当中
                result = f"模型的第{epoch + 1}次测试的结果，当前模型为  CNN_model网络模型的准确率为：{(accuracy_total / ((j + 1) * batch_size)).item()}"
                with open(f'log/{batch_size}CNN_model_test_accuracy.txt', 'a', encoding='utf-8') as file:
                    # 将结果写入文件
                    file.write(result)
                    file.write('\n')
                    file.close()
        # 打印每次训练的数据用于画图
        loss_test_list.append((loss_test_total/(j+1)))
        accuracy_test_list.append((accuracy_total / ((j + 1) * batch_size)).item())
    print('训练集的训练数据', accuracy_train_list)
    print('测试集上的结果是：', accuracy_test_list)
    print('训练集上的损失值：',loss_train_list)
    print('测试集上的损失值：',loss_test_list)
    # 打印混淆矩阵

