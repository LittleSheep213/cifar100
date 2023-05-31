import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import nn
from dataset import mydataset, unpickle
from net import ResNet18, simpleCnn
from torch.nn import functional as F
from metrics import *

from torch.utils.tensorboard import SummaryWriter


# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
cifar100_train = unpickle('../data/cifar-100-python/train')
train_data = mydataset(cifar100_train, train=True, method='None')
train_data_len = len(train_data)
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)

cifar100_test = unpickle('../data/cifar-100-python/test')
test_data = mydataset(cifar100_test, train=False)
test_data_len = len(test_data)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建网络模型
# res_model = ResNet18()
# res_model = res_model.to(device)
# res_model = torch.nn.DataParallel(res_model, device_ids=[0, 1, 2])
sim_model = simpleCnn()
sim_model = sim_model.to(device)


# 损失函数
loss_fun = nn.CrossEntropyLoss()
# loss_fun = WeightedFocalLoss()
loss_fun = loss_fun.to(device)


# 优化器
learning_rate = 0.001
optimizer = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练轮数
epoch = 30
# 记录训练次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

for i in tqdm(range(epoch)):
    for n, data in enumerate(train_data_loader):
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = sim_model(imgs)
        targets = torch.as_tensor(targets).reshape(-1, 100)
        # targets = F.one_hot(targets, num_classes=100)
        loss = loss_fun(outputs, targets.float())

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'----------------------epoch{i}---------------------')

    with torch.no_grad():
        train_data1 = mydataset(cifar100_train, train=False, method='None')
        train_data_loader1 = DataLoader(train_data1, batch_size=64, shuffle=True)
        correct_num = 0
        train_loss = []

        for data in train_data_loader1:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = sim_model(imgs)
            targets = torch.as_tensor(targets)
            train_loss.append(loss_fun(outputs, targets.float()).cpu())
            targets = targets.argmax(1)
            correct_num += topN_count(outputs, targets, 5)
        train_loss_epoch = np.mean(train_loss)
        train_acc = correct_num / train_data_len
        print(f'train_loss:{train_loss_epoch}')
        print(f'train_acc:{train_acc}')

    with torch.no_grad():
        correct_num = 0
        test_loss = []
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = sim_model(imgs)
            targets = torch.as_tensor(targets)
            test_loss.append(loss_fun(outputs, targets.float()).cpu())
            targets = targets.argmax(1)
            correct_num += topN_count(outputs, targets, 5)
        test_loss_epoch = np.mean(test_loss)
        test_acc = correct_num / test_data_len
        print(f'test_loss:{test_loss_epoch}')
        print(f'test_acc:{test_acc}')
    torch.save(sim_model.state_dict(), f'weight/sim/sim_epoch{i + 1}.pth')  # 保存参数
    writer.add_scalars('sim_loss', {'train': train_loss_epoch,
                                    'test': test_loss_epoch}, i+1)
    writer.add_scalars('sim_acc', {'train': train_acc,
                                   'test': test_acc}, i+1)

writer.close()
