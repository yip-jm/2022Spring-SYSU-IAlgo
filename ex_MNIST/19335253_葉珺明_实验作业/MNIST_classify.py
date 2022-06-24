import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib
from matplotlib import pyplot as plt
import random 
from torch.optim import lr_scheduler as lr_scheduler

def random_flip_and_rotate(x):
    x = x.numpy()
    x = np.transpose(x,(0,2,3,1))
    x_out = np.zeros(x.shape)
    for i in range(x.shape[0]):
        temp = x[i,...]
        if random.random() < 0.5:
            # 上下翻转
            imtemp1 = np.flipud(temp)
        if random.random() < 0.5:
            # 左右翻转
            temp = np.fliplr(temp)
        # 旋转
        angle = random.choice([0, 1, 2, 3])
        temp = np.rot90(temp, angle)

        x_out[i,...] = temp
    x_out = np.transpose(x_out,(0,3,1,2))
    x_out = x_out.astype(np.float32)
    x_out = torch.from_numpy(x_out)
    return x_out

def draw_train(loss_, acc_):
    x_ = []
    for i in range (len(loss_)):
        x_.append(i+1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(x_, loss_, 'g', label='Train_Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("LOSS")
    ax1.set_title("Training")

    ax2 = ax1.twinx()
    lns2 = ax2.plot(x_, acc_, 'b', label='Train_Acc')
    ax2.set_ylabel("ACC")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.show()
    plt.savefig("./optimization/pic/train.png") 
    plt.close()

def draw_test(acc):
    x_ = []
    for i in range (len(acc)):
        x_.append(i+1)

    plt.title('Test')
    plt.xlabel("Epochs")
    plt.ylabel("ACC")
    plt.plot(x_, acc, 'r', label='Test_Acc')

    plt.savefig("./optimization/pic/test.png")
    plt.close()


# 超参数
T_max = 100
eta_min = 1e-5
batch_size = 100
epochs = 50
lr = 1e-3
# 加载数据集，toTensor将img[h,w,c]->[c,h,w]
train_data = mnist.MNIST('./optimization/', 
                         train=True, 
                         transform=transforms.ToTensor(), 
                         download=False)
test_data = mnist.MNIST('./optimization/', 
                        train=False, 
                        transform=transforms.ToTensor(),
                        download=False)
train_loader = DataLoader(train_data, 
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_data, 
                         batch_size=1,
                         shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 网络搭建
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc3 = nn.Linear(64*7*7, 1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, 10)
        self.softmax4 = nn.Softmax(dim=1)

    def forward(self, x):

        # 网络各层连接
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # feature map转换成一维
        x = x.view(-1, 64*7*7)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax4(x)
        return x

net = CNN()

# 使用交叉熵
criterion = nn.CrossEntropyLoss()   
# 选择Adam优化
optimizer = optim.Adam(net.parameters(), lr=lr) 
# 学习率自适应 cos函数衰减
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

# cuda加速
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = net.to(device)

epoch_loss = []
epoch_acc = []
epo_test_acc = []

for epoch in range (epochs):
    train_loss = []
    train_acc = []
    my_loss = 0.0

    for i, data in enumerate(train_loader, 0): 
        # 训练一个batch
        # x, labels = data
        x, labels = data[0].to(device), data[1].to(device)
        # x = random_flip_and_rotate(x).to(device)
        optimizer.zero_grad()   # 上一batch的梯度清除
        outputs = net(x)    # 前向传播预测
        loss = criterion(outputs, labels)
        loss.backward()     # 反向传播，计算当前梯度
        optimizer.step()    # 更新参数

        my_loss += loss.item()  # 平均loss

        train_loss.append(loss.item())

        correct = 0.0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        train_acc.append(correct/total)
    
    scheduler.step()    # 每一个epoch学习率衰减
    epoch_loss.append(sum(train_loss)/len(train_loss))
    epoch_acc.append(sum(train_acc)/len(train_acc))

    test_acc = []
    for i, data in enumerate(test_loader, 0):
        x, labels = data[0].to(device), data[1].to(device)
        outputs = net(x)
        correct = 0.0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        # 记录预测正确
        correct = (predicted == labels).sum().item()
        test_acc.append(correct)

    epo_test_acc.append(sum(test_acc)/len(test_acc))

    print("epoch: {}\tloss: {}\ttrain_acc: {}\ttest_acc: {}"\
                .format(epoch, sum(train_loss)/len(train_loss), \
                    sum(train_acc)/len(train_acc), sum(test_acc)/len(test_acc)))

draw_train(epoch_loss, epoch_acc)
draw_test(epo_test_acc)
print(epo_test_acc.index(max(epo_test_acc)), max(epo_test_acc))
