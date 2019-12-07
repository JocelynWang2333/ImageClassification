import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)  ###分类哺乳类和禽类
        self.softmax1 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(150, 3)  ####分类三个种类
        self.softmax2 = nn.Softmax(dim=1)

        # 前向传播网络
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # print(x.shape)
        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x1 = F.dropout(x, training=self.training)

        # 通过第一个softmax生成classes的预测概率
        x_classes_output = self.fc2(x1)
        x_classes_output = self.softmax1(x_classes_output)

        # 通过第二个softmax生成species的预测概率
        x_species = self.fc3(x1)
        x_species = self.softmax2(x_species)

        out = []  ###两个预测结果放到out中
        out.append(x_classes_output)
        out.append(x_species)
        return out
