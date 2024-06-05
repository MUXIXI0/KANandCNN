#  Residual Network（ResNet）网络模型
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, padding=1)
        self.Tanh = nn.Tanh()
    def forward(self, x):
        y = self.Tanh(self.conv1(x))
        y = self.conv2(y)
        return self.Tanh(x + y)  # 两次卷积后的输出y，加上两次卷积前的输入x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)  # ResNet网络不改变输入输出维度
        self.rblock2 = ResidualBlock(32)
        self.fc = nn.Linear(512, 10)
        self.Tanh = nn.Tanh()
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp( self.Tanh(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp( self.Tanh(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

