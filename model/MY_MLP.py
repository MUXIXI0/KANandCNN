import torch.nn.functional as F
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Lay1 = nn.Linear(28*28, 256)
        self.Lay2 = nn.Linear(256, 64)
        self.Lay3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.Lay1(x)
        x =  F.tanh(x)
        x = F.tanh(self.Lay2(x))
        x = F.tanh(self.Lay3(x))
        return x
