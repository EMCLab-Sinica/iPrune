from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')

class LeNet_5(nn.Module):
    def __init__(self, prune):
        super(LeNet_5, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ip1 = nn.Linear(16*5*5, 120)
        self.ip2 = nn.Linear(120, 84)
        self.ip3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 16*5*5)

        x = self.ip1(x)
        x = self.relu(x)
        x = self.ip2(x)
        x = self.relu(x)
        x = self.ip3(x)
        return x
