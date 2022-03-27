from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
class MNIST(nn.Module):
    def __init__(self, prune):
        super(MNIST, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.ip1 = nn.Linear(4*4*16, 10)
        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 4*4*16)
        x = self.ip1(x)
        return x
