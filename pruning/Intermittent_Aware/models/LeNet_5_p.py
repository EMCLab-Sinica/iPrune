from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet_5_p(nn.Module):
    def __init__(self, prune):
        super(LeNet_5_p, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=(2,2))
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=(2,2))
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ip1 = nn.Linear(16*7*7, 128)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(128, 84)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(84, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 16*7*7)

        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return x
