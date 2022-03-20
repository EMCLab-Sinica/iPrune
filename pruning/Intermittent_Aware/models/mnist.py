from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://github.com/pytorch/examples/blob/main/mnist/main.py
class MNIST(nn.Module):
    def __init__(self, prune):
        super(MNIST, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.relu_conv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.relu_conv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.ip1 = nn.Linear(5*5*64, 128)
        self.relu_ip1 = nn.ReLU()
        self.ip2 = nn.Linear(128, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), 5*5*64)

        x = self.ip1(x)
        x = self.dropout2(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        return x
