from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')
class HAR_CNN(nn.Module):
    def __init__(self, prune, n_channels=9, n_classes=6):
        super(HAR_CNN, self).__init__()
        self.prune = prune
        # (batch, 9, 128) -> (batch, 18, 64)
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=18, kernel_size=(1,2), stride=1, padding=(0,1))
        self.relu_conv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=2, padding=0)
        # (batch, 18, 64) -> (batch, 36, 32)
        self.conv2 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(1,2), stride=1, padding=(0,1))
        self.relu_conv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=2, padding=0)

        # (batch, 36, 32) -> (batch, 72, 16)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(1,2), stride=1, padding=(0,1))
        self.relu_conv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=2, padding=0)

        self.ip1 = nn.Linear(16*72, n_classes)

    def forward(self, x):
        # (batch, 9, 1, 128)
        x = self.conv1(x)
        x = self.relu_conv1(x)
        # (batch, 18, 128)
        x = self.pool1(x)
        # (batch, 18, 1, 64)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        # (batch, 36, 64)
        x = self.pool2(x)
        # (batch, 36, 1, 32)
        x = self.conv3(x)
        x = self.relu_conv3(x)
        # (batch, 72, 16)
        x = self.pool3(x)

        x = x.view(x.size(0), 16*72)

        x = self.ip1(x,)
        return x
