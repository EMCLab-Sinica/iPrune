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
        # (batch, 128, 9) -> (batch, 64, 18)
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=18, kernel_size=2, stride=1, padding='same')
        self.relu_conv1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (batch, 64, 18) -> (batch, 32, 36)
        self.conv2 = nn.Conv1d(in_channels=18, out_channels=36, kernel_size=2, stride=1, padding='same')
        self.relu_conv2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # (batch, 32, 36) -> (batch, 16, 72)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=2, stride=1, padding='same')
        self.relu_conv3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.ip1 = nn.Linear(16*72, n_classes)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu_conv3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), 16*72)

        x = self.ip1(x,)
        return x
