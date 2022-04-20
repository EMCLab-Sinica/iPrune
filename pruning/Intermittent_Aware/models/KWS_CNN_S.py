from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')

class KWS_CNN_S(nn.Module):
    def __init__(self, prune, n_channels=1):
        super(KWS_CNN_S, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=28, kernel_size=(10,4), stride=1)
        self.bn1 = nn.BatchNorm2d(28)
        # NCHW(1, 28, 40, 7)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=30, kernel_size=(10,4), stride=(2, 1))
        self.bn2 = nn.BatchNorm2d(30)
        # NCHW(1, 30, 16, 4)
        self.ip1 = nn.Linear(16*4*30, 16)
        self.ip2 = nn.Linear(16, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.ip3 = nn.Linear(128, 12)
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(x.size(0), 16*4*30)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.ip3(x)
        return x
