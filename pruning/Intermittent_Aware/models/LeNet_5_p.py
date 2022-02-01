from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')

# https://github.com/pytorch/examples/tree/master/mnist
class LeNet_5_p(nn.Module):
    def __init__(self, prune):
        super(LeNet_5_p, self).__init__()
        self.prune = prune
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.ip1 = nn.Linear(2304, 128)
        self.ip2 = nn.Linear(128, 10)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.ip1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.ip2(x)
        output = F.log_softmax(x, dim=1)
        return output

