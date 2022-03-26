from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'../')

class KWS_DNN_S(nn.Module):
    def __init__(self, prune):
        super(KWS_DNN_S, self).__init__()
        self.prune = prune
        self.ip1 = nn.Linear(250, 144)
        self.ip2 = nn.Linear(144, 144)
        self.ip3 = nn.Linear(144, 144)
        self.ip4 = nn.Linear(144, 12)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, x):
        x = x.view(x.size(0), 25*10)

        x = self.ip1(x)
        x = self.relu(x)
        x = self.ip2(x)
        x = self.relu(x)
        x = self.ip3(x)
        x = self.relu(x)
        x = self.ip4(x)
        x = self.softmax(x)
        return x
