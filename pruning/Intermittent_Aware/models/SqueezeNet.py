from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
import math
cwd = os.getcwd()
sys.path.append(cwd+'../')

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1) #squeeze
        self.relu_squeeze = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1) #expend 1x1
        self.relu_expand1x1 = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=3, padding=(1,1)) #expend 1x1
        self.relu_expand3x3 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))


    def forward(self, x):
        x = self.squeeze(x)
        x = self.relu_squeeze(x)
        return torch.cat([
            self.relu_expand1x1(self.expand1x1(x)),
            self.relu_expand3x3(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, prune):
        super(SqueezeNet, self).__init__()
        self.prune = prune
        # input_shape = [64, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire1 = Fire(64, 16, 64, 64)
        self.fire2 = Fire(128, 16, 64, 64)
        self.fire3 = Fire(128, 32, 128, 128)
        # self.fire3 = Fire(256, 32, 128, 128)
        self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(256, 10, kernel_size=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax(dim=1)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)

        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        # [128, 10, 1, 1]
        x = self.softmax(x)
        # [128, 10]
        return x
