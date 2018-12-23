'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, n_blocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        n_channels = 2*growth_rate
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_layers(n_channels, n_blocks[0])
        n_channels += n_blocks[0]*growth_rate
        out_channels = int(math.floor(n_channels*reduction))
        self.trans1 = Transition(n_channels, out_channels)
        n_channels = out_channels

        self.dense2 = self._make_layers(n_channels, n_blocks[1])
        n_channels += n_blocks[1]*growth_rate
        out_channels = int(math.floor(n_channels*reduction))
        self.trans2 = Transition(n_channels, out_channels)
        n_channels = out_channels

        self.dense3 = self._make_layers(n_channels, n_blocks[2])
        n_channels += n_blocks[2]*growth_rate
        out_channels = int(math.floor(n_channels*reduction))
        self.trans3 = Transition(n_channels, out_channels)
        n_channels = out_channels

        self.dense4 = self._make_layers(n_channels, n_blocks[3])
        n_channels += n_blocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(n_channels)
        self.linear = nn.Linear(n_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, in_channels, n_block):
        layers = []
        for i in range(n_block):
            layers.append(Bottleneck(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet([6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet([6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet([6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet([6,12,36,24], growth_rate=48)

