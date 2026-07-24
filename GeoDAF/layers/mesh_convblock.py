# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.mesh_conv import MeshConv
from layers.attention_module import (SELayer, SpatialAttention, ChannelAttention, CBAM, ECALayer, CoordAttLayer)


class EdsConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4,
                 re_sample=False, use_1x1conv=False, attention=None):
        super(EdsConv, self).__init__()

        hidden_dim = int(in_channels * expand_ratio)
        self.conv0 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(hidden_dim)
        self.conv1 = MeshConv(hidden_dim, hidden_dim, groups=hidden_dim)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv1_1 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)

    def forward(self, ne, x):

        x = F.leaky_relu(self.bn0(self.conv0(x)))
        x = F.leaky_relu(self.bn1(self.conv1(ne, x)))
        x = self.bn1_1(self.conv1_1(x))
        return x

class EdsConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4,
                 re_sample=False, attention=None):
        super(EdsConvBlock, self).__init__()

        self.conv1 = EdsConv(in_channels, out_channels, expand_ratio)
        self.conv2 = EdsConv(out_channels, out_channels, expand_ratio)

        if attention == 'SE':
            self.attention = SELayer(out_channels)
        elif attention == 'CA':
            self.attention = ChannelAttention(out_channels)
        elif attention == 'SA':
            self.attention = SpatialAttention()
        elif attention == 'CBAM':
            self.attention = CBAM(out_channels)
        elif attention == 'ECA':
            self.attention = ECALayer(out_channels)
        elif attention == 'CoordAtt':
            self.attention = CoordAttLayer(out_channels)
        elif attention == None:
            self.attention = None

        if re_sample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = None

    def forward(self, ne, x):

        temp = x
        x = self.conv1(ne, x)
        x = F.leaky_relu(x)
        x = self.conv2(ne, x)

        if self.attention != None:
           x = self.attention(ne, x)

        if self.shortcut:
            temp = self.shortcut(temp)

        x += temp
        return F.leaky_relu(x)

  