# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

from layers.build_nefe import build_nefe


class MeshConv(nn.Module):

    def __init__(self, in_channels, out_channels, k=4, bias=True, groups=1):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, k),
            bias=bias,
            groups=groups
        )
        self._initialize_weights()

    def forward(self, ne, x):
        x = x.squeeze(-1).squeeze(0)        
        x = build_nefe(ne, x)        
        x = self.conv(x)       
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


SRM_4_18 = np.load(r'/home/jiqingzhi/GNN/GeoDAF/layers/SRM_4_18.npy')
class MRF(nn.Module):
    def __init__(self, stride=1):
        super(MRF, self).__init__()
        self.in_channels = 1
        self.out_channels = 18
        self.kernel_size = (1, 4)

        self.weight = Parameter(torch.Tensor(18, 1, 1, 4), requires_grad=True)
        self.bias = Parameter(torch.Tensor(18), requires_grad=True)
        self.groups = 1

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_4_18
        self.bias.data.zero_()

    def forward(self, ne, fe):
        fe = fe.squeeze(-1).squeeze(0)
        x = build_nefe(ne, fe)
        return F.conv2d(x, self.weight, self.bias, self.groups)

