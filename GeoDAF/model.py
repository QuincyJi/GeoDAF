# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers.build_nefe import build_nefe
from layers.mesh_conv import MeshConv, MRF
from layers.mesh_convblock import EdsConvBlock


class MeshCNN_Stego_divide3_IR_ALL4(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.MRF_F_0 = MRF()
        self.bn1_0 = nn.BatchNorm2d(18)
        self.MRF_L_0 = MeshConv(1, 18, groups=1)
        self.bn2_0 = nn.BatchNorm2d(18)

        self.MRF_F_1 = MRF()
        self.bn1_1 = nn.BatchNorm2d(18)
        self.MRF_L_1 = MeshConv(1, 18, groups=1)
        self.bn2_1 = nn.BatchNorm2d(18)

        self.MRF_F_2 = MRF()
        self.bn1_2 = nn.BatchNorm2d(18)
        self.MRF_L_2 = MeshConv(1, 18, groups=1)
        self.bn2_2 = nn.BatchNorm2d(18)

        self.EdsConvBlock1_0 = EdsConvBlock(36, 64, attention=None, re_sample=True)
        self.EdsConvBlock2_0 = EdsConvBlock(64, 64, attention=None)
        self.EdsConvBlock3_0 = EdsConvBlock(64, 128, attention=None, re_sample=True)
        self.EdsConvBlock4_0 = EdsConvBlock(128, 128, attention=None)

        self.EdsConvBlock1_1 = EdsConvBlock(36, 64, attention=None, re_sample=True)
        self.EdsConvBlock2_1 = EdsConvBlock(64, 64, attention=None)
        self.EdsConvBlock3_1 = EdsConvBlock(64, 128, attention=None, re_sample=True)
        self.EdsConvBlock4_1 = EdsConvBlock(128, 128, attention=None)

        self.EdsConvBlock1_2 = EdsConvBlock(36, 64, attention=None, re_sample=True)
        self.EdsConvBlock2_2 = EdsConvBlock(64, 64, attention=None)
        self.EdsConvBlock3_2 = EdsConvBlock(64, 128, attention=None, re_sample=True)
        self.EdsConvBlock4_2 = EdsConvBlock(128, 128, attention=None)

        self.EdsConvBlock5 = EdsConvBlock(128, 256, attention='SE', re_sample=True)
        self.EdsConvBlock6 = EdsConvBlock(256, 256, attention='SE')
        self.EdsConvBlock7 = EdsConvBlock(256, 512, attention='SE', re_sample=True)

        self.gp = nn.AdaptiveAvgPool2d(1)

        self.lin1 = nn.Linear(512, 100)
        self.lin2 = nn.Linear(100, 1)

    def forward(self, ne, fe):
        # 输入
        x_0 = fe[:, 0, :].unsqueeze(1).unsqueeze(-1)
        x_1 = fe[:, 1, :].unsqueeze(1).unsqueeze(-1)
        x_2 = fe[:, 2, :].unsqueeze(1).unsqueeze(-1)

        # MRF
        x1 = self.MRF_F_0(ne, x_0)
        x1 = self.bn1_0(x1)
        x1 = F.leaky_relu(x1)
        x2 = self.MRF_L_0(ne, x_0)
        x2 = self.bn2_0(x2)
        x2 = F.leaky_relu(x2)
        x_0 = torch.cat([x1, x2], dim=1)
        # MRF
        x1 = self.MRF_F_1(ne, x_1)
        x1 = self.bn1_1(x1)
        x1 = F.leaky_relu(x1)
        x2 = self.MRF_L_1(ne, x_1)
        x2 = self.bn2_1(x2)
        x2 = F.leaky_relu(x2)
        x_1 = torch.cat([x1, x2], dim=1)
        # MRF
        x1 = self.MRF_F_2(ne, x_2)
        x1 = self.bn1_2(x1)
        x1 = F.leaky_relu(x1)
        x2 = self.MRF_L_2(ne, x_2)
        x2 = self.bn2_2(x2)
        x2 = F.leaky_relu(x2)
        x_2 = torch.cat([x1, x2], dim=1)

        # SNEM
        x = self.EdsConvBlock1_0(ne, x_0)
        x = self.EdsConvBlock2_0(ne, x)
        x = self.EdsConvBlock3_0(ne, x)
        x_0 = self.EdsConvBlock4_0(ne, x)

        x = self.EdsConvBlock1_1(ne, x_1)
        x = self.EdsConvBlock2_1(ne, x)
        x = self.EdsConvBlock3_1(ne, x)
        x_1 = self.EdsConvBlock4_1(ne, x)

        x = self.EdsConvBlock1_2(ne, x_2)
        x = self.EdsConvBlock2_2(ne, x)
        x = self.EdsConvBlock3_2(ne, x)
        x_2 = self.EdsConvBlock4_2(ne, x)

        x = x_0 + x_1 + x_2

        # CDFEM
        x = self.EdsConvBlock5(ne, x)
        x = self.EdsConvBlock6(ne, x)
        x = self.EdsConvBlock7(ne, x)

        x = self.gp(x)
        x = x.squeeze(-1).squeeze(-1)

        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        return x

