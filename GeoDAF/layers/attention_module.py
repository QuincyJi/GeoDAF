# -*-coding:utf-8-*-
import torch
from torch import nn
from layers.mesh_conv import MeshConv


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, ne, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = MeshConv(2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ne, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(ne, y)
        y = self.sigmoid(y)
        return  x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 8, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, ne, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        y = avg_out + max_out
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, ne, x):
        x = self.ca(ne, x)
        x = self.sa(ne, x)

        return x

import math
class ECALayer(nn.Module):
    """
    
    """
    def __init__(self, channel, k_size=None, gamma=2, b=1):
        super().__init__()
        # 自适应 kernel size（保证为奇数且 >=3）
        if k_size is None:
            t = int(abs((math.log2(channel) / gamma) + b))
            k_size = t if t % 2 else t + 1
            k_size = max(3, k_size)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.act  = nn.Sigmoid()

    def forward(self, ne, x):
        # x: [B,C,H,W]
        y = self.avg_pool(x)                 # [B,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)    # [B,1,C]
        y = self.conv(y)
        y = self.act(y).transpose(1, 2).unsqueeze(-1)  # [B,C,1,1]
        return x * y.expand_as(x)

class CoordAttLayer(nn.Module):
    """
    与 SELayer 对齐
    """
    def __init__(self, channel, reduction=32):
        super().__init__()
        mip = max(8, channel // reduction)

        # 共享的变换
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.ReLU(inplace=True)

        # 分别生成 H/W 方向权重
        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ne, x):
        b, c, h, w = x.size()
        # 沿宽度聚合 -> [B,C,H,1]；沿高度聚合 -> [B,C,1,W]
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B,C,W,1]

        # 拼接后共享变换
        y = torch.cat([x_h, x_w], dim=2)               # [B,C,H+W,1]
        y = self.act(self.bn1(self.conv1(y)))

        # 拆分回 H/W 两支
        x_h, x_w = torch.split(y, [h, w], dim=2)       # x_h:[B,mip,H,1], x_w:[B,mip,W,1]
        x_w = x_w.permute(0, 1, 3, 2)                  # [B,mip,1,W]

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = x * a_h * a_w
        return out

