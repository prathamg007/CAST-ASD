"""Causal SE-ResNet-34 audio encoder.

Adapted from TaoRuijie/TalkNet-ASD's `audioEncoder` (MIT). Causal because
convolutions pad time on the left only and the SE squeeze pools over frequency
only, never over time. BatchNorm in train mode is the one train-time leak.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Conv2d):
    """Input layout [batch, channel, frequency, time]."""

    def forward(self, x):
        frequency_padding = ((self.kernel_size[0] - 1) * self.dilation[0]) // 2
        time_padding = (self.kernel_size[1] - 1) * self.dilation[1]
        x = F.pad(x, (time_padding, 0, frequency_padding, frequency_padding))
        return F.conv2d(
            x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
        )


class SELayer(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Squeeze over frequency ONLY -- pooling over time would leak the future.
        y = x.mean(dim=2)
        y = self.fc(y.transpose(1, 2)).transpose(1, 2)
        return x * y.unsqueeze(2)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = CausalConv2d(inplanes, planes, 3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CausalConv2d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class CausalSEResNet(nn.Module):
    def __init__(self, layers, num_filters):
        super().__init__()
        self.inplanes = num_filters[0]
        self.conv1 = CausalConv2d(1, num_filters[0], 7, stride=(2, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(num_filters[0], layers[0])
        self.layer2 = self._make_layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(num_filters[3], layers[3], stride=(1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                CausalConv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [SEBasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.mean(x, dim=2)          # collapse frequency
        return x.transpose(1, 2)          # [B, T, C]
