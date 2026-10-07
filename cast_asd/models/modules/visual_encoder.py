"""TalkNet-ASD's visual encoder with causal temporal padding.

3D-conv front end + per-frame ResNet-18 (from deep_avsr) -> V-TCN -> 1D convs.
Every temporal convolution pads on the left only unless `causal=False`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CConv1d(nn.Conv1d):
    """A 1-D convolution padded on the left only, so output t reads inputs <= t.

    `causal=False` splits the same padding either side (the original model).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        total = (kernel_size - 1) * dilation
        pad = (total, 0) if causal else (total // 2, total - total // 2)
        self.pad = nn.ConstantPad1d(padding=pad, value=0)

    def forward(self, input):
        return super().forward(self.pad(input))


class CausalConv3d(nn.Conv3d):
    """[batch, channel, time, height, width]; time is padded on the left only.

    Spatial padding comes from the `padding` kwarg; time is padded explicitly by
    `kernel_size[0] - 1` (split either side when `causal=False`).
    """

    def __init__(self, *args, causal: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal = causal

    def forward(self, x):
        total = (self.kernel_size[0] - 1) * self.dilation[0]
        if self.causal:
            x = F.pad(x, (0, 0, 0, 0, total, 0))          # left only
        else:
            x = F.pad(x, (0, 0, 0, 0, total // 2, total - total // 2))
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, self.padding[1], self.padding[2]),
            self.dilation,
            self.groups,
        )


class VisualEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.frontend = visualFrontend(causal=causal)
        self.tcn = visualTCN(causal=causal)
        self.conv1d = visualConv1D(out_dim=out_dim, causal=causal)

        self.register_buffer("mean", torch.tensor(0.4161))
        self.register_buffer("std", torch.tensor(0.1688))

    def forward(self, x):
        B, T, H, W = x.shape
        x = (x.view(B, 1, T, H, W).float() / 255.0 - self.mean) / self.std
        x = self.frontend(x)
        x = x.view(B, T, -1).transpose(1, 2)
        x = self.tcn(x)
        x = self.conv1d(x)
        return x.transpose(1, 2)


class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stride = stride
        self.downsample = nn.Conv2d(
            inplanes, outplanes, kernel_size=(1, 1), stride=stride, bias=False
        )
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch


class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 4), stride=(1, 1))

        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class CausalLayerNorm(nn.Module):
    """LayerNorm over channels at each timestep of a [B, C, T] tensor."""

    def __init__(self, channel_size, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        self.eps = eps

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        return self.gamma * y_norm + self.beta


class visualFrontend(nn.Module):
    """[B, 1, T, H, W] -> [B, T, 512]: a 5-frame causal 3D conv, then ResNet-18
    per frame. TalkNet-ASD's `visualFrontend`."""

    def __init__(self, causal: bool = True):
        super(visualFrontend, self).__init__()
        self.frontend3D = nn.Sequential(
            CausalConv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                         bias=False, causal=causal),
            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet()
        return

    def forward(self, inputBatch):
        # inputBatch: [B, 1, T, H, W]
        batchsize, _, T = inputBatch.shape[:3]
        batch = self.frontend3D(inputBatch)          # [B, 64, T, H', W']
        batch = batch.transpose(1, 2)                 # [B, T, 64, H', W']
        batch = batch.reshape(batchsize * T, batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)               # [B*T, 512, 1, 1]
        outputBatch = outputBatch.reshape(batchsize, T, 512)
        return outputBatch


class DSConv1d(nn.Module):
    """Depthwise-separable residual block of TalkNet-ASD's V-TCN."""

    def __init__(self, causal: bool = True):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            CConv1d(512, 512, 3, stride=1, dilation=1, groups=512, bias=False, causal=causal),
            nn.PReLU(),
            CausalLayerNorm(512),
            CConv1d(512, 512, 1, bias=False, causal=causal),
        )

    def forward(self, x):
        out = self.net(x)
        return out + x


class visualTCN(nn.Module):
    def __init__(self, causal: bool = True):
        super(visualTCN, self).__init__()
        self.net = nn.Sequential(*[DSConv1d(causal=causal) for _ in range(5)])

    def forward(self, x):
        out = self.net(x)
        return out


class visualConv1D(nn.Module):
    def __init__(self, out_dim: int = 256, causal: bool = True):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            CConv1d(512, 256, 5, stride=1, causal=causal),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            CConv1d(256, out_dim, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out
