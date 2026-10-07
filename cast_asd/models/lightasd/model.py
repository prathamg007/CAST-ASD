"""Light-ASD (Liao et al., CVPR 2023), reimplemented with a causality switch.

Follows the released `model/Encoder.py` and `model/Classifier.py`:

  visual   3 dual-path blocks (32/64/128 ch) with pooling -> [B, T, 128]
  audio    3 dual-path blocks over MFCCs, mean over frequency -> [B, T, 128]
  fuse     addition
  temporal two unidirectional GRUs with a flip between ("BGRU"), GELU after each
  head     linear to 2 logits

Each block sums a 3-wide and a 5-wide branch, each factorised into a spatial
(or frequency) convolution then a temporal one, followed by a 1x1 projection.
`causal=True` left-pads every temporal convolution and drops the backward GRU;
`causal=False` is the published model. Run this module to print the parameter
count (the paper reports ~1.0M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN = dict(momentum=0.01, eps=0.001)          # the released model's BatchNorm args


class _TemporalConv(nn.Module):
    """A temporal convolution that is left-padded when causal, centred when not."""

    def __init__(self, ch_in, ch_out, width, causal, spatial_dims):
        super().__init__()
        self.causal = causal
        self.width = width
        self.spatial_dims = spatial_dims          # 2 for video (H, W), 1 for audio (F)
        kernel = (width,) + (1,) * spatial_dims
        conv = nn.Conv3d if spatial_dims == 2 else nn.Conv2d
        self.conv = conv(ch_in, ch_out, kernel, padding=0, bias=False)
        self.norm = (nn.BatchNorm3d if spatial_dims == 2 else nn.BatchNorm2d)(ch_out, **BN)

    def forward(self, x):
        total = self.width - 1
        pre, post = (total, 0) if self.causal else (total // 2, total - total // 2)
        pad = (0, 0) * self.spatial_dims + (pre, post)
        return self.norm(self.conv(F.pad(x, pad)))


class _Branch(nn.Module):
    """One path of a dual-path block: spatial convolution, then temporal."""

    def __init__(self, ch_in, ch_out, width, causal, spatial_dims, stride):
        super().__init__()
        conv = nn.Conv3d if spatial_dims == 2 else nn.Conv2d
        norm = nn.BatchNorm3d if spatial_dims == 2 else nn.BatchNorm2d
        pad = width // 2
        spatial_k = (1,) + (width,) * spatial_dims
        spatial_p = (0,) + (pad,) * spatial_dims
        spatial_s = (1,) + (stride,) * spatial_dims
        self.spatial = conv(ch_in, ch_out, spatial_k, stride=spatial_s,
                            padding=spatial_p, bias=False)
        self.spatial_norm = norm(ch_out, **BN)
        self.temporal = _TemporalConv(ch_out, ch_out, width, causal, spatial_dims)

    def forward(self, x):
        return self.temporal(F.relu(self.spatial_norm(self.spatial(x))))


class _Block(nn.Module):
    """Dual-path (3-wide + 5-wide), summed, then a 1x1 projection."""

    def __init__(self, ch_in, ch_out, causal, spatial_dims, stride=1):
        super().__init__()
        self.a = _Branch(ch_in, ch_out, 3, causal, spatial_dims, stride)
        self.b = _Branch(ch_in, ch_out, 5, causal, spatial_dims, stride)
        conv = nn.Conv3d if spatial_dims == 2 else nn.Conv2d
        norm = nn.BatchNorm3d if spatial_dims == 2 else nn.BatchNorm2d
        self.project = conv(ch_out, ch_out, 1, bias=False)
        self.project_norm = norm(ch_out, **BN)

    def forward(self, x):
        return F.relu(self.project_norm(self.project(F.relu(self.a(x) + self.b(x)))))


class LightASDVisual(nn.Module):
    """[B, T, 112, 112] grey face crops -> [B, T, 128]."""

    def __init__(self, causal=True):
        super().__init__()
        self.block1 = _Block(1, 32, causal, spatial_dims=2, stride=2)
        self.block2 = _Block(32, 64, causal, spatial_dims=2)
        self.block3 = _Block(64, 128, causal, spatial_dims=2)
        self.pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):                          # [B, T, H, W]
        x = ((x / 255.0) - 0.4161) / 0.1688        # the released normalisation
        x = x.unsqueeze(1)                         # [B, 1, T, H, W]
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        x = F.adaptive_max_pool3d(x, (x.shape[2], 1, 1))
        return x.squeeze(-1).squeeze(-1).transpose(1, 2)     # [B, T, 128]


class LightASDAudio(nn.Module):
    """[B, F, T] mel/MFCC -> [B, T, 128]."""

    def __init__(self, causal=True):
        super().__init__()
        self.block1 = _Block(1, 32, causal, spatial_dims=1)
        self.block2 = _Block(32, 64, causal, spatial_dims=1)
        self.block3 = _Block(64, 128, causal, spatial_dims=1)
        self.pool = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x):                          # [B, F, T]
        x = x.transpose(1, 2).unsqueeze(1)         # [B, 1, T, F]
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        return x.mean(dim=3).transpose(1, 2)       # mean over frequency -> [B, T, 128]


class BGRU(nn.Module):
    """The released Classifier: forward GRU, GELU, then (non-causal only) a
    second GRU over the flipped sequence, flipped back, GELU."""

    def __init__(self, channel, causal=True):
        super().__init__()
        self.causal = causal
        self.forward_gru = nn.GRU(channel, channel, num_layers=1, batch_first=True)
        self.backward_gru = None if causal else nn.GRU(
            channel, channel, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.forward_gru(x)
        x = F.gelu(x)
        if self.backward_gru is not None:
            x = torch.flip(x, dims=[1])
            x, _ = self.backward_gru(x)
            x = torch.flip(x, dims=[1])
            x = F.gelu(x)
        return x


class LightASD(nn.Module):
    """[B, T, H, W] faces + [B, F, T] MFCCs -> [B, T, 2] logits."""

    def __init__(self, causal=True, channels=128):
        super().__init__()
        self.causal = causal
        self.visual = LightASDVisual(causal)
        self.audio = LightASDAudio(causal)
        self.gru = BGRU(channels, causal)
        self.head = nn.Linear(channels, 2)

    def forward(self, visual, audio):
        v, a = self.visual(visual), self.audio(audio)
        if a.shape[1] != v.shape[1]:               # align to the video grid
            a = F.interpolate(a.transpose(1, 2), size=v.shape[1],
                              mode="nearest").transpose(1, 2)
        return self.head(self.gru(v + a))


if __name__ == "__main__":
    for causal in (True, False):
        m = LightASD(causal=causal)
        n = sum(p.numel() for p in m.parameters())
        print(f"causal={causal!s:<6} parameters {n/1e6:.3f}M   "
              f"(the paper reports ~1.0M)")
