"""Causal alignment of native-rate audio features onto the video frame grid.

Only the features are aligned; the audio itself is never resampled. The rate
ratio is implied by the two lengths: `Ta` audio features and `N` video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_frame_resample(
    features, audio_lengths, frame_lengths, align="frame_start", target_len=None
):
    """[B, Ta, D] at the encoder's own rate -> [B, N, D] on the video frame grid.

    The clip's audio spans the same duration as its N frames, so frame `n` sits
    at `t_n = n * D / N`, and feature `j` (covering `[j, j+1) * D / Ta`) has
    finished by `t_n` iff `j + 1 <= n * Ta / N`. Hence

        end_n = floor((n + offset) * Ta / N)

    and frame `n` averages features `[end_{n-1}, end_n)`. With
    `align="frame_start"` (offset 0) frame n sees only audio finished before it
    begins; `"frame_end"` (offset 1) admits audio up to the end of frame n.
    An empty span holds the last completed feature (zeros before the first), so
    encoders slower than the video get a causal zero-order hold.
    """
    if align not in ("frame_start", "frame_end"):
        raise ValueError(f"unknown align: {align!r}")

    batch, source_len, dim = features.shape
    device = features.device
    # `target_len` may exceed the longest clip when batch widths are quantised;
    # extra columns hold the last feature and are masked downstream. Each row's
    # own frame_length still sets its pooling ratio.
    target_len = int(frame_lengths.max()) if target_len is None else int(target_len)

    audio_lengths = audio_lengths.to(device)
    frame_lengths = frame_lengths.to(device)

    offset = 1.0 if align == "frame_end" else 0.0
    steps = (
        torch.arange(target_len, device=device, dtype=torch.float64) + offset
    ).unsqueeze(0)
    scale = audio_lengths.to(torch.float64).unsqueeze(1) / frame_lengths.to(
        torch.float64
    ).unsqueeze(1).clamp_min(1.0)
    ends = torch.floor(steps * scale).to(torch.int64)
    ends = ends.clamp_(min=0).minimum(audio_lengths.unsqueeze(1))
    ends = torch.cummax(ends, dim=1).values
    starts = F.pad(ends[:, :-1], (1, 0))
    counts = ends - starts

    cumulative = F.pad(features.cumsum(dim=1, dtype=torch.float32), (0, 0, 1, 0))
    gather_end = ends.unsqueeze(-1).expand(batch, target_len, dim)
    gather_start = starts.unsqueeze(-1).expand(batch, target_len, dim)
    totals = cumulative.gather(1, gather_end) - cumulative.gather(1, gather_start)
    pooled = totals / counts.clamp(min=1).unsqueeze(-1).to(totals.dtype)

    # Empty span -> hold the most recent completed feature; zeros before any.
    positions = torch.arange(target_len, device=device).expand(batch, target_len)
    source = torch.cummax(positions.masked_fill(counts == 0, -1), dim=1).values
    pooled = pooled.gather(
        1, source.clamp(min=0).unsqueeze(-1).expand(batch, target_len, dim)
    )
    pooled = pooled * (source >= 0).unsqueeze(-1).to(pooled.dtype)
    return pooled.to(features.dtype)


class CausalFrameResampler(nn.Module):
    def __init__(self, align="frame_start"):
        super().__init__()
        self.align = align

    def forward(self, features, audio_lengths, frame_lengths, target_len=None):
        return causal_frame_resample(
            features, audio_lengths, frame_lengths,
            align=self.align, target_len=target_len,
        )
