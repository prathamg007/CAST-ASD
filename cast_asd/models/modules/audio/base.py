"""Base class for the audio encoders.

An encoder takes 16 kHz audio and returns features at its own rate;
`AudioFrontend` resamples them onto the video frame grid.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module, ABC):
    #: Feature `k` covers audio in [k/rate, (k+1)/rate) and reads no later
    #: sample. Set to False only in a non-causal model; `delay_samples` is then
    #: not applied.
    causal: bool = True

    #: Input delay (samples) that cancels the encoder's internal lookahead.
    #: Checked by `tests/test_causality.py`.
    delay_samples: int = 0

    #: Resampler alignment (see `causal_frame_resample`): "frame_start" for
    #: encoders slower than the video, "frame_end" plus `delay_samples` for
    #: encoders at the video rate.
    resample_align: str = "frame_start"

    #: Input samples consumed per output feature, at 16 kHz.
    samples_per_feature: int = 160

    def __init__(self) -> None:
        super().__init__()
        self.frozen = False

    @property
    @abstractmethod
    def out_dim(self) -> int: ...

    @abstractmethod
    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, 1, S] @16 kHz -> [B, Ta, out_dim] at the encoder's own rate."""

    def feature_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """How many of the emitted features are real rather than padding."""
        return torch.div(
            audio_lengths, self.samples_per_feature, rounding_mode="floor"
        ).clamp_min(1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if self.delay_samples and self.causal:
            # Delay, then truncate to the original length so the feature count
            # is unchanged.
            waveform = F.pad(waveform, (self.delay_samples, 0))[
                ..., : waveform.shape[-1]
            ]
        if self.frozen:
            with torch.no_grad():
                features = self._encode(waveform)
        else:
            features = self._encode(waveform)
        return features.float()

    def freeze(self):
        self.frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True):
        # A frozen encoder stays in eval(), so its BatchNorm statistics never update.
        return super().train(False if self.frozen else mode)
