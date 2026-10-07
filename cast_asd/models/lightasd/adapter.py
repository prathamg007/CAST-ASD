"""Light-ASD behind `AudioVisualASD`'s interface.

Computes 13 MFCCs at 100 Hz with torchaudio (40 mel bands) and maps Light-ASD's
single two-logit head onto the three heads `ASDTask` reads. The auxiliary slots
repeat the fused logits, so their loss weights must be 0 in the config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .model import LightASD


class LightASDModel(nn.Module):
    """Light-ASD behind `AudioVisualASD`'s forward signature."""

    #: Two logits per frame, like TalkNet's head.
    head_type = "talknet"

    def __init__(self, cfg: dict):
        super().__init__()
        self.causal = bool(cfg.get("causal", True))
        self.net = LightASD(causal=self.causal)
        self.n_mfcc = 13
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16_000, n_mfcc=self.n_mfcc,
            melkwargs=dict(n_fft=512, win_length=400, hop_length=160,
                           n_mels=40, center=False),
        )

    def _features(self, waveform):
        """[B, (1,) samples] -> [B, n_mfcc, frames] at 100 Hz.

        `center=False`: a centred frame would read 12.5 ms ahead.
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return self.mfcc(waveform.float())

    def forward(self, waveform, visual, audio_lengths=None, frame_lengths=None,
                key_padding_mask=None, return_embeddings=False, source_fps=None,
                scene=None, labels=None):
        if scene is not None:
            # One waveform per scene, expanded to each of its faces.
            waveform = waveform[scene["index"]]
        audio = self._features(waveform)
        # `visual` arrives as [B, T, H, W] floats in 0-255.
        logits = self.net(visual, audio)
        target = visual.shape[1]
        if logits.shape[1] != target:
            logits = F.interpolate(logits.transpose(1, 2), size=target,
                                   mode="nearest").transpose(1, 2)
        if return_embeddings:
            # The auxiliary slots repeat the single head (loss weights must be 0).
            zeros = logits.new_zeros(logits.shape[0], target, 1)
            return logits, logits, logits, zeros, zeros, {}
        return logits, logits, logits
