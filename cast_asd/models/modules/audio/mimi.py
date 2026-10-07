"""Kyutai Mimi (moshi) as a frozen causal audio encoder."""

import torch
import torch.nn.functional as F
import torchaudio

from .base import AudioEncoder

MIMI_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_WEIGHTS = "tokenizer-e351c8d8-checkpoint125.safetensors"

# torchaudio's sinc resampler is symmetric: at 16->24 kHz, output sample m reads
# input up to m*2/3 + 6 (measured). A 16-sample (1 ms) delay covers that.
_RESAMPLE_DELAY_16K = 16


def causal_resample_16k_to_24k(waveform: torch.Tensor) -> torch.Tensor:
    """16 -> 24 kHz without reading the future.

    Delaying the input by `_RESAMPLE_DELAY_16K` samples and keeping the nominal
    output length makes output m depend on input up to m*2/3 - 10 < m*2/3.
    """
    length = round(waveform.shape[-1] * 24_000 / 16_000)
    padded = F.pad(waveform, (_RESAMPLE_DELAY_16K, 0))
    return torchaudio.functional.resample(padded, 16_000, 24_000)[..., :length]


class MimiAudioEncoder(AudioEncoder):
    """Mimi's pretrained encoder, tapped for continuous (unquantised) latents.

    Mimi is causal by construction (causal SEANet convolutions, causal encoder
    transformer, causal 2x downsample). Output: 512-d latents at 12.5 Hz.
    """

    #: 1920 samples at 24 kHz = 1280 at 16 kHz.
    samples_per_feature = 1280

    def __init__(self, weights=None):
        super().__init__()
        try:
            from moshi.models import loaders
        except ImportError as error:
            raise ImportError(
                "asd.audio.name=mimi needs moshi: pip install -e '.[mimi]', or "
                "pip install --no-deps moshi==0.2.12 && pip install safetensors "
                "sentencepiece 'huggingface-hub<1.0' einops"
            ) from error

        if weights is None:
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(MIMI_REPO, MIMI_WEIGHTS)

        mimi = loaders.get_mimi(weights, device="cpu", num_codebooks=8)
        if not getattr(mimi, "causal", True):
            raise ValueError("Mimi was built non-causal")
        mimi.decoder = None
        mimi.decoder_transformer = None
        self.mimi = mimi
        self.frame_size = int(mimi.frame_size)

    @property
    def out_dim(self) -> int:
        return 512

    def _encode(self, waveform):
        audio = causal_resample_16k_to_24k(waveform)
        # Right-padding to a whole Mimi frame is causal.
        remainder = (-audio.shape[-1]) % self.frame_size
        if remainder:
            audio = F.pad(audio, (0, remainder))
        # moshi's torch.compile needs Triton (unavailable on Windows); the
        # reported runs used the uncompiled encoder.
        from moshi.utils.compile import no_compile

        with no_compile():
            latent = self.mimi.encode_to_latent(audio, quantize=False)   # [B, 512, T]
        return latent.transpose(1, 2)
