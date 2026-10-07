"""Audio/visual conversion shared by the pack writers and the pack reader."""

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

SAMPLE_RATE = 16_000
FACE_SIZE = 112
ASD_TARGET_RMS_DBFS = -25.0
ASD_MAX_GAIN_DB = 20.0
ASD_PEAK_LIMIT = 0.99


def normalize_waveform(
    waveform: torch.Tensor,
    target_rms_dbfs: float = ASD_TARGET_RMS_DBFS,
    max_gain_db: float = ASD_MAX_GAIN_DB,
    peak_limit: float = ASD_PEAK_LIMIT,
) -> tuple[torch.Tensor, float]:
    """Remove DC and scale to the target RMS, capped by max gain and peak."""
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    rms = waveform.square().mean().sqrt()
    peak = waveform.abs().max()
    if rms <= 1e-8 or peak <= 1e-8:
        return waveform.contiguous(), 1.0

    target_rms = 10 ** (target_rms_dbfs / 20)
    gain = min(target_rms / rms.item(), 10 ** (max_gain_db / 20), peak_limit / peak.item())
    return (waveform * gain).contiguous(), gain


def load_waveform(path: str | Path) -> torch.Tensor:
    """[1, samples] float32 mono PCM in [-1, 1] at `SAMPLE_RATE`, source gain."""
    waveform, source_rate = torchaudio.load(Path(path), normalize=True)
    # Lossy codecs such as AAC can decode slightly outside nominal PCM bounds.
    waveform = waveform.clamp(-1.0, 1.0).to(torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{path}: audio contains NaN or infinite values")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform.contiguous(), source_rate, SAMPLE_RATE)
    return waveform.contiguous()


def validate_visual(array: np.ndarray) -> np.ndarray:
    """Check that faces are uint8 [T, 112, 112], as the visual frontend expects."""
    if array.ndim != 3 or tuple(array.shape[1:]) != (FACE_SIZE, FACE_SIZE):
        raise ValueError(f"visual must be [T, {FACE_SIZE}, {FACE_SIZE}], got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"visual must be uint8 in [0, 255], got {array.dtype}")
    return array
