"""TalkNet-style train-only augmentation (TalkNet-ASD dataLoader.py).

Neither operation changes timing: SNR mixing keeps the clip length and the
visual transforms act frame by frame.
"""

import random

import cv2
import numpy as np
import torch


def augment_visual(faces: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
    """Apply one of orig/flip/crop/rotate, drawn once per clip, to every frame.

    One draw per clip (as TalkNet's `load_visual`) avoids spurious motion.
    """
    rng = rng or random
    height = faces.shape[1]
    aug_type = rng.choice(["orig", "flip", "crop", "rotate"])
    if aug_type == "orig":
        return faces

    out = np.empty_like(faces)
    if aug_type == "flip":
        for index, face in enumerate(faces):
            out[index] = cv2.flip(face, 1)
    elif aug_type == "crop":
        new_size = int(height * rng.uniform(0.7, 1.0))
        x = rng.randint(0, height - new_size)
        y = rng.randint(0, height - new_size)
        for index, face in enumerate(faces):
            out[index] = cv2.resize(
                face[y : y + new_size, x : x + new_size], (height, height)
            )
    else:  # rotate
        matrix = cv2.getRotationMatrix2D(
            (height / 2, height / 2), rng.uniform(-15, 15), 1
        )
        for index, face in enumerate(faces):
            out[index] = cv2.warpAffine(face, matrix, (height, height))
    return out


def mix_audio_overlap(
    waveform: torch.Tensor, noise: torch.Tensor, rng: random.Random | None = None
) -> torch.Tensor:
    """Mix in another clip's audio at a random SNR in [-5, 5] dB (TalkNet's `overlap`).

    Works on float PCM in [-1, 1] rather than int16, hence the clamp range;
    the dB computation is TalkNet's.
    """
    rng = rng or random
    length = waveform.shape[-1]
    if noise.shape[-1] < length:
        reps = -(-length // max(noise.shape[-1], 1))
        noise = noise.repeat(1, reps)
    noise = noise[..., :length]

    snr_db = rng.uniform(-5, 5)
    noise_db = 10 * torch.log10(noise.double().square().mean() + 1e-8)
    clean_db = 10 * torch.log10(waveform.double().square().mean() + 1e-8)
    scale = (10 ** ((clean_db - noise_db - snr_db) / 10)).sqrt()

    mixed = waveform.double() + scale * noise.double()
    return mixed.clamp(-1.0, 1.0).to(waveform.dtype)
