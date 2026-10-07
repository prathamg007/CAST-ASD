"""Losses averaged over unpadded frames."""

import torch
import torch.nn.functional as F


def frame_weights(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`[tracks, time]` weights, uniform over valid frames and summing to 1."""
    weights = mask.to(dtype)
    return weights / weights.sum().clamp_min(1e-12)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Binary cross entropy averaged over valid frames."""
    weight = frame_weights(mask, logits.dtype)
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    return (losses * weight).sum()


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """2-class cross entropy (TalkNet's head) averaged over valid frames.

    `logits`: `[..., 2]`; `targets`/`mask`: `logits.shape[:-1]`.
    """
    weight = frame_weights(mask, logits.dtype).reshape(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1).long(),
        reduction="none",
    )
    return (losses * weight).sum()
