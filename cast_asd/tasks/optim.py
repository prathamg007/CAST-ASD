"""Learning-rate schedules."""

import torch


def get_cosine_warmup_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    """Linear warmup from 1% of the LR, then cosine decay to 1%. Steps per batch."""
    total_steps = max(1, int(total_steps))
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps - 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=optimizer.param_groups[0]["lr"] * 0.01
    )
    if warmup_steps == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


def get_step_decay_scheduler(optimizer, gamma=0.95):
    """TalkNet's schedule: constant LR within an epoch, x`gamma` after. Steps per epoch."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
