"""Scene mixers: let the faces of a scene inform each other.

Each mixer maps the fused per-track stream `[tracks, time, dim]` plus the scene
grouping to the same shape. Unless `causal: false`, frame `t` of any track reads
no later frame of any track (`tests/test_scene_causality.py`).

    cast      CAST-ASD: attention across the faces at each frame, then causal
              self-attention over time within each face, `n_layers` times.
    loconet   LoCoNet's inter-speaker block, reimplemented with causal padding
              (following SJTUwxz/LoCoNet_ASD @ 68d90c8): SIM (convolution over
              a fixed `s`-face roster) then LIM (self-attention over time),
              `n_layers` times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, TransformerModel, ffn_block


class SceneMixer(nn.Module):
    """Scatter to `[scene, slot, time, dim]`, mix, gate, gather back.

    Subclasses implement `mix`. With `gated: true` a zero-initialised scalar
    gates the block output against its input, so the block starts as the
    identity. The gate wraps the whole block because the sublayers are Post-LN:
    zeroing a residual branch would still leave a LayerNorm on the path.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.dim = cfg["d_model"]
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.1)))
        self.gate = nn.Parameter(torch.zeros(())) if cfg.get("gated", True) else None

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def mix(self, dense: torch.Tensor, track_mask: torch.Tensor) -> torch.Tensor:
        """`[scene, slot, time, dim]` -> same shape."""
        raise NotImplementedError

    def forward(self, x, scene_index, track_slot, track_mask):
        tracks, time, dim = x.shape
        scenes, max_tracks = track_mask.shape
        if dim != self.dim:
            raise ValueError(f"expected width {self.dim}, got {dim}")

        dense = x.new_zeros(scenes, max_tracks, time, dim)
        dense[scene_index, track_slot] = x
        mixed = self.mix(dense, track_mask)

        if self.gate is not None:
            g = self.gate.to(mixed.dtype)
            mixed = dense + g * (mixed - dense)
        return mixed[scene_index, track_slot]


class LoCoNetSIM(nn.Module):
    """LoCoNet's `ConvLayer` (`model/convLayer.py`), with causal time padding.

        conv2d       = Conv2d(dim, dim * s, (s, k))     no padding on the slot axis
        out = conv -> view(b, c, s, t) -> LN -> 1x1 -> GELU -> 1x1 + identity

    The kernel height equals the speaker count `s`, so one layer mixes exactly
    `s` slots all-to-all. Upstream pads time symmetrically (`k // 2` frames of
    lookahead); here time is left-padded by `k - 1` unless `causal=False`.
    """

    def __init__(self, dim: int, num_speakers: int, time_kernel: int = 7,
                 causal: bool = True):
        super().__init__()
        self.dim = dim
        self.num_speakers = num_speakers
        self.time_kernel = time_kernel
        self.causal = causal
        # Time padding is applied in `forward` so it can be left-only.
        self.conv2d = nn.Conv2d(dim, dim * num_speakers, (num_speakers, time_kernel))
        self.ln = nn.LayerNorm(dim)
        self.conv2d_1x1 = nn.Conv2d(dim, dim * 2, (1, 1))
        self.conv2d_1x1_2 = nn.Conv2d(dim * 2, dim, (1, 1))
        self.gelu = nn.GELU()

    def forward(self, x):
        """`[groups, s, time, dim]` -> same shape."""
        groups, slots, time, dim = x.shape
        if slots != self.num_speakers:
            raise ValueError(
                f"LoCoNet's SIM takes exactly {self.num_speakers} slots; got {slots}"
            )
        identity = x
        out = x.permute(0, 3, 1, 2)                      # [g, dim, s, t]
        if self.causal:
            out = F.pad(out, (self.time_kernel - 1, 0, 0, 0))
        else:
            left = self.time_kernel // 2
            out = F.pad(out, (left, self.time_kernel - 1 - left, 0, 0))
        out = self.conv2d(out)                           # [g, dim * s, 1, t]
        out = out.view(groups, dim, slots, time)         # upstream's channel layout
        out = out.permute(0, 2, 3, 1)                    # [g, s, t, dim]
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)                    # [g, dim, s, t]
        out = self.conv2d_1x1(out)
        out = self.gelu(out)
        out = self.conv2d_1x1_2(out)
        out = out.permute(0, 2, 3, 1)                    # [g, s, t, dim]
        return out + identity


class LoCoNetMixer(SceneMixer):
    """LoCoNet's audio-visual backend: `n_layers` x (SIM, then LIM).

    LIM is upstream's `attentionLayer` (Post-LN self-attention over time plus a
    ReLU FFN, with biases and PyTorch's default init), masked causally unless
    `causal: false`.

    Roster: the SIM only accepts exactly `s` slots, so, as in upstream's
    `get_speaker_context`, each face is scored in its own group of `s`: itself
    at slot 0, then `s - 1` other faces of the scene in shuffled order, padded
    by resampling (or by copies of itself when alone). Only slot 0 is read back.
    Upstream shuffles in eval too; `roster_shuffle: false` uses slot order
    instead, for deterministic tests.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dim = self.dim
        layers = int(cfg.get("n_layers", 3))
        self.num_speakers = int(cfg.get("num_speakers", 3))
        self.time_kernel = int(cfg.get("time_kernel", 7))
        self.roster_shuffle = bool(cfg.get("roster_shuffle", True))
        causal = bool(cfg.get("causal", True))

        lim = {
            "d_model": dim,
            "m_ff": int(cfg.get("m_ff", 4)),
            "n_heads": int(cfg.get("n_heads", 8)),
            "n_layers": 1,
            "dropout": float(cfg.get("dropout", 0.1)),
            "causal": causal,
            "bias": True,
            "init": "pytorch_default",
        }
        self.sim = nn.ModuleList(
            LoCoNetSIM(dim, self.num_speakers, self.time_kernel, causal)
            for _ in range(layers)
        )
        self.lim = nn.ModuleList(TransformerModel(lim) for _ in range(layers))

    def _build_roster(self, track_mask):
        """`[scenes, slots]` alive mask -> `[scenes, slots, s]` slot indices.

        Row `(scene, i)` is the group face `i` is scored in. Rows for dead slots
        hold valid indices and are discarded by the gather in `forward`.
        """
        scenes, slots = track_mask.shape
        device = track_mask.device
        s = self.num_speakers

        # Alive slots first, in random order when shuffling.
        if self.roster_shuffle:
            score = torch.rand(scenes, slots, device=device)
        else:
            score = (
                torch.arange(slots, device=device, dtype=torch.float)
                .expand(scenes, slots).contiguous()
            )
        score = score.masked_fill(~track_mask, float("inf"))
        order = score.argsort(dim=1)                                  # [S, slots]

        positions = torch.arange(slots, device=device).expand(scenes, slots)
        rank = torch.empty_like(order).scatter_(1, order, positions)

        # Candidates available to each target: every other alive face.
        n_candidates = (track_mask.sum(1) - 1).clamp(min=0).view(scenes, 1, 1)

        k = torch.arange(s - 1, device=device).view(1, 1, s - 1)
        if s > 1:
            # Beyond the candidate count, resample with replacement.
            span = n_candidates.clamp(min=1)
            if self.roster_shuffle:
                draw = (torch.rand(scenes, slots, s - 1, device=device) * span).long()
                draw = draw.clamp(max=(span - 1).expand_as(draw))
            else:
                draw = k.expand(scenes, slots, s - 1) % span
            pick = torch.where(k < n_candidates, k.expand_as(draw), draw)
            # Index into "alive faces except me" -> real slot: skip own rank.
            pick = pick + (pick >= rank.view(scenes, slots, 1)).long()
            pick = pick.clamp(max=slots - 1)
            neighbours = torch.gather(
                order.unsqueeze(1).expand(scenes, slots, slots), 2, pick
            )
        else:
            neighbours = positions.new_zeros(scenes, slots, 0)

        target = positions.reshape(scenes, slots, 1)
        # A face alone in its scene is grouped with copies of itself.
        neighbours = torch.where(
            n_candidates > 0, neighbours, target.expand_as(neighbours)
        )
        return torch.cat([target, neighbours], dim=2)

    def mix(self, dense, track_mask):
        scenes, slots, time, dim = dense.shape
        s = self.num_speakers

        roster = self._build_roster(track_mask)
        index = roster.reshape(scenes, slots * s, 1, 1).expand(-1, -1, time, dim)
        grouped = torch.gather(dense, 1, index).reshape(scenes * slots, s, time, dim)

        for sim, lim in zip(self.sim, self.lim):
            grouped = sim(grouped)
            flat = grouped.reshape(scenes * slots * s, time, dim)
            grouped = lim(flat).reshape(scenes * slots, s, time, dim)

        # Position 0 is the target face.
        return grouped[:, 0].reshape(scenes, slots, time, dim)


class CASTBlock(SceneMixer):
    """CAST: a speaker pass and a time pass, repeated `n_layers` times.

    Speaker pass: Post-LN attention across the faces of a scene, separately at
    each frame, then an FFN. Padded slots are masked and slots carry no order,
    so the pass is permutation-equivariant. Time pass: a one-layer
    `TransformerModel` within each face, causal unless `causal: false`.
    `order` is `speaker_time` (default) or `time_speaker`.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        dim = self.dim
        heads = int(cfg.get("n_heads", 8))
        dropout = float(cfg.get("dropout", 0.1))
        m_ff = int(cfg.get("m_ff", 4))
        layers = int(cfg.get("n_layers", 1))
        self.order = str(cfg.get("order", "speaker_time"))
        if self.order not in ("speaker_time", "time_speaker"):
            raise ValueError(
                f"asd.scene.order must be speaker_time or time_speaker; got {self.order!r}"
            )

        self.face_attn = nn.ModuleList(
            MultiHeadAttention(dim, heads, dropout) for _ in range(layers)
        )
        self.ln_face = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.ffn = nn.ModuleList(
            ffn_block(dim, m_ff * dim, dropout=dropout) for _ in range(layers)
        )
        self.ln_ffn = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))

        temporal = {
            "d_model": dim,
            "m_ff": m_ff,
            "n_heads": heads,
            "n_layers": 1,
            "dropout": dropout,
            "causal": bool(cfg.get("causal", True)),
        }
        self.temporal = nn.ModuleList(
            TransformerModel(temporal) for _ in range(layers)
        )
        self.apply(self._init_weights)

    def _sublayer(self, x, norm, fn):
        return norm(x + self.dropout(fn(x)))

    def mix(self, dense, track_mask):
        scenes, slots, time, dim = dense.shape

        bias = torch.zeros(scenes, slots, device=dense.device, dtype=dense.dtype)
        bias.masked_fill_(~track_mask, float("-inf"))
        bias = bias.unsqueeze(1).expand(scenes, time, slots)
        bias = bias.reshape(scenes * time, 1, 1, slots)

        for attn, ln_face, ffn, ln_ffn, temporal in zip(
            self.face_attn, self.ln_face, self.ffn, self.ln_ffn, self.temporal
        ):
            def across_speakers(dense):
                flat = dense.permute(0, 2, 1, 3).reshape(scenes * time, slots, dim)
                flat = self._sublayer(
                    flat, ln_face, lambda z: attn(z, z, z, mask=bias)
                )
                flat = self._sublayer(flat, ln_ffn, ffn)
                return flat.reshape(scenes, time, slots, dim).permute(0, 2, 1, 3)

            def over_time(dense):
                seq = dense.reshape(scenes * slots, time, dim)
                return temporal(seq).reshape(scenes, slots, time, dim)

            if self.order == "speaker_time":
                dense = over_time(across_speakers(dense))
            else:
                dense = across_speakers(over_time(dense))

        return dense


SCENE_MIXERS = {"cast": CASTBlock, "loconet": LoCoNetMixer}


def build_scene_mixer(cfg: dict, dim: int):
    """The mixer selected by `mechanism`, or None for the per-face model."""
    scene = dict(cfg or {})
    scene.pop("max_tracks", None)                 # a dataloader knob
    mechanism = scene.pop("mechanism", None) or "none"
    if mechanism == "none":
        return None
    if mechanism not in SCENE_MIXERS:
        raise ValueError(
            f"unknown asd.scene.mechanism: {mechanism!r}; "
            f"pick one of {sorted(SCENE_MIXERS) + ['none']}"
        )
    return SCENE_MIXERS[mechanism]({**scene, "d_model": dim})
