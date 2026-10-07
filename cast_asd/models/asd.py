import torch
import torch.nn as nn

from .modules.attention import TransformerModel
from .modules.audio import build_audio_encoder
from .modules.scene_mixers import build_scene_mixer
from .modules.projection import LinearHead
from .modules.resample import CausalFrameResampler
from .modules.visual_encoder import VisualEncoder


class AudioFrontend(nn.Module):
    """Audio encoder at its own rate -> causal resample to video frames -> projection.

    The projection (Linear + LayerNorm) is pointwise in time, so applying it
    after resampling does not affect causality.
    """

    def __init__(self, cfg, d_model):
        super().__init__()
        self.encoder = build_audio_encoder(cfg)
        # The alignment policy is a property of the encoder's rate.
        self.resampler = CausalFrameResampler(align=self.encoder.resample_align)
        self.proj = nn.Linear(self.encoder.out_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, waveform, audio_lengths, frame_lengths, target_len=None):
        features = self.encoder(waveform)
        lengths = self.encoder.feature_lengths(audio_lengths).clamp(
            max=features.shape[1]
        )
        aligned = self.resampler(
            features, lengths, frame_lengths, target_len=target_len
        )
        return self.norm(self.proj(aligned))


class AudioVisualASD(nn.Module):
    """TalkNet-shaped per-face backbone with an optional scene mixer.

    Audio and visual encoders -> cross-attention both ways -> temporal
    self-attention over the concatenated stream -> (scene mixer) -> heads.
    `causal` (default True) is read by every module that could look ahead;
    `causal: false` gives the same architecture and weight shapes with
    bidirectional reach.
    """

    def __init__(self, cfg):
        super().__init__()
        cross_dim = cfg["cross"]["d_model"]
        temporal_dim = cfg["temporal"]["d_model"]
        if temporal_dim != cross_dim * 2:
            raise ValueError("ASD temporal d_model must be twice cross d_model")

        self.causal = bool(cfg.get("causal", True))
        self.audio_frontend = AudioFrontend(cfg["audio"], cross_dim)
        self.visual_encoder = VisualEncoder(out_dim=cross_dim, causal=self.causal)
        self.norm_visual = nn.LayerNorm(cross_dim)
        if not self.causal:
            # Also skips the encoder's lookahead payback (`delay_samples`).
            self.audio_frontend.encoder.causal = False
        cross_cfg = {**cfg["cross"], "causal": self.causal}
        temporal_cfg = {**cfg["temporal"], "causal": self.causal}
        self.cross_v2a = TransformerModel(cross_cfg, cross_attention=True)
        self.cross_a2v = TransformerModel(cross_cfg, cross_attention=True)
        self.transformer = TransformerModel(temporal_cfg)

        # "binary": one logit per head, BCE. "talknet": TalkNet-ASD's bare
        # Linear(dim, 2), cross-entropy, scored by softmax[..., 1].
        self.head_type = cfg.get("head", "binary")
        if self.head_type == "talknet":
            self.av_head = nn.Linear(temporal_dim, 2)
            self.audio_head = nn.Linear(cross_dim, 2)
            self.visual_head = nn.Linear(cross_dim, 2)
        elif self.head_type == "binary":
            self.av_head = LinearHead(temporal_dim, 1)
            self.audio_head = LinearHead(cross_dim, 1)
            self.visual_head = LinearHead(cross_dim, 1)
        else:
            raise ValueError(f"unknown head type: {self.head_type!r}")

        # Built last, so the modules above get the same initial weights with or
        # without a mixer. The mixer takes the temporal settings and the
        # model-wide `causal` unless `scene` overrides them.
        scene_cfg = {**cfg["temporal"], **(cfg.get("scene", {}) or {})}
        scene_cfg.setdefault("causal", self.causal)
        # Named `cross_track` for checkpoint compatibility; None when per-face.
        self.cross_track = build_scene_mixer(scene_cfg, temporal_dim)

    def forward(
        self,
        waveform,
        visual,
        audio_lengths=None,
        frame_lengths=None,
        key_padding_mask=None,
        return_embeddings=False,
        source_fps=None,
        scene=None,
        labels=None,
    ):
        """`scene`, when given, groups the rows of `visual` into scenes.

        It holds `index`/`slot`/`mask` (scene and slot of each track row) and
        per-scene `audio_lengths` and `frame_lengths`. `waveform` then has one
        row per scene; its features are computed once and expanded via `index`.
        `source_fps` and `labels` are accepted but unused.
        """
        if audio_lengths is None:
            audio_lengths = torch.full(
                (waveform.shape[0],), waveform.shape[-1], device=waveform.device
            )
        if frame_lengths is None:
            frame_lengths = torch.full(
                (visual.shape[0],), visual.shape[1], device=visual.device
            )
        # Align the audio to the visual stream's actual (possibly padded) width.
        target_len = visual.shape[1]
        if scene is None:
            audio_embedding = self.audio_frontend(
                waveform, audio_lengths, frame_lengths, target_len
            )
        else:
            audio_embedding = self.audio_frontend(
                waveform, scene["audio_lengths"], scene["frame_lengths"], target_len
            )[scene["index"]]
        visual_embedding = self.norm_visual(self.visual_encoder(visual))
        if audio_embedding.shape[1] != visual_embedding.shape[1]:
            raise ValueError(
                f"audio and visual lengths differ: {audio_embedding.shape[1]} and "
                f"{visual_embedding.shape[1]}"
            )

        outputs = self._decode(
            audio_embedding, visual_embedding, key_padding_mask, scene,
            return_embeddings=return_embeddings,
        )
        # The training path expects a trailing dict of auxiliary outputs.
        return (*outputs, {}) if return_embeddings else outputs

    def _decode(
        self,
        audio_embedding,
        visual_embedding,
        key_padding_mask,
        scene=None,
        return_embeddings=False,
    ):
        audio_cross = self.cross_v2a(
            audio_embedding, src=visual_embedding, key_padding_mask=key_padding_mask
        )
        visual_cross = self.cross_a2v(
            visual_embedding, src=audio_embedding, key_padding_mask=key_padding_mask
        )
        fused = self.transformer(
            torch.cat([audio_cross, visual_cross], dim=-1),
            key_padding_mask=key_padding_mask,
        )
        if self.cross_track is not None:
            if scene is None:
                raise ValueError(
                    "a scene mechanism is selected but the batch carries no "
                    "scene grouping"
                )
            # Only the fused stream is mixed; the auxiliary unimodal heads stay
            # per-face.
            fused = self.cross_track(
                fused, scene["index"], scene["slot"], scene["mask"],
            )

        av_logits = self.av_head(fused)
        audio_logits = self.audio_head(audio_cross)
        visual_logits = self.visual_head(visual_cross)
        if self.head_type == "binary":
            av_logits = av_logits.squeeze(-1)
            audio_logits = audio_logits.squeeze(-1)
            visual_logits = visual_logits.squeeze(-1)

        if return_embeddings:
            return av_logits, audio_logits, visual_logits, audio_cross, visual_cross
        return av_logits, audio_logits, visual_logits
