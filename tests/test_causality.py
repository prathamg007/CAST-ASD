"""Perturbation tests: the causal models never read the future.

Inputs are perturbed in place rather than truncated, so shapes, kernels and
reduction order are unchanged and an output that does not depend on the
perturbed region is bit-identical (`torch.equal`).
"""

import importlib.util

import pytest
import torch

from cast_asd.models.asd import AudioFrontend, AudioVisualASD
from cast_asd.models.lightasd.model import LightASD
from cast_asd.models.modules.attention import CausalMultiHeadAttention, TransformerModel
from cast_asd.models.modules.audio import build_audio_encoder
from cast_asd.models.modules.visual_encoder import CConv1d

requires_moshi = pytest.mark.skipif(
    importlib.util.find_spec("moshi") is None, reason="moshi is not installed"
)
ENCODERS = ["talknet", "vggish", pytest.param("mimi", marks=requires_moshi)]

SAMPLES_PER_FRAME = 640  # 25 fps at 16 kHz


def _cfg(name, head="binary"):
    return {
        "audio": {"name": name},
        "head": head,
        "cross": {"d_model": 128, "m_ff": 2, "n_heads": 4, "n_layers": 1, "dropout": 0.0},
        "temporal": {"d_model": 256, "m_ff": 2, "n_heads": 4, "n_layers": 1, "dropout": 0.0},
    }


def _frozen_model(cfg):
    torch.manual_seed(0)
    model = AudioVisualASD(cfg).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _clip(batch, frames):
    waveform = torch.randn(batch, 1, frames * SAMPLES_PER_FRAME) * 0.1
    visual = torch.randint(0, 256, (batch, frames, 112, 112)).float()
    return waveform, visual


@pytest.mark.parametrize("name", ENCODERS)
def test_audio_encoder_is_causal(name):
    torch.manual_seed(0)
    encoder = build_audio_encoder({"name": name}).eval()
    waveform = torch.randn(2, 1, 24 * SAMPLES_PER_FRAME) * 0.1
    with torch.no_grad():
        reference = encoder(waveform)
    rate = SAMPLES_PER_FRAME * reference.shape[1] / waveform.shape[-1]

    for cut_sample in (4 * SAMPLES_PER_FRAME, 12 * SAMPLES_PER_FRAME):
        perturbed = waveform.clone()
        perturbed[..., cut_sample:] = torch.randn_like(perturbed[..., cut_sample:])
        with torch.no_grad():
            other = encoder(perturbed)
        safe = int(cut_sample * rate / SAMPLES_PER_FRAME)
        assert torch.equal(reference[:, :safe], other[:, :safe]), (
            f"{name}: a feature before sample {cut_sample} sees the future"
        )


#: Upper bound on how long before `t_n` a frame's audio may end. Every encoder's
#: grid shares boundaries with the 25 fps video grid at least every other frame,
#: so a well-calibrated encoder's worst frame sits near 0.
_MAX_STALENESS_MS = 16.0


@pytest.mark.parametrize("name", ENCODERS)
def test_audio_frontend_deadline_is_tight_and_causal(name):
    """Encoder + resampler: frame `n`'s audio ends at `t_n`, not after, not long before.

    Consecutive frames are probed: an encoder slower than the video grid can
    violate the deadline on alternate frames only.
    """
    torch.manual_seed(0)
    frontend = AudioFrontend({"name": name}, d_model=128).eval()
    for parameter in frontend.parameters():
        parameter.requires_grad_(False)

    frames = 32
    total = frames * SAMPLES_PER_FRAME
    waveform = torch.randn(1, 1, total) * 0.1
    audio_lengths = torch.tensor([total])
    frame_lengths = torch.tensor([frames])
    with torch.no_grad():
        reference = frontend(waveform, audio_lengths, frame_lengths)

    margins = []
    for frame in range(10, 20):
        # Binary-search the first sample this frame depends on not at all.
        low, high = 0, total
        while low < high:
            middle = (low + high) // 2
            probe = waveform.clone()
            probe[..., middle:] = torch.randn_like(probe[..., middle:])
            with torch.no_grad():
                other = frontend(probe, audio_lengths, frame_lengths)
            if torch.equal(other[0, frame], reference[0, frame]):
                high = middle
            else:
                low = middle + 1
        margins.append((low - 1 - frame * SAMPLES_PER_FRAME) / 16.0)

    worst = max(margins)
    assert worst <= 0.0, f"{name}: a frame reads {worst:.1f} ms past t_n"
    assert worst >= -_MAX_STALENESS_MS, (
        f"{name}: every frame's audio ends {-worst:.1f} ms before t_n"
    )


@pytest.mark.parametrize("name", ENCODERS)
def test_model_is_causal_end_to_end(name):
    model = _frozen_model(_cfg(name))
    batch, frames = 2, 24
    waveform, visual = _clip(batch, frames)
    with torch.no_grad():
        reference = model(waveform, visual)[0]

    cut = frames // 2
    perturbed_audio = waveform.clone()
    perturbed_audio[..., cut * SAMPLES_PER_FRAME:] = torch.randn_like(
        perturbed_audio[..., cut * SAMPLES_PER_FRAME:]
    )
    perturbed_visual = visual.clone()
    perturbed_visual[:, cut:] = torch.randint(0, 256, perturbed_visual[:, cut:].shape).float()
    with torch.no_grad():
        other = model(perturbed_audio, perturbed_visual)[0]

    assert torch.equal(reference[:, :cut], other[:, :cut])


def test_non_causal_model_does_read_ahead():
    cfg = {**_cfg("talknet"), "causal": False}
    model = _frozen_model(cfg)
    waveform, visual = _clip(1, 16)
    with torch.no_grad():
        reference = model(waveform, visual)[0]
        perturbed = visual.clone()
        perturbed[:, 8:] = torch.randint(0, 256, perturbed[:, 8:].shape).float()
        other = model(waveform, perturbed)[0]
    assert not torch.equal(reference[:, :8], other[:, :8])


def test_batchnorm_is_the_only_train_time_leak():
    """In train() BatchNorm pools over B*T, so past frames shift; eval uses
    running statistics and is causal."""
    torch.manual_seed(0)
    model = AudioVisualASD(_cfg("talknet")).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frames = 16
    waveform, visual = _clip(2, frames)
    with torch.no_grad():
        reference = model(waveform, visual)[0]
        perturbed = visual.clone()
        perturbed[:, frames // 2:] = torch.randint(0, 256, perturbed[:, frames // 2:].shape).float()
        other = model(waveform, perturbed)[0]
    assert not torch.equal(reference[:, :frames // 2], other[:, :frames // 2])


def test_padding_cannot_leak_or_disable_the_causal_mask():
    """A short clip's outputs must not change when batched beside a longer one."""
    model = _frozen_model(_cfg("talknet"))
    short, long = 10, 24
    torch.manual_seed(1)
    short_audio, short_visual = _clip(1, short)

    with torch.no_grad():
        alone = model(
            short_audio, short_visual,
            torch.tensor([short_audio.shape[-1]]), torch.tensor([short]),
        )[0]

    audio = torch.zeros(2, 1, long * SAMPLES_PER_FRAME)
    audio[0, :, : short * SAMPLES_PER_FRAME] = short_audio
    audio[1] = torch.randn(1, long * SAMPLES_PER_FRAME) * 0.1
    visual = torch.randint(0, 256, (2, long, 112, 112)).float()
    visual[0, :short] = short_visual
    visual[0, short:] = short_visual[0, -1]
    mask = torch.zeros(2, long, dtype=torch.bool)
    mask[0, :short] = True
    mask[1] = True

    with torch.no_grad():
        batched = model(
            audio, visual,
            torch.tensor([short * SAMPLES_PER_FRAME, long * SAMPLES_PER_FRAME]),
            torch.tensor([short, long]),
            key_padding_mask=mask,
        )[0]

    assert torch.allclose(alone[0, :short], batched[0, :short], atol=1e-5)


def test_causal_attention_is_causal():
    torch.manual_seed(0)
    attn = CausalMultiHeadAttention(dim=32, num_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 12, 32)
    perturbed = x.clone()
    perturbed[:, 6:] = torch.randn_like(perturbed[:, 6:])
    with torch.no_grad():
        reference = attn(x, x, x)
        other = attn(perturbed, perturbed, perturbed)
    assert torch.equal(reference[:, :6], other[:, :6])
    assert not torch.equal(reference[:, 6:], other[:, 6:])


@pytest.mark.parametrize("cross_attention", [False, True])
def test_transformer_causal_switch(cross_attention):
    """`causal: true` never reads ahead; `causal: false` does."""
    cfg = dict(d_model=32, m_ff=4, n_layers=1, n_heads=4, dropout=0.0)
    torch.manual_seed(0)
    x, src = torch.randn(1, 10, 32), torch.randn(1, 10, 32)
    later_x, later_src = x.clone(), src.clone()
    later_x[:, 6:] = torch.randn(1, 4, 32)
    later_src[:, 6:] = torch.randn(1, 4, 32)
    for causal in (True, False):
        m = TransformerModel({**cfg, "causal": causal}, cross_attention=cross_attention).eval()
        with torch.no_grad():
            before = m(x, src=src)
            after = m(later_x, src=later_src)
        assert torch.equal(before[:, :6], after[:, :6]) is causal


def test_causal_conv_padding_switches_sides():
    x = torch.zeros(1, 1, 9)
    x[0, 0, 4] = 1.0                                  # an impulse at t=4
    for causal in (True, False):
        c = CConv1d(1, 1, 5, bias=False, causal=causal)
        with torch.no_grad():
            c.weight.fill_(1.0)
            y = c(x)
        reads_ahead = bool((y[0, 0, :4].abs() > 1e-6).any())
        assert reads_ahead is not causal


def test_trainable_and_frozen_encoders():
    talknet = build_audio_encoder({"name": "talknet"})
    vggish = build_audio_encoder({"name": "vggish"})
    for encoder in (talknet, vggish):
        assert not encoder.frozen
        assert all(p.requires_grad for p in encoder.parameters())


@requires_moshi
def test_mimi_is_frozen():
    encoder = build_audio_encoder({"name": "mimi"})
    assert encoder.frozen and not encoder.training
    assert not any(p.requires_grad for p in encoder.parameters())


def test_talknet_head_shape():
    """TalkNet-ASD's head: bare Linear(dim, 2) on all three streams."""
    model = _frozen_model(_cfg("talknet", head="talknet"))
    batch, frames = 2, 12
    waveform, visual = _clip(batch, frames)
    with torch.no_grad():
        outputs = model(waveform, visual)
    assert all(out.shape == (batch, frames, 2) for out in outputs)


# --- Light-ASD -------------------------------------------------------------


def _lightasd_drift(causal, perturb_audio):
    torch.manual_seed(0)
    m = LightASD(causal=causal).eval()
    vis = torch.rand(1, 12, 112, 112) * 255
    aud = torch.randn(1, 13, 12)
    v2, a2 = vis.clone(), aud.clone()
    v2[:, 7:] = torch.rand(1, 5, 112, 112) * 255
    if perturb_audio:
        a2[:, :, 7:] = torch.randn(1, 13, 5)
    with torch.no_grad():
        before, after = m(vis, aud), m(v2, a2)
    return float((before[:, :7] - after[:, :7]).abs().max())


def test_lightasd_causal_variant_does_not_read_the_future():
    # Exactly zero: causality is structural, so any difference is a leak.
    assert _lightasd_drift(causal=True, perturb_audio=True) == 0.0


def test_lightasd_published_variant_does_read_the_future():
    assert _lightasd_drift(causal=False, perturb_audio=False) > 0.0


def test_lightasd_matches_the_published_parameter_count():
    n = sum(p.numel() for p in LightASD(causal=False).parameters())
    assert 0.90e6 < n < 1.15e6, f"{n / 1e6:.3f}M is not the published ~1.0M"
