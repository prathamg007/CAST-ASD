"""Official TalkNet and LoCoNet (AVA weights) run frame by frame with no future input.

Both models are offline: every layer may read the whole clip. Here the output for
frame t comes from running the unmodified network on frames 1..t only and keeping
its last output, so nothing after t reaches it. The weights are the authors'
released AVA-ActiveSpeaker checkpoints; nothing is retrained.

    TalkNet-ASD   https://github.com/TaoRuijie/TalkNet-ASD     pretrain_AVA.model
    LoCoNet_ASD   https://github.com/SJTUwxz/LoCoNet_ASD       loconet_ava_best.model

The per-frame part of the visual front end is computed once: with the future
missing, only the 3D-convolution outputs of the last two frames of a prefix
differ from the full-clip ones, so those two are recomputed for every t.
"""

import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import python_speech_features
import torch
import torch.nn as nn

FPS, RATE, HOP, WINDOW = 25, 16_000, 160, 400      # 10 ms hop, 25 ms window: 4 audio frames per video frame


def _import(repo, module):
    """Import `module` from one of the two repos; both ship a top-level `model` package."""
    for name in [m for m in sys.modules if m == "model" or m.startswith("model.") or m.startswith("torchvggish")]:
        del sys.modules[name]
    sys.path.insert(0, str(repo))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(repo))


def _load(net, head, path, prefix):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict({k[len(prefix) + 6:]: v for k, v in state.items() if k.startswith(prefix + "model.")}, strict=True)
    head.load_state_dict({k.rsplit(".", 1)[1]: v for k, v in state.items() if k.startswith(prefix + "lossAV.FC.")}, strict=True)
    return net.eval().cuda(), head.eval().cuda()


def load_talknet(repo, weights):
    net = _import(repo, "model.talkNetModel").talkNetModel()
    return _load(net, nn.Linear(256, 2), weights, "")


def load_loconet(repo, weights):
    cfg = SimpleNamespace(MODEL=SimpleNamespace(NUM_SPEAKERS=3, AV_layers=3, ADJUST_ATTENTION=0))
    net = _import(repo, "model.loconet_encoder").locoencoder(cfg)
    net.mel = _import(repo, "torchvggish.mel_features")
    return _load(net, nn.Linear(256, 2), weights, "model.module.")


# ---------------------------------------------------------------- features

def faces_tensor(scene):
    """[faces, frames, 112, 112]; a frame with no box is black, LoCoNet's convention for a missing face."""
    faces = torch.stack(scene["visual"]).clone()
    for k, boxes in enumerate(scene["boxes"]):
        faces[k, [b is None for b in boxes]] = 0
    return faces


@torch.no_grad()
def visual_cache(net, faces):
    """Full-clip ResNet features [F, T, 512] and, per t, those of frames t-1 and t with the future absent."""
    front = net.visualFrontend
    full, tails = [], []
    for track in faces:
        x = (track.float().cuda() / 255 - 0.4161) / 0.1688
        T, H, W = x.shape
        full.append(torch.cat([_frontend_chunk(front, x, i, min(i + 250, T)) for i in range(0, T, 250)]))
        padded = torch.cat([x.new_zeros(3, H, W), x])                   # zeros == the convolution's own padding
        out = []
        for i in range(0, T, 48):
            idx = torch.arange(i, min(i + 48, T), device=x.device)[:, None] + torch.arange(4, device=x.device)
            y = front.frontend3D(padded[idx][:, None])                  # [n, 64, 4, 28, 28], right edge zero-padded
            y = y[:, :, 2:].transpose(1, 2).reshape(-1, *y.shape[1:2], *y.shape[3:])
            out.append(front.resnet(y).view(-1, 2, 512))
        tails.append(torch.cat(out))
    return torch.stack(full), torch.stack(tails)


def _frontend_chunk(front, x, a, b):
    """Full-clip features of frames [a, b): run with two frames of real context on each side."""
    lo, hi = max(0, a - 2), min(len(x), b + 2)
    y = front(x[lo:hi].view(-1, 1, 1, *x.shape[1:])).view(-1, 512)
    return y[a - lo:a - lo + (b - a)]


def visual_prefix(net, full, tails, t):
    seq = tails[:, t, 1:] if t == 0 else torch.cat([full[:, :t - 1], tails[:, t]], 1)
    return net.visualConv1D(net.visualTCN(seq.transpose(1, 2))).transpose(1, 2)


def audio_cache(pcm, frames, feature):
    """Audio features of the clip [4T, d] and, per t, the last two feature frames of the prefix ending at t.

    A 25 ms window starting inside frame t runs up to 15 ms past its end; the
    tail versions see zeros there instead.
    """
    full = feature(pcm)
    need = 4 * frames
    if len(full) < need:
        full = np.pad(full, ((0, need - len(full)), (0, 0)), "wrap")    # as both official loaders do
    tails = []
    for t in range(frames):
        end = (t + 1) * 4 * HOP
        chunk = np.concatenate([pcm[max(0, end - 8 * HOP):end], np.zeros(WINDOW - HOP, pcm.dtype)])
        tails.append(feature(chunk)[-2:])
    return torch.from_numpy(full[:need]).float().cuda(), torch.from_numpy(np.stack(tails)).float().cuda()


def audio_prefix(full, tails, t):
    return torch.cat([full[:4 * t + 2], tails[t]])[None]


def mfcc(pcm):
    return python_speech_features.mfcc(pcm, RATE, numcep=13, winlen=0.025, winstep=0.010)


def log_mel(mel):
    def feature(pcm):
        return mel.log_mel_spectrogram(pcm, audio_sample_rate=RATE, log_offset=0.01, window_length_secs=0.025,
                                       hop_length_secs=0.010, num_mel_bins=64, lower_edge_hertz=125,
                                       upper_edge_hertz=7500)
    return feature


# ---------------------------------------------------------------- models

def speaker_groups(faces, seed=0):
    """LoCoNet's context: each face with up to two others, padded by repetition (as its val loader)."""
    rng = random.Random(seed)
    groups = []
    for i in range(faces):
        others = [j for j in range(faces) if j != i]
        rng.shuffle(others)
        if not others:
            group = [i, i, i]
        elif len(others) < 2:
            group = [i, others[0], others[0]]
        else:
            group = [i] + others[:2]
        groups.append(group)
    return torch.tensor(groups).cuda()


@torch.no_grad()
def stream_talknet(model, scene, offline=False):
    net, head = model
    full_v, tail_v = visual_cache(net, faces_tensor(scene))
    T = full_v.shape[1]
    full_a, tail_a = audio_cache(scene["pcm"], T, mfcc)
    out = full_v.new_zeros(full_v.shape[0], T)
    for t in ([T - 1] if offline else range(T)):
        v = net.visualConv1D(net.visualTCN(full_v.transpose(1, 2))).transpose(1, 2) if offline \
            else visual_prefix(net, full_v, tail_v, t)
        a = net.forward_audio_frontend(full_a[None] if offline else audio_prefix(full_a, tail_a, t))
        a, v = net.forward_cross_attention(a.expand(len(v), -1, -1), v)
        scores = head(net.forward_audio_visual_backend(a, v)).softmax(-1)[:, 1].view(len(v), -1)
        if offline:
            return scores.cpu().numpy()
        out[:, t] = scores[:, -1]
    return out.cpu().numpy()


@torch.no_grad()
def stream_loconet(model, scene, offline=False):
    net, head = model
    full_v, tail_v = visual_cache(net, faces_tensor(scene))
    F, T = full_v.shape[:2]
    full_a, tail_a = audio_cache(scene["pcm"], T, log_mel(net.mel))
    groups = speaker_groups(F).flatten()
    out = full_v.new_zeros(F, T)
    for t in ([T - 1] if offline else range(T)):
        v = net.visualConv1D(net.visualTCN(full_v.transpose(1, 2))).transpose(1, 2) if offline \
            else visual_prefix(net, full_v, tail_v, t)
        a = net.forward_audio_frontend((full_a[None] if offline else audio_prefix(full_a, tail_a, t))[:, None])
        a, v = net.forward_cross_attention(a.expand(F, -1, -1), v)
        fused = net.forward_audio_visual_backend(a[groups], v[groups], b=F, s=3)
        scores = head(fused).softmax(-1)[:, 1].view(F, 3, -1)[:, 0]
        if offline:
            return scores.cpu().numpy()
        out[:, t] = scores[:, -1]
    return out.cpu().numpy()


def load(root):
    """`root` holds `TalkNet-ASD/`, `LoCoNet_ASD/` and `weights/`. Returns [(key, label, causal, scorer)]."""
    root = Path(root)
    talknet = load_talknet(root / "TalkNet-ASD", root / "weights" / "talknet_pretrain_AVA.model")
    loconet = load_loconet(root / "LoCoNet_ASD", root / "weights" / "loconet_ava_best.model")
    return [
        ("loconet", "LoCoNet (causal)", True, lambda scene: stream_loconet(loconet, scene)),
        ("talknet", "TalkNet (causal)", True, lambda scene: stream_talknet(talknet, scene)),
        ("loconet_offline", "LoCoNet (offline)", False,
         lambda scene: stream_loconet(loconet, scene, offline=True)),
        ("talknet_offline", "TalkNet (offline)", False,
         lambda scene: stream_talknet(talknet, scene, offline=True)),
    ]
