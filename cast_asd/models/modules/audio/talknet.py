"""TalkNet's audio encoder: 13-dim MFCC -> causal SE-ResNet-34."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AudioEncoder
from .talknet_seresnet import CausalSEResNet

#: `python_speech_features` replaces exactly-zero energies with
#: `numpy.finfo(float).eps`; clamping is equivalent for real audio.
_PSF_EPS = 2.220446049250313e-16


class CausalPSFMFCC(nn.Module):
    """`python_speech_features.mfcc` (TalkNet-ASD's front end) on a causal grid.

    Matches psf's defaults after framing: pre-emphasis, rectangular window,
    power spectrum, 26-filter HTK mel bank, log, orthonormal DCT-II, lifter, and
    C0 replaced by log frame energy. Frames are 25 ms with a 10 ms hop,
    left-padded by `win - hop` so frame k ends at sample `(k+1)*hop - 1`
    (`T // hop` frames, none reading past its own end).

    `scale=32768` restores the int16 amplitude TalkNet-ASD reads with scipy;
    only C0 depends on it.
    """

    def __init__(
        self,
        sample_rate=16000,
        numcep=13,
        nfilt=26,
        nfft=512,
        win_ms=25.0,
        hop_ms=10.0,
        preemph=0.97,
        ceplifter=22,
        scale=32768.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.numcep = numcep
        self.nfft = nfft
        self.preemph = preemph
        self.scale = scale
        self.win = int(round(sample_rate * win_ms / 1000))
        self.hop = int(round(sample_rate * hop_ms / 1000))
        self.pad = self.win - self.hop

        self.register_buffer("fbank", self._mel_filterbank(nfilt, nfft, sample_rate))
        self.register_buffer("dct", self._dct_matrix(nfilt, numcep))
        n = torch.arange(numcep, dtype=torch.float64)
        lift = 1 + (ceplifter / 2) * torch.sin(math.pi * n / ceplifter)
        self.register_buffer("lift", lift.float())

    @staticmethod
    def _mel_filterbank(nfilt, nfft, samplerate, lowfreq=0.0, highfreq=None):
        highfreq = highfreq or samplerate / 2

        def hz2mel(hz):
            return 2595 * math.log10(1 + hz / 700.0)

        melpoints = torch.linspace(hz2mel(lowfreq), hz2mel(highfreq), nfilt + 2, dtype=torch.float64)
        hzpoints = 700 * (10 ** (melpoints / 2595.0) - 1)
        bins = torch.floor((nfft + 1) * hzpoints / samplerate).long().tolist()

        fbank = torch.zeros(nfilt, nfft // 2 + 1, dtype=torch.float64)
        for j in range(nfilt):
            left, centre, right = bins[j], bins[j + 1], bins[j + 2]
            for i in range(left, centre):
                fbank[j, i] = (i - left) / max(centre - left, 1)
            for i in range(centre, right):
                fbank[j, i] = (right - i) / max(right - centre, 1)
        return fbank.float()

    @staticmethod
    def _dct_matrix(nfilt, numcep):
        """scipy.fftpack.dct(x, type=2, norm='ortho')[:numcep], as a linear map."""
        n = torch.arange(nfilt, dtype=torch.float64).unsqueeze(0)
        k = torch.arange(numcep, dtype=torch.float64).unsqueeze(1)
        basis = torch.cos(math.pi / nfilt * (n + 0.5) * k)
        scale = torch.full((numcep, 1), math.sqrt(2.0 / nfilt), dtype=torch.float64)
        scale[0] = math.sqrt(1.0 / nfilt)
        return (basis * scale).float()

    def _preemphasize(self, signal):
        """psf pre-emphasises the whole signal, before framing."""
        return torch.cat(
            [signal[:, :1], signal[:, 1:] - self.preemph * signal[:, :-1]], dim=1
        )

    def _cepstra(self, frames):
        """[..., T, win] framed signal -> [..., T, numcep] psf cepstra."""
        spectrum = torch.fft.rfft(frames, n=self.nfft, dim=-1)
        power = spectrum.abs().square() / self.nfft
        energy = power.sum(dim=-1).clamp_min(_PSF_EPS)

        filtered = (power @ self.fbank.t()).clamp_min(_PSF_EPS)
        cepstra = torch.log(filtered) @ self.dct.t()
        cepstra = cepstra * self.lift
        # appendEnergy=True: C0 <- log frame energy.
        return torch.cat([torch.log(energy).unsqueeze(-1), cepstra[..., 1:]], dim=-1)

    def forward(self, waveform):
        # fp32 regardless of autocast: the int16-scale power spectrum overflows fp16.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            signal = waveform.squeeze(1).float() * self.scale
            padded = F.pad(self._preemphasize(signal), (self.pad, 0))
            frames = padded.unfold(-1, self.win, self.hop)  # [B, T, win]
            cepstra = self._cepstra(frames)
            return cepstra.transpose(1, 2)  # [B, numcep, T]


class TalkNetAudioEncoder(AudioEncoder):
    """13-dim MFCC at 100 Hz -> causal SE-ResNet-34 (stride 4) -> 128-d at 25 Hz.

    Each feature reads 13.4 ms past its nominal end (measured). A one-hop input
    delay with "frame_end" alignment gives frame `n` audio ending ~0.1 ms
    before `t_n`.
    """

    samples_per_feature = 640
    resample_align = "frame_end"

    def __init__(self):
        super().__init__()
        self.mfcc = CausalPSFMFCC()
        num_filters = (16, 32, 64, 128)
        self.net = CausalSEResNet([3, 4, 6, 3], list(num_filters))
        self._out_dim = num_filters[-1]
        self.delay_samples = self.mfcc.hop

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def _encode(self, waveform):
        features = self.mfcc(waveform)                  # [B, n_mfcc, T]
        return self.net(features.unsqueeze(1))          # [B, T/4, C]
