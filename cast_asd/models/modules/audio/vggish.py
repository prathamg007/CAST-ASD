"""LoCoNet's audio front end (VGGish trunk + FPN head), reimplemented causally.

Follows `torchvggish/vggish.py` and `forward_audio_frontend` in
SJTUwxz/LoCoNet_ASD @ 68d90c8. Unlike stock VGGish the trunk has three
max-pools and no classifier; it is tapped at indices 9 (stride 4) and 14
(stride 8), the deeper tap is upsampled and concatenated, and 1x1 convs map
512 -> 256 -> 128. Output: 128-d at 25 Hz (stride 4 on the 100 Hz mel grid).

Causal in three places: the mel frame overhang (`delay_samples`), the 3x3
convolutions (time padded on the left), and the FPN upsample (shifted one
stride-4 step so a stride-8 frame is read only once complete). `causal = False`
restores upstream's behaviour in all three.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .base import AudioEncoder

# vggish_params.py @ 68d90c8
SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025      # 400 samples
STFT_HOP_SECONDS = 0.010         # 160 samples
NUM_MEL_BINS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01                # mel_features.log_mel_spectrogram default

WINDOW_SAMPLES = int(STFT_WINDOW_SECONDS * SAMPLE_RATE)   # 400
HOP_SAMPLES = int(STFT_HOP_SECONDS * SAMPLE_RATE)         # 160
N_FFT = 512                                               # 2**ceil(log2(400))

VGGISH_URL = (
    "https://github.com/harritaylor/torchvggish/releases/download/v0.1/"
    "vggish-10086976.pth"
)


def _front_padded_hann(_n_fft: int, **kwargs) -> torch.Tensor:
    """A 400-sample periodic Hann in the first 400 slots of a 512-point frame.

    Upstream zero-pads the end of each 400-sample frame to 512. torchaudio's
    `win_length < n_fft` would centre the taper instead, 56 samples later.
    """
    taper = torch.hann_window(WINDOW_SAMPLES, periodic=True, **kwargs)
    return torch.cat([taper, taper.new_zeros(N_FFT - WINDOW_SAMPLES)])


class VGGishTrunk(nn.Module):
    """`make_layers()` + the FPN head from `VGG.forward`, with upstream's
    parameter names so the AudioSet checkpoint loads by key."""

    #: The two FPN taps, by index into `features`.
    TAP_STRIDE4 = 9
    TAP_STRIDE8 = 14

    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal
        layers = []
        in_channels = 1
        for v in [64, "M", 128, "M", 256, 256, "M", 512, 512]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Padded by hand in `_pad`, so time can be left-only.
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=0),
                           nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1)
        self.conv2 = nn.Conv2d(256, 128, 1, stride=1)

    def _pad(self, x):
        """[B, C, time, freq] -> padded for a 3x3 conv.

        F.pad takes the LAST dim first, so (freq_l, freq_r, time_l, time_r).
        """
        if self.causal:
            return F.pad(x, (1, 1, 2, 0))       # freq symmetric, time left-only
        return F.pad(x, (1, 1, 1, 1))           # upstream's padding=1

    def forward(self, x):
        """[B, 1, time, freq] -> [B, 128, time/4, freq/4]."""
        tap4 = tap8 = None
        for i, layer in enumerate(self.features):
            x = layer(self._pad(x)) if isinstance(layer, nn.Conv2d) else layer(x)
            if i == self.TAP_STRIDE4:
                tap4 = x
            elif i == self.TAP_STRIDE8:
                tap8 = x
        up = self.deconv(tap8)                  # [B, 256, time/4, freq/4]
        if self.causal:
            # Stride-8 frame s spans stride-4 frames 2s and 2s+1; shift one
            # step so 2s reads s-1 and 2s+1 reads s.
            up = F.pad(up, (0, 0, 1, 0))[:, :, : up.shape[2]]
        if up.shape[2] != tap4.shape[2]:
            # An odd stride-4 length makes the upsample one frame long or short.
            up = up[:, :, : tap4.shape[2]] if up.shape[2] > tap4.shape[2] else \
                F.pad(up, (0, 0, 0, tap4.shape[2] - up.shape[2]))

        merged = torch.cat((tap4, up), 1)       # [B, 512, time/4, freq/4]
        return self.conv2(self.conv1(merged))


class VGGishAudioEncoder(AudioEncoder):
    """LoCoNet's front end, trained (AudioSet weights are only the initialisation)."""

    #: Mel at 100 Hz, trunk stride 4 -> 25 Hz, one feature per video frame.
    samples_per_feature = 640
    resample_align = "frame_end"
    #: 240 (mel frame k spans [160k, 160k+400)) + 640 (the 25 Hz feature for
    #: frame n would otherwise cover [t_n, t_n + 40 ms)). See tests/test_causality.py.
    delay_samples = 880

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.trunk = VGGishTrunk()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            # Full-width window with the taper at the front; see _front_padded_hann.
            n_fft=N_FFT,
            win_length=N_FFT,
            hop_length=HOP_SAMPLES,
            f_min=MEL_MIN_HZ,
            f_max=MEL_MAX_HZ,
            n_mels=NUM_MEL_BINS,
            center=False,           # center=True would read half a window ahead
            power=2.0,
            mel_scale="htk",
            window_fn=_front_padded_hann,
        )
        if pretrained:
            self._load_audioset()

    def _load_audioset(self):
        """AudioSet weights cover `features.*` only; deconv/conv1/conv2 start random."""
        from torch import hub

        state = hub.load_state_dict_from_url(VGGISH_URL, progress=False)
        self.trunk.load_state_dict(state, strict=False)

    @property
    def out_dim(self) -> int:
        return 128

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, 1, S] @16 kHz -> [B, S/640, 128]."""
        # `AudioVisualASD` may set `self.causal` after construction.
        self.trunk.causal = self.causal
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        # Mel in fp32: a half-precision power spectrogram underflows on quiet frames.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel = self.mel(waveform.float())                 # [B, mels, time]
        mel = torch.log(mel + LOG_OFFSET)
        mel = mel.transpose(1, 2).unsqueeze(1)               # [B, 1, time, mels]

        # Pad the mel length to a multiple of 8 (three halvings, so both FPN taps
        # share a grid) and to at least 4x the feature count (samples // 640).
        target = max(1, waveform.shape[-1] // self.samples_per_feature)
        want = max(mel.shape[2], 4 * target)
        want += (-want) % 8
        if want > mel.shape[2]:
            mel = F.pad(mel, (0, 0, 0, want - mel.shape[2]))

        # fp32 trunk: it has no normalisation, and its activations overflow fp16.
        with torch.autocast(device_type=mel.device.type, enabled=False):
            features = self.trunk(mel.float())               # [B, 128, t/4, mels/4]
        features = features.mean(dim=3)                      # pool frequency -> [B, 128, t/4]
        features = features[:, :, :target]
        return features.transpose(1, 2).contiguous()         # [B, target, 128]
