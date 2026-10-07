"""Audio encoders, selected by `asd.audio.name`: mimi, talknet or vggish.

Mimi is pretrained and frozen. TalkNet's encoder is trained from scratch.
VGGish starts from AudioSet weights and is fine-tuned, as in LoCoNet.
"""

from .base import AudioEncoder
from .talknet import TalkNetAudioEncoder
from .vggish import VGGishAudioEncoder

AUDIO_ENCODERS = {
    "talknet": TalkNetAudioEncoder,
    "vggish": VGGishAudioEncoder,
}


def build_audio_encoder(cfg: dict) -> AudioEncoder:
    name = cfg["name"]
    if name == "mimi":
        # Imported lazily so the other encoders work without moshi installed.
        from .mimi import MimiAudioEncoder

        return MimiAudioEncoder().freeze()
    if name not in AUDIO_ENCODERS:
        raise ValueError(
            f"unknown audio encoder {name!r}; choose from "
            f"{sorted([*AUDIO_ENCODERS, 'mimi'])}"
        )
    return AUDIO_ENCODERS[name]()


__all__ = ["AudioEncoder", "AUDIO_ENCODERS", "build_audio_encoder"]
