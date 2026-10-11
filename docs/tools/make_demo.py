"""Build the project-page demo clips from MSDWild with the released checkpoints.

    python docs/tools/make_demo.py --msdwild data/msdwild \
        00897:0 00134:0 01036:0 02808:1 00934:0

`--msdwild` holds `videos/<video>.{mp4,csv}` and `rttms/all.rttm`, as in the
README. Each `<video>:<window>` is one 30 s scene, cut exactly as
`cast_asd.preprocess.msdwild` cuts it. The scene runs through the causal and
offline MSDWild checkpoints and is scored with `SceneDiagnostics`. Further
systems can be scored on the same scenes for comparison:

    --model "talknet=TalkNet (MFCC),configs/msdwild/talknet.yaml,runs/talknet/last.ckpt"

Output goes to `docs/demo/<video>_<window>/`:

    clip.mp4      the decoded frames and audio of the scene, H.264/AAC
    data.js       boxes, reference labels, probabilities, metrics
    face_<k>.jpg  one thumbnail per face track
    poster.jpg    first frame

Needs a CUDA GPU and FFmpeg on PATH.
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cast_asd.config import load_config  # noqa: E402
from cast_asd.data.dataloaders.scene_packed import SceneASDCollator  # noqa: E402
from cast_asd.data.media import SAMPLE_RATE, load_waveform, normalize_waveform  # noqa: E402
from cast_asd.preprocess.msdwild import (  # noqa: E402
    crop_face, read_annotations, read_frames, read_rttm, retained_speakers, video_metadata,
)
from cast_asd.preprocess.writer import _pcm16  # noqa: E402
from cast_asd.tasks.scene_asd_task import SceneASDTask  # noqa: E402
from cast_asd.tasks.scene_metrics import SceneDiagnostics  # noqa: E402

CLIP_SECONDS, MIN_FACE_RATIO = 30.0, 0.9
METRICS = ["ap", "ap_competing", "f1", "precision", "recall", "fa_competing@90r",
           "oracle_der", "frames", "competing_rate"]


def load_task(config, checkpoint):
    cfg = load_config(REPO / config, "asd")
    cfg["evaluation"] = {}
    task = SceneASDTask(cfg)
    state = torch.load(REPO / checkpoint, map_location="cpu", weights_only=False)
    task.on_load_checkpoint(state)
    task.load_state_dict(state["state_dict"], strict=True)
    return task.eval().cuda()


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def build_scene(root, activity, video_id, window):
    """The scene as the packer writes it and the pack reader returns it."""
    video = root / "videos" / f"{video_id}.mp4"
    annotations = read_annotations(video.with_suffix(".csv"))
    fps, total = video_metadata(video)
    clip = round(CLIP_SECONDS * fps)
    first, last = min(annotations), min(max(annotations), total - 1)
    start = list(range(first, last + 1, clip))[window]
    speakers = retained_speakers(annotations, start, min(clip, total - start), MIN_FACE_RATIO)
    frames, _ = read_frames(video, start, min(clip, total - start))
    n, start_time = len(frames), start / fps

    with tempfile.TemporaryDirectory() as tmp:
        # torchaudio may lack an MP4 decoder; FFmpeg decodes at the native rate
        # and `load_waveform` downmixes and resamples as the packer does.
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-c:a", "pcm_f32le", str(wav)],
                       check=True)
        waveform = load_waveform(wav)
        native, native_rate = sf.read(str(wav), dtype="float32", always_2d=True)

    samples = round(n / fps * SAMPLE_RATE)
    first_sample = round(start_time * SAMPLE_RATE)
    audio = waveform[:, first_sample:first_sample + samples]
    audio = torch.nn.functional.pad(audio, (0, samples - audio.shape[-1]))
    pcm = _pcm16(audio)                                  # source gain, for the official baselines
    audio = torch.from_numpy(_pcm16(audio).astype(np.float32) / 32768.0).unsqueeze(0)
    audio, _ = normalize_waveform(audio)

    centres = start_time + (np.arange(n) + 0.5) / fps

    def speaking(speaker):
        mask = np.zeros(n, bool)
        for a, b in activity.get(speaker, []):
            mask |= (centres >= a) & (centres < b)
        return mask

    boxes = [[annotations.get(start + i, {}).get(s) for i in range(n)] for s in speakers]
    offscreen = np.zeros(n, bool)
    for speaker in activity:
        if speaker not in speakers:
            offscreen |= speaking(speaker)
    native_first = round(start_time * native_rate)
    return {
        "video": video_id, "window": window, "fps": fps, "start_time": start_time, "frames": frames,
        "speakers": speakers, "boxes": boxes, "audio": audio, "pcm": pcm,
        "visual": [torch.from_numpy(np.stack([crop_face(f, b) for f, b in zip(frames, bx)])) for bx in boxes],
        "labels": [torch.from_numpy(speaking(s).astype(np.float32)) for s in speakers],
        "offscreen": offscreen,
        "native_audio": native[native_first:native_first + round(n / fps * native_rate)],
        "native_rate": native_rate,
    }


def scene_batch(scene):
    metadata = [{"source_fps": scene["fps"], "start_sec": scene["start_time"],
                 "sample_id": f"{scene['video']}_{scene['window']:05d}_{s}"} for s in scene["speakers"]]
    batch = SceneASDCollator()([(scene["audio"], scene["visual"], scene["labels"], metadata)])
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def predict(task, scene):
    """[faces, frames] speaking probabilities of a checkpoint trained with this repository."""
    batch = scene_batch(scene)
    with torch.autocast("cuda", dtype=torch.float16):
        logits, _, _ = task._shared_step(batch)
    return task.probabilities(logits.float())[:, :len(scene["frames"])].cpu().numpy()


def evaluate(scene, scores, reference):
    batch = scene_batch(scene)
    mask = batch["frame_mask"]
    probabilities = torch.zeros_like(batch["labels"])
    probabilities[:, :scores.shape[1]] = torch.from_numpy(scores).to(probabilities)
    labels, valid = batch["labels"] > 0.5, mask
    others = (labels & valid).sum(0, keepdim=True) - labels.long() > 0     # another tracked face speaks
    alarm = (probabilities >= 0.5) & ~labels & valid
    diagnostics = SceneDiagnostics(0.5, recall_target=0.9, rttm={scene["video"]: reference})
    diagnostics.update(batch["sample_ids"], probabilities, batch["labels"], mask, batch["scene_index"], 1,
                       start_sec=batch["start_sec"], source_fps=batch["source_fps"])
    metrics = diagnostics.compute()
    kept = {k: metrics.get(k) for k in METRICS}
    # False-alarm rates at the fixed 0.5 threshold, overall and on competing negatives.
    kept["fa@0.5"] = (alarm.sum() / (~labels & valid).sum().clamp(min=1)).item()
    kept["fa_competing@0.5"] = ((alarm & others).sum() / (~labels & valid & others).sum().clamp(min=1)).item()
    return kept


def write_video(scene, path):
    """Encode the decoded frames themselves, so boxes stay frame-aligned in the browser."""
    h, w = scene["frames"][0].shape[:2]
    scale = min(1.0, 720 / h)
    size = f"{round(w * scale / 2) * 2}:{round(h * scale / 2) * 2}"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        sf.write(str(wav), scene["native_audio"], scene["native_rate"], subtype="PCM_16")
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(scene["fps"]), "-i", "-", "-i", str(wav), "-vf", f"scale={size}:flags=lanczos",
             "-c:v", "libx264", "-preset", "slow", "-crf", "25", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        for frame in scene["frames"]:
            ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.close()
        if ffmpeg.wait():
            raise RuntimeError(f"ffmpeg failed on {path}")
    return w, h


def thumbnail(scene, k):
    boxes = scene["boxes"][k]
    sizes = [max(b[2] - b[0], b[3] - b[1]) if b else 0 for b in boxes]
    i = int(np.argsort(sizes)[3 * len(sizes) // 4])                  # a large, typical box
    x1, y1, x2, y2 = boxes[i]
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.62
    frame = scene["frames"][i]
    H, W = frame.shape[:2]
    crop = frame[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
    return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--msdwild", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "docs" / "demo")
    parser.add_argument("--model", action="append", default=[], metavar="KEY=LABEL,CONFIG,CHECKPOINT",
                        help="an extra causal system to score on the same scenes (repeatable)")
    parser.add_argument("--official", type=Path, metavar="DIR",
                        help="also run the official TalkNet and LoCoNet AVA weights, causally and offline "
                             "(see official_stream.py)")
    parser.add_argument("--keep-video", action="store_true", help="do not re-encode an existing clip.mp4")
    parser.add_argument("scenes", nargs="+", help="<video>:<window>")
    args = parser.parse_args()

    activity = read_rttm(args.msdwild / "rttms" / "all.rttm")
    causal = load_task("configs/msdwild/cast.yaml", "weights/msdwild_cast.ckpt")
    offline = load_task("configs/msdwild/cast_offline.yaml", "weights/msdwild_cast_offline.ckpt")
    def trained(task):
        return lambda scene: predict(task, scene)

    systems = [("causal", "CAST-ASD (causal)", True, trained(causal)),
               ("offline", "CAST-ASD (offline)", False, trained(offline))]
    for spec in args.model:
        key, rest = spec.split("=", 1)
        label, config, checkpoint = rest.rsplit(",", 2)
        systems.append((key, label, True, trained(load_task(config, checkpoint))))
    if args.official:
        import official_stream
        systems += official_stream.load(args.official)

    for item in args.scenes:
        video_id, window = item.split(":")
        scene = build_scene(args.msdwild, activity[video_id], video_id, int(window))
        probabilities, metrics = {}, {}
        for key, _, _, scorer in systems:
            torch.manual_seed(0)                       # LoCoNet shuffles its roster in eval
            probabilities[key] = scorer(scene)
            metrics[key] = evaluate(scene, probabilities[key], activity[video_id])

        name = f"{video_id}_{window}"
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        h, w = scene["frames"][0].shape[:2]
        if not (args.keep_video and (out / "clip.mp4").exists()):
            write_video(scene, out / "clip.mp4")
        n = len(scene["frames"])
        tracks = []
        for k, speaker in enumerate(scene["speakers"]):
            cv2.imwrite(str(out / f"face_{k}.jpg"), thumbnail(scene, k), [cv2.IMWRITE_JPEG_QUALITY, 88])
            tracks.append({
                "id": speaker,
                "boxes": [[round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]
                          if b else None for b in scene["boxes"][k]],
                "gt": runs(scene["labels"][k].numpy() > 0.5),
                **{key: b64(np.round(p[k][:n] * 255).astype(np.uint8)) for key, p in probabilities.items()},
            })
        data = {
            "video": video_id, "segment": int(window), "fps": scene["fps"], "frames": n, "width": w, "height": h,
            "start_time": scene["start_time"], "tracks": tracks, "offscreen": runs(scene["offscreen"]),
            "models": [{"key": key, "label": label, "causal": is_causal} for key, label, is_causal, _ in systems],
            "metrics": metrics,
        }
        # A script rather than JSON, so the page also works when opened from disk.
        (out / "data.js").write_text(
            f"(window.CAST_DEMO=window.CAST_DEMO||{{}})[{json.dumps(name)}]="
            + json.dumps(data, separators=(",", ":")) + ";\n")
        poster = cv2.resize(scene["frames"][0], (w * 360 // h // 2 * 2, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "poster.jpg"), poster, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"{name}: {len(tracks)} faces, " + ", ".join(f"{key} mAP {m['ap']:.2f}" for key, m in metrics.items()))


if __name__ == "__main__":
    main()
