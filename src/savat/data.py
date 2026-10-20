"""Media decoding, clip datasets and grouped splits.

A clip is a video file that contains an audio track. Frames are uniformly
sampled, resized to a square resolution and returned channel-first. Audio is
resampled to mono, padded or truncated to a fixed duration, and converted to a
log-Mel spectrogram.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from savat.reliability import (
    ReliabilityNormalizer,
    ReliabilityScores,
    reliability_scores,
)

MEDIA_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
VIDEO_MEAN = (0.43216, 0.394666, 0.37645)
VIDEO_STD = (0.22803, 0.22145, 0.216989)
AUDIO_TIME_FRAMES = 320
WINDOW_DURATION = 5.0
_GROUP_SUFFIX = re.compile(r"[_-]?\d+$")


@dataclass(frozen=True)
class ClipSample:
    """One media clip with its label and source group."""

    path: Path
    label: int
    label_name: str
    group: str


def _group_from_stem(stem: str) -> str:
    """Strip a trailing index so that ``match_001`` and ``match_002`` group."""
    return _GROUP_SUFFIX.sub("", stem) or stem


def discover_clips(
    root: Path, label_order: list[str] | None = None
) -> tuple[list[ClipSample], list[str]]:
    """Scan ``root/<label>/<clip>`` and return clips and label names.

    The label is the first directory below ``root``. Files directly under
    ``root`` are rejected, because an unlabelled clip cannot be used for
    supervised training. ``label_order`` lets evaluation reuse the class order
    stored in a checkpoint.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    labels = sorted(
        directory.name
        for directory in root.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    if not labels:
        raise FileNotFoundError(f"no label directory found under {root}")
    if label_order is not None:
        unknown = set(labels) - set(label_order)
        if unknown:
            raise ValueError(f"labels absent from the checkpoint: {sorted(unknown)}")
        labels = list(label_order)
    label_index = {name: index for index, name in enumerate(labels)}

    clips: list[ClipSample] = []
    for label_name in labels:
        label_root = root / label_name
        for path in sorted(label_root.rglob("*")):
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            relative = path.relative_to(label_root)
            if len(relative.parts) > 1:
                group_name = "/".join(relative.parts[:-1])
            else:
                group_name = _group_from_stem(path.stem)
            clips.append(
                ClipSample(
                    path=path,
                    label=label_index[label_name],
                    label_name=label_name,
                    group=f"{label_name}/{group_name}",
                )
            )
    if not clips:
        raise FileNotFoundError(f"no media file found under {root}")
    return clips, labels


def grouped_split(
    clips: list[ClipSample],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 17,
) -> tuple[list[ClipSample], list[ClipSample], list[ClipSample]]:
    """Split by source group, stratified by label.

    Every group belongs to exactly one split, so a clip and its re-encodings
    never cross the train / validation / test boundary. Groups are shuffled
    with a fixed seed, which makes the split reproducible.
    """
    train_fraction, validation_fraction, test_fraction = fractions
    if not math.isclose(
        train_fraction + validation_fraction + test_fraction, 1.0, abs_tol=1e-6
    ):
        raise ValueError("fractions must sum to one")
    if min(fractions) < 0:
        raise ValueError("fractions must be non-negative")

    by_label: dict[int, dict[str, list[ClipSample]]] = {}
    for clip in clips:
        by_label.setdefault(clip.label, {}).setdefault(clip.group, []).append(clip)

    generator = random.Random(seed)
    splits: tuple[list[ClipSample], list[ClipSample], list[ClipSample]] = ([], [], [])
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        generator.shuffle(groups)
        count = len(groups)
        if count < 3:
            validation_count = 1 if count > 1 else 0
            test_count = 0
        else:
            validation_count = max(1, int(round(count * validation_fraction)))
            test_count = max(1, int(round(count * test_fraction)))
            while validation_count + test_count >= count:
                if validation_count >= test_count and validation_count > 1:
                    validation_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break
        train_count = count - validation_count - test_count
        for index, group in enumerate(groups):
            if index < train_count:
                splits[0].extend(by_label[label][group])
            elif index < train_count + validation_count:
                splits[1].extend(by_label[label][group])
            else:
                splits[2].extend(by_label[label][group])

    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        if not split:
            message = f"the {name} split is empty; provide more groups per class"
            raise ValueError(message)
    return tuple(sorted(split, key=lambda clip: str(clip.path)) for split in splits)  # type: ignore[return-value]


def sample_frame_indices(total: int, count: int) -> list[int]:
    """Return ``count`` uniformly spaced frame indices, with repetition."""
    if total <= 0:
        raise ValueError("a clip must contain at least one frame")
    if count <= 0:
        raise ValueError("count must be positive")
    return [min(total - 1, int(index * total / count)) for index in range(count)]


def frames_to_tensor(
    frames: list[np.ndarray],
    num_frames: int,
    image_size: int = 112,
    normalize: bool = True,
) -> torch.Tensor:
    """Stack RGB frames into a ``(3, T, H, W)`` float tensor."""
    if not frames:
        raise ValueError("no frame to convert")
    indices = sample_frame_indices(len(frames), num_frames)
    selected = np.stack([frames[index] for index in indices], axis=0)
    tensor = torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    if normalize:
        mean = torch.tensor(VIDEO_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(VIDEO_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3).contiguous()


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_mels: int = 64,
    time_frames: int = 256,
) -> torch.Tensor:
    """Convert a mono waveform to a ``(1, n_mels, time_frames)`` log-Mel map."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256
    )
    amplitude = torchaudio.transforms.AmplitudeToDB(top_db=80)
    spectrogram = amplitude(mel(waveform.unsqueeze(0)))
    spectrogram = torch.nn.functional.pad(
        spectrogram, (0, max(0, time_frames - spectrogram.shape[-1]))
    )
    spectrogram = spectrogram[..., :time_frames]
    return (spectrogram.clamp(-80, 20) + 80) / 100


def _decode_window(
    path: Path,
    *,
    start: float,
    end: float,
    sample_rate: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Decode only the frames whose timestamps fall inside ``[start, end)``."""
    video_frames: list[np.ndarray] = []
    audio_chunks: list[np.ndarray] = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

    with av.open(str(path)) as container:
        streams = [
            stream for stream in container.streams if stream.type in {"video", "audio"}
        ]
        if not any(stream.type == "video" for stream in streams):
            raise ValueError(f"no video stream found in {path}")
        has_video = any(stream.type == "video" for stream in streams)
        has_audio = any(stream.type == "audio" for stream in streams)
        if start > 0:
            container.seek(int(start * av.time_base), backward=True)

        video_done = not has_video
        audio_done = not has_audio
        for frame in container.decode(*streams):
            timestamp = frame.time
            if timestamp is not None and timestamp < start:
                continue
            if timestamp is not None and timestamp >= end:
                if isinstance(frame, av.VideoFrame):
                    video_done = True
                elif isinstance(frame, av.AudioFrame):
                    audio_done = True
                if video_done and audio_done:
                    break
                continue
            if isinstance(frame, av.VideoFrame):
                video_frames.append(frame.to_ndarray(format="rgb24"))
            elif isinstance(frame, av.AudioFrame):
                for resampled in resampler.resample(frame):
                    audio_chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            audio_chunks.append(resampled.to_ndarray().reshape(-1))
    return video_frames, audio_chunks


def decode_media(
    path: Path,
    *,
    num_frames: int = 16,
    image_size: int = 112,
    sample_rate: int = 16_000,
    duration: float = WINDOW_DURATION,
    window_start: float | None = None,
    time_frames: int = AUDIO_TIME_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a common audio-visual window into video, waveform and log-Mel.

    The same temporal window is used for both streams. By default the window is
    centred in the clip; ``window_start`` overrides this for ablations. A
    single pass demuxes both streams, and the decoder stops once both streams
    have passed the end of the window.
    """
    with av.open(str(path)) as container:
        total_duration = (
            float(container.duration) / av.time_base
            if container.duration is not None
            else duration
        )
    if window_start is None:
        start = max(0.0, (total_duration - duration) / 2.0)
    else:
        start = max(0.0, min(float(window_start), max(0.0, total_duration - duration)))
    end = start + duration

    video_frames, audio_chunks = _decode_window(
        path, start=start, end=end, sample_rate=sample_rate
    )
    if not video_frames:
        raise ValueError(f"no video frame decoded from {path} in [{start}, {end})")

    video = frames_to_tensor(video_frames, num_frames=num_frames, image_size=image_size)
    waveform = (
        np.concatenate(audio_chunks).astype(np.float32)
        if audio_chunks
        else np.zeros(0, dtype=np.float32)
    )
    target = int(sample_rate * duration)
    audio = torch.from_numpy(waveform)
    if audio.numel() == 0:
        audio = torch.zeros(target)
    elif audio.numel() < target:
        audio = torch.nn.functional.pad(audio, (0, target - audio.numel()))
    else:
        audio = audio[:target]
    spectrogram = waveform_to_logmel(
        audio, sample_rate=sample_rate, time_frames=time_frames
    )
    return video, audio, spectrogram


class MultimodalDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]
):
    """One item per clip: video, spectrogram, reliability, label and index.

    The reliability tensor contains ``[video, audio]`` geometric-mean scores
    computed from the decoded signals. It is a training signal, not an
    annotation, and is therefore computed on the fly.
    """

    def __init__(
        self,
        clips: list[ClipSample],
        num_frames: int = 16,
        image_size: int = 112,
        duration: float = WINDOW_DURATION,
        window_start: float | None = None,
        time_frames: int = AUDIO_TIME_FRAMES,
        normalizer: ReliabilityNormalizer | None = None,
    ) -> None:
        if not clips:
            raise ValueError("clip list is empty")
        self.clips = clips
        self.num_frames = num_frames
        self.image_size = image_size
        self.duration = duration
        self.window_start = window_start
        self.time_frames = time_frames
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.clips)

    def raw_reliability(self, index: int) -> ReliabilityScores:
        """Decode one clip and return its unnormalised reliability scores."""
        clip = self.clips[index]
        video, _waveform, spectrogram = decode_media(
            clip.path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            duration=self.duration,
            window_start=self.window_start,
            time_frames=self.time_frames,
        )
        return reliability_scores(video, spectrogram)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        clip = self.clips[index]
        video, waveform, spectrogram = decode_media(
            clip.path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            duration=self.duration,
            window_start=self.window_start,
            time_frames=self.time_frames,
        )
        scores = reliability_scores(video, spectrogram)
        if self.normalizer is not None:
            scores = self.normalizer.transform(scores)
        return video, spectrogram, waveform, scores.as_tensor(), clip.label, index


def fit_reliability_normalizer(
    dataset: MultimodalDataset, limit: int = 0
) -> ReliabilityNormalizer:
    """Fit reliability bounds on the training partition only.

    Args:
        dataset: Training dataset. Validation and test datasets must receive the
            returned normalizer rather than fitting their own.
        limit: Maximum number of clips to inspect; ``0`` uses the whole split.
    """
    count = len(dataset) if limit <= 0 else min(limit, len(dataset))
    samples = []
    for index in range(count):
        scores = dataset.raw_reliability(index)
        samples.append(
            {
                "video": scores.video_indicators,
                "audio": scores.audio_indicators,
            }
        )
    return ReliabilityNormalizer().fit(samples)


__all__ = [
    "AUDIO_TIME_FRAMES",
    "ClipSample",
    "MultimodalDataset",
    "MEDIA_SUFFIXES",
    "VIDEO_MEAN",
    "VIDEO_STD",
    "WINDOW_DURATION",
    "decode_media",
    "discover_clips",
    "fit_reliability_normalizer",
    "frames_to_tensor",
    "grouped_split",
    "sample_frame_indices",
    "waveform_to_logmel",
]
