"""Tests for clip discovery, grouped splits and tensor transforms."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from savat.data import (
    ClipSample,
    decode_media,
    discover_clips,
    frames_to_tensor,
    grouped_split,
    sample_frame_indices,
    waveform_to_logmel,
)


def make_tree(root: Path, labels: tuple[str, ...] = ("football", "tennis")) -> Path:
    for label in labels:
        for index in range(4):
            directory = root / label
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"source{index}_001.mp4").write_bytes(b"")
    return root


def test_discover_clips(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "data")
    clips, labels = discover_clips(root)
    assert labels == ["football", "tennis"]
    assert len(clips) == 8
    assert {clip.group for clip in clips} == {
        "football/source0",
        "football/source1",
        "football/source2",
        "football/source3",
        "tennis/source0",
        "tennis/source1",
        "tennis/source2",
        "tennis/source3",
    }


def test_discover_clips_uses_parent_directory_as_group(tmp_path: Path) -> None:
    root = tmp_path / "data"
    clip = root / "football" / "match_a" / "fragment.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"")
    clips, _labels = discover_clips(root)
    assert clips[0].group == "football/match_a"


def test_discover_clips_rejects_unlabelled_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "orphan.mp4").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        discover_clips(root)


def test_grouped_split_is_disjoint(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "data")
    clips, _labels = discover_clips(root)
    train, validation, test = grouped_split(clips, seed=17)
    groups = [{clip.group for clip in split} for split in (train, validation, test)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_grouped_split_is_deterministic(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "data")
    clips, _labels = discover_clips(root)
    first = grouped_split(clips, seed=29)
    second = grouped_split(clips, seed=29)
    assert [[clip.path for clip in split] for split in first] == [
        [clip.path for clip in split] for split in second
    ]


def test_grouped_split_rejects_bad_fractions(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "data")
    clips, _labels = discover_clips(root)
    with pytest.raises(ValueError):
        grouped_split(clips, fractions=(0.5, 0.2, 0.1))


def test_sample_frame_indices() -> None:
    indices = sample_frame_indices(100, 8)
    assert indices[0] == 0
    assert indices[-1] < 100
    assert len(indices) == 8
    with pytest.raises(ValueError):
        sample_frame_indices(0, 4)


def test_frames_to_tensor_shape() -> None:
    frames = [np.zeros((40, 30, 3), dtype=np.uint8) for _ in range(5)]
    tensor = frames_to_tensor(frames, num_frames=4, image_size=32)
    assert tensor.shape == (3, 4, 32, 32)


def test_waveform_to_logmel_shape() -> None:
    waveform = torch.zeros(16_000 * 5)
    spectrogram = waveform_to_logmel(waveform, time_frames=64)
    assert spectrogram.shape == (1, 64, 64)
    assert float(spectrogram.min()) >= 0.0
    assert float(spectrogram.max()) <= 1.0


def test_decode_media_without_audio(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=8)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for index in range(6):
        array = np.full((32, 32, 3), index * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    from savat.data import decode_media

    video, audio, spectrogram = decode_media(path, num_frames=4, image_size=32)
    assert video.shape == (3, 4, 32, 32)
    assert audio.shape == (16_000 * 5,)
    assert spectrogram.shape == (1, 64, 320)


def _write_av_video(path: Path, *, seconds: int = 3, rate: int = 6) -> None:
    sample_rate = 16_000
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=rate)
        video_stream.width = 32
        video_stream.height = 32
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "mono"
        packets = []
        for index in range(seconds * rate):
            value = np.uint8((index * 255) // (seconds * rate))
            array = np.full((32, 32, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, rate)
            packets.extend(video_stream.encode(frame))
        packets.extend(video_stream.encode())

        tones = (440.0, 880.0, 1760.0)
        for offset in range(0, sample_rate * seconds, 1024):
            count = min(1024, sample_rate * seconds - offset)
            time = np.arange(offset, offset + count) / sample_rate
            frequency = tones[min(len(tones) - 1, int(time[0]))]
            samples = (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(
                samples.reshape(1, -1), format="fltp", layout="mono"
            )
            frame.sample_rate = sample_rate
            frame.pts = offset
            packets.extend(audio_stream.encode(frame))
        packets.extend(audio_stream.encode())
        for packet in packets:
            container.mux(packet)


def test_decode_media_aligns_audio_and_video_windows(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_av_video(path)

    video, audio, _spectrogram = decode_media(
        path,
        num_frames=1,
        image_size=32,
        duration=1.0,
        window_start=1.0,
    )
    spectrum = np.abs(np.fft.rfft(audio.numpy()))
    frequencies = np.fft.rfftfreq(audio.numel(), d=1.0 / 16_000)
    peak = float(frequencies[int(np.argmax(spectrum))])
    assert 800.0 <= peak <= 960.0
    # The first frame is black; the selected window is much brighter.
    assert float(video.mean()) > -0.7


def test_decode_media_uses_the_requested_window(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=4)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for index in range(16):
        array = np.full((32, 32, 3), index * 12, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    video, _audio, _spectrogram = decode_media(
        path,
        num_frames=1,
        image_size=32,
        duration=1.0,
        window_start=2.0,
    )
    # Frames in [2 s, 3 s) are much brighter than the first frame, whose
    # normalized mean is close to -1.8.
    assert float(video.mean()) > -0.5


def test_clip_sample_fields() -> None:
    sample = ClipSample(Path("a.mp4"), 0, "football", "match")
    assert sample.group == "match"
