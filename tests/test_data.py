"""Tests for clip discovery, grouped splits and tensor transforms."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch

from savat.data import (
    ClipSample,
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
    groups = [
        {clip.group for clip in split} for split in (train, validation, test)
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_grouped_split_is_deterministic(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "data")
    clips, _labels = discover_clips(root)
    first = grouped_split(clips, seed=29)
    second = grouped_split(clips, seed=29)
    assert [
        [clip.path for clip in split] for split in first
    ] == [[clip.path for clip in split] for split in second]


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
    assert spectrogram.shape == (1, 64, 256)


def test_clip_sample_fields() -> None:
    sample = ClipSample(Path("a.mp4"), 0, "football", "match")
    assert sample.group == "match"
