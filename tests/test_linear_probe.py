"""Tests for the frozen-encoder linear probe."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from savat.linear_probe import build_parser, extract_features, parse_args
from savat.model import SAVAT, SAVATConfig


class FakeClipDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(3, 8, 32, 32),
            torch.randn(1, 16, 32),
            torch.randn(16_000),
            torch.tensor([0.8, 0.6]),
            torch.tensor(index % 2),
            torch.tensor(index),
            torch.tensor(index // 2),
        )


def tiny_model() -> SAVAT:
    return SAVAT(
        SAVATConfig(
            image_size=32,
            num_frames=8,
            video_patch=16,
            tubelet=4,
            audio_mels=16,
            audio_time=32,
            audio_patch=8,
            embed_dim=32,
            depth=1,
            heads=4,
            decoder_depth=1,
            num_classes=2,
        )
    )


def test_extract_features_returns_reliability_weighted_fused_features() -> None:
    loader = DataLoader(FakeClipDataset(), batch_size=2)
    features, labels = extract_features(tiny_model(), loader, torch.device("cpu"))
    assert features.shape == (4, 64)
    assert labels.shape == (4,)
    assert labels.tolist() == [0, 1, 0, 1]


def test_linear_probe_parser_accepts_the_shared_profile(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--config",
            str(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "linear_probe_sportvista.yaml"
            ),
            "--data-root",
            str(tmp_path),
        ]
    )
    assert args.epochs == 50
    assert args.batch_size == 256
    assert args.class_weights == "balanced"
    assert (
        build_parser()
        .parse_args(["--data-root", "/tmp", "--checkpoint", "/tmp/best.pt"])
        .epochs
        == 50
    )
