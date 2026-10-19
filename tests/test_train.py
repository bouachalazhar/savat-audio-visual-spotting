"""Tests for the training objective and argument handling."""

from __future__ import annotations

from pathlib import Path

import torch

from savat.model import SAVAT, SAVATConfig
from savat.train import parse_args, reconstruction_loss, training_loss


def tiny_model() -> tuple[SAVAT, SAVATConfig]:
    config = SAVATConfig(
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
        num_classes=3,
    )
    return SAVAT(config), config


def test_training_loss_is_finite_and_backward() -> None:
    model, _config = tiny_model()
    video = torch.randn(2, 3, 8, 32, 32)
    audio = torch.randn(2, 1, 16, 32)
    labels = torch.tensor([0, 2])
    reliability = torch.tensor([[0.9, 0.4], [0.2, 0.8]])
    output = model(
        video, audio, video_mask_ratio=0.5, audio_mask_ratio=0.5
    )
    loss, components = training_loss(
        output,
        labels,
        reliability,
        alignment_weight=0.1,
        reconstruction_weight=0.1,
        temperature=0.07,
        same_class_negatives=False,
        video=video,
        audio=audio,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert set(components) >= {"loss", "joint_ce", "alignment", "reconstruction"}


def test_reconstruction_loss_without_masks_is_zero() -> None:
    model, _config = tiny_model()
    output = model(torch.randn(2, 3, 8, 32, 32), torch.randn(2, 1, 16, 32))
    loss = reconstruction_loss(output, torch.empty(0), torch.empty(0))
    assert float(loss) == 0.0


def test_parse_args_reads_yaml_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("epochs: 3\nbatch_size: 2\nseed: 43\n", encoding="utf-8")
    args = parse_args(["--config", str(config), "--data-root", str(tmp_path)])
    assert args.epochs == 3
    assert args.batch_size == 2
    assert args.seed == 43


def test_parse_args_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("unknown_option: 1\n", encoding="utf-8")
    try:
        parse_args(["--config", str(config), "--data-root", str(tmp_path)])
    except ValueError:
        return
    raise AssertionError("unknown configuration keys should raise")
