"""Tests for the training objective and argument handling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from savat.model import SAVAT, SAVATConfig, SAVATOutput
from savat.train import (
    parse_args,
    reconstruction_loss,
    reconstruction_losses,
    training_loss,
)


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
    output = model(video, audio, video_mask_ratio=0.5, audio_mask_ratio=0.5)
    loss, components = training_loss(
        output,
        labels,
        reliability,
        alignment_weight=0.1,
        video_reconstruction_weight=0.1,
        audio_reconstruction_weight=0.1,
        temperature=0.07,
        same_class_negatives=False,
        video=video,
        audio=audio,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert set(components) >= {
        "loss",
        "alignment",
        "video_reconstruction",
        "audio_reconstruction",
    }


def test_reconstruction_loss_without_masks_is_zero() -> None:
    model, _config = tiny_model()
    output = model(torch.randn(2, 3, 8, 32, 32), torch.randn(2, 1, 16, 32))
    loss = reconstruction_loss(output, torch.empty(0), torch.empty(0))
    assert float(loss) == 0.0


def test_reconstruction_loss_downweights_unreliable_samples() -> None:
    reconstruction = torch.zeros(2, 2, 3)
    target = torch.stack([torch.zeros(2, 3), torch.full((2, 3), 2.0)], dim=0)
    mask = torch.ones(2, 2, dtype=torch.bool)
    output = SAVATOutput(
        joint_logits=torch.zeros(2, 3),
        video_logits=torch.zeros(2, 3),
        audio_logits=torch.zeros(2, 3),
        video_embedding=torch.zeros(2, 4),
        audio_embedding=torch.zeros(2, 4),
        video_reconstruction=reconstruction,
        video_mask=mask,
        video_target=target,
    )
    reliable_video = reconstruction_loss(
        output,
        video=torch.empty(0),
        audio=torch.empty(0),
        reliability=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    )
    unreliable_audio = reconstruction_loss(
        output,
        video=torch.empty(0),
        audio=torch.empty(0),
        reliability=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )
    assert float(reliable_video) == 0.0
    assert float(unreliable_audio) == 0.0

    all_weighted = reconstruction_loss(
        output,
        video=torch.empty(0),
        audio=torch.empty(0),
        reliability=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )
    assert float(all_weighted) == 2.0


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


def test_paper_pretraining_loss_has_no_supervised_term() -> None:
    model, _config = tiny_model()
    video = torch.randn(2, 3, 8, 32, 32)
    audio = torch.randn(2, 1, 16, 32)
    labels = torch.tensor([0, 2])
    reliability = torch.ones(2, 2)
    output = model(
        video,
        audio,
        video_mask_ratio=0.75,
        audio_mask_ratio=0.5,
        reliability=reliability,
    )
    loss, components = training_loss(
        output,
        labels,
        reliability,
        alignment_weight=0.3,
        video_reconstruction_weight=0.4,
        audio_reconstruction_weight=0.3,
        classification_weight=0.0,
        same_class_negatives=False,
        video=video,
        audio=audio,
    )
    assert torch.isfinite(loss)
    assert components["joint_ce"] == 0.0
    assert components["video_ce"] == 0.0
    assert components["audio_ce"] == 0.0
    expected = (
        0.3 * components["alignment"]
        + 0.4 * components["video_reconstruction"]
        + 0.3 * components["audio_reconstruction"]
    )
    assert components["loss"] == pytest.approx(expected, rel=1e-5)


def test_separate_modality_losses_are_reported_separately() -> None:
    model, _config = tiny_model()
    video = torch.randn(2, 3, 8, 32, 32)
    audio = torch.randn(2, 1, 16, 32)
    reliability = torch.ones(2, 2)
    output = model(
        video,
        audio,
        video_mask_ratio=0.75,
        audio_mask_ratio=0.5,
        reliability=reliability,
    )
    terms = reconstruction_losses(output, reliability)
    assert set(terms) == {"video", "audio"}
    assert torch.isfinite(terms["video"])
    assert torch.isfinite(terms["audio"])


def test_classification_term_can_be_enabled_for_transfer() -> None:
    model, _config = tiny_model()
    video = torch.randn(2, 3, 8, 32, 32)
    audio = torch.randn(2, 1, 16, 32)
    labels = torch.tensor([0, 2])
    reliability = torch.ones(2, 2)
    output = model(video, audio, reliability=reliability)
    _loss, components = training_loss(
        output,
        labels,
        reliability,
        classification_weight=1.0,
        alignment_weight=0.05,
        video_reconstruction_weight=0.05,
        audio_reconstruction_weight=0.05,
        same_class_negatives=False,
        video=video,
        audio=audio,
    )
    assert components["joint_ce"] > 0.0
    assert components["video_ce"] > 0.0
    assert components["audio_ce"] > 0.0
