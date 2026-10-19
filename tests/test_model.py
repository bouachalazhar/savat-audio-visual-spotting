"""Tests for the SAVAT transformer."""

from __future__ import annotations

import torch

from savat.model import SAVAT, SAVATConfig, count_parameters


def tiny_config() -> SAVATConfig:
    return SAVATConfig(
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


def test_forward_shapes() -> None:
    model = SAVAT(tiny_config())
    output = model(torch.randn(2, 3, 8, 32, 32), torch.randn(2, 1, 16, 32))
    assert output.joint_logits.shape == (2, 3)
    assert output.video_logits.shape == (2, 3)
    assert output.audio_logits.shape == (2, 3)
    assert output.video_embedding.shape == (2, 32)
    assert count_parameters(model) > 0


def test_masked_reconstruction_shapes() -> None:
    model = SAVAT(tiny_config())
    output = model(
        torch.randn(2, 3, 8, 32, 32),
        torch.randn(2, 1, 16, 32),
        video_mask_ratio=0.5,
        audio_mask_ratio=0.5,
    )
    assert output.video_reconstruction is not None
    assert output.audio_reconstruction is not None
    assert output.video_mask is not None
    assert output.video_mask.shape == (2, 8)
    assert output.video_target is not None
    assert output.video_target.shape == (2, 8, 3 * 4 * 16 * 16)


def test_modality_dropout_keeps_one_branch() -> None:
    model = SAVAT(tiny_config())
    model.train()
    video = torch.randn(8, 3, 8, 32, 32)
    audio = torch.randn(8, 1, 16, 32)
    output = model(video, audio, modality_dropout=0.9)
    assert torch.isfinite(output.joint_logits).all()


def test_fused_probabilities_shape() -> None:
    model = SAVAT(tiny_config())
    model.eval()
    probabilities = model.fused_probabilities(
        torch.randn(3, 3, 8, 32, 32),
        torch.randn(3, 1, 16, 32),
        torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]),
    )
    assert probabilities.shape == (3, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)


def test_invalid_modality_dropout() -> None:
    model = SAVAT(tiny_config())
    try:
        model(
            torch.randn(2, 3, 8, 32, 32),
            torch.randn(2, 1, 16, 32),
            modality_dropout=1.0,
        )
    except ValueError:
        return
    raise AssertionError("invalid modality dropout should raise")
