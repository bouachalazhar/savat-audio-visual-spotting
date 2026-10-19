"""Tests for audio-visual contrastive alignment."""

from __future__ import annotations

import torch

from savat.alignment import alignment_accuracy, symmetric_infonce


def test_identical_modalities_are_perfectly_aligned() -> None:
    embeddings = torch.eye(4)
    assert alignment_accuracy(embeddings, embeddings) == 1.0


def test_shuffled_pairs_have_lower_accuracy() -> None:
    embeddings = torch.eye(4)
    shuffled = embeddings[[1, 2, 3, 0]]
    assert alignment_accuracy(embeddings, shuffled) == 0.0


def test_aligned_loss_is_lower_than_shuffled_loss() -> None:
    torch.manual_seed(0)
    video = torch.randn(8, 16)
    audio = video + 0.01 * torch.randn(8, 16)
    aligned = symmetric_infonce(video, audio, temperature=0.1)
    permuted = audio[[1, 2, 3, 4, 5, 6, 7, 0]]
    shuffled = symmetric_infonce(video, permuted, temperature=0.1)
    assert aligned < shuffled


def test_zero_weights_give_zero_loss() -> None:
    video = torch.randn(4, 8)
    audio = torch.randn(4, 8)
    loss = symmetric_infonce(video, audio, sample_weights=torch.zeros(4))
    assert float(loss) == 0.0


def test_same_class_negatives_require_labels() -> None:
    video = torch.randn(4, 8)
    audio = torch.randn(4, 8)
    try:
        symmetric_infonce(video, audio, same_class_negatives=True)
    except ValueError:
        return
    raise AssertionError("labels should be required")


def test_same_class_negatives_run() -> None:
    torch.manual_seed(1)
    video = torch.randn(6, 8)
    audio = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = symmetric_infonce(
        video, audio, labels=labels, same_class_negatives=True
    )
    assert torch.isfinite(loss)
    assert float(loss) > 0
