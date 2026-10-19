"""Tests for reliability-weighted fusion."""

from __future__ import annotations

import torch

from savat.fusion import (
    geometric_fusion,
    normalize_reliability,
    reliability_weights,
)


def probabilities() -> dict[str, torch.Tensor]:
    return {
        "video": torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
        "audio": torch.tensor([[0.6, 0.4], [0.9, 0.1]]),
    }


def test_normalize_reliability_sums_to_one() -> None:
    weights = normalize_reliability(torch.tensor([[1.0, 3.0], [0.0, 0.0]]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(weights[0], torch.tensor([0.25, 0.75]))
    assert torch.allclose(weights[1], torch.tensor([0.5, 0.5]))


def test_normalize_reliability_rejects_negative() -> None:
    try:
        normalize_reliability(torch.tensor([[-1.0, 2.0]]))
    except ValueError:
        return
    raise AssertionError("negative reliability should raise")


def test_arithmetic_fusion_with_uniform_weights() -> None:
    fused = geometric_fusion(
        probabilities(),
        {"video": torch.tensor([0.5, 0.5]), "audio": torch.tensor([0.5, 0.5])},
        strategy="arithmetic",
    )
    expected = 0.5 * probabilities()["video"][0] + 0.5 * probabilities()["audio"][0]
    assert torch.allclose(fused[0], expected, atol=1e-6)


def test_zero_weight_ignores_modality() -> None:
    fused = geometric_fusion(
        probabilities(),
        {"video": torch.tensor([1.0, 1.0]), "audio": torch.tensor([0.0, 0.0])},
        strategy="geometric",
    )
    assert torch.allclose(fused[0], probabilities()["video"][0], atol=1e-5)


def test_hard_fusion_selects_most_reliable() -> None:
    fused = geometric_fusion(
        probabilities(),
        {"video": torch.tensor([0.1, 0.1]), "audio": torch.tensor([0.9, 0.9])},
        strategy="hard",
    )
    assert torch.allclose(fused[0], probabilities()["audio"][0])


def test_video_only_and_audio_only() -> None:
    weights = {"video": torch.tensor([0.5, 0.5]), "audio": torch.tensor([0.5, 0.5])}
    assert torch.allclose(
        geometric_fusion(probabilities(), weights, strategy="video_only")[0],
        probabilities()["video"][0],
    )
    assert torch.allclose(
        geometric_fusion(probabilities(), weights, strategy="audio_only")[0],
        probabilities()["audio"][0],
    )


def test_fused_rows_sum_to_one() -> None:
    weights = {"video": torch.tensor([0.2, 0.8]), "audio": torch.tensor([0.8, 0.2])}
    for strategy in ("arithmetic", "geometric", "hard"):
        fused = geometric_fusion(probabilities(), weights, strategy=strategy)
        assert torch.allclose(fused.sum(dim=1), torch.ones(2), atol=1e-5)


def test_reliability_weights_mapping() -> None:
    weights = reliability_weights(torch.tensor([[2.0, 2.0], [1.0, 3.0]]))
    assert set(weights) == {"video", "audio"}
    assert torch.allclose(weights["video"], torch.tensor([0.5, 0.25]))
    assert torch.allclose(weights["audio"], torch.tensor([0.5, 0.75]))
