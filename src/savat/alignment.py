"""Reliability-weighted contrastive audio-visual alignment.

The symmetric InfoNCE loss pulls the video and audio embeddings of the same
clip together and pushes embeddings of other clips apart. When labels are
available, negatives can be restricted to the same normalised class, which
prevents the model from separating sports by scene or ambient sound instead of
by fine temporal correspondence.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, dim=-1, eps=1e-6)


def symmetric_infonce(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    temperature: float = 0.07,
    sample_weights: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    same_class_negatives: bool = False,
) -> torch.Tensor:
    """Symmetric InfoNCE between paired video and audio embeddings.

    Args:
        video: ``(B, D)`` video embeddings.
        audio: ``(B, D)`` audio embeddings.
        temperature: Softmax temperature, must be positive.
        sample_weights: Optional ``(B,)`` non-negative weights, typically the
            product of the two reliability scores. Zero-weight samples stay in
            the batch but do not contribute to the loss.
        labels: Optional ``(B,)`` class labels. Required when
            ``same_class_negatives`` is true.
        same_class_negatives: Restrict negative pairs to clips sharing a label.

    Returns:
        A scalar loss.
    """
    if video.shape != audio.shape:
        raise ValueError("video and audio embeddings must share a shape")
    if video.ndim != 2:
        raise ValueError("embeddings must have shape (B, D)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if same_class_negatives and labels is None:
        raise ValueError("labels are required for same-class negatives")

    video = _l2_normalize(video)
    audio = _l2_normalize(audio)
    logits = video @ audio.T / temperature
    batch = logits.shape[0]
    targets = torch.arange(batch, device=logits.device)

    if same_class_negatives:
        labels = torch.as_tensor(labels, device=logits.device).view(-1)
        if labels.shape[0] != batch:
            raise ValueError("one label per embedding is required")
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~same
        logits = logits.masked_fill(negative_mask, float("-inf"))

    weights = (
        torch.ones(batch, device=logits.device, dtype=logits.dtype)
        if sample_weights is None
        else torch.as_tensor(sample_weights, device=logits.device, dtype=logits.dtype)
    )
    if weights.shape != (batch,):
        raise ValueError("sample_weights must have shape (B,)")
    if torch.any(weights < 0):
        raise ValueError("sample_weights must be non-negative")

    row_loss = F.cross_entropy(logits, targets, reduction="none")
    column_loss = F.cross_entropy(logits.T, targets, reduction="none")
    weighted = weights * 0.5 * (row_loss + column_loss)
    denominator = weights.sum().clamp_min(1e-6)
    return weighted.sum() / denominator


def alignment_accuracy(video: torch.Tensor, audio: torch.Tensor) -> float:
    """Fraction of clips whose nearest audio embedding is the paired one."""
    if video.shape != audio.shape:
        raise ValueError("video and audio embeddings must share a shape")
    similarity = _l2_normalize(video) @ _l2_normalize(audio).T
    predictions = similarity.argmax(dim=1)
    targets = torch.arange(video.shape[0], device=video.device)
    return float((predictions == targets).float().mean())


__all__ = ["alignment_accuracy", "symmetric_infonce"]
