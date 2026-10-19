"""Reliability-weighted fusion of unimodal predictions.

The article defines the fused representation from modality features weighted
by reliability. At decision level, the same idea is applied to the unimodal
probability vectors: a modality with a low reliability contributes less, and a
fully unreliable modality can be ignored without removing the sample.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

STRATEGIES = ("arithmetic", "geometric", "hard", "video_only", "audio_only")


def normalize_reliability(reliability: torch.Tensor) -> torch.Tensor:
    """Normalise per-sample reliability scores so that they sum to one.

    Args:
        reliability: Tensor of shape ``(B, M)`` or ``(M,)`` with non-negative
            scores. Columns are modalities, in the order used by the caller.

    Returns:
        A tensor with the same shape whose last dimension sums to one. Rows
        with all-zero scores fall back to uniform weights, so a sample is never
        discarded.
    """
    if not isinstance(reliability, torch.Tensor):
        reliability = torch.as_tensor(reliability, dtype=torch.float32)
    if reliability.ndim not in (1, 2):
        raise ValueError("reliability must be one- or two-dimensional")
    if torch.any(reliability < 0):
        raise ValueError("reliability scores must be non-negative")

    total = reliability.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(reliability, 1.0 / reliability.shape[-1])
    return torch.where(total > 0, reliability / total.clamp_min(1e-12), uniform)


def _as_weight_tensor(
    probabilities: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    if isinstance(weights, Mapping):
        missing = set(probabilities) - set(weights)
        if missing:
            raise KeyError(f"missing weights for modalities: {sorted(missing)}")
        return {name: torch.as_tensor(weights[name]) for name in probabilities}

    tensor = torch.as_tensor(weights)
    names = list(probabilities)
    if tensor.shape[-1] != len(names):
        raise ValueError("weight tensor must have one column per modality")
    return {name: tensor[..., index] for index, name in enumerate(names)}


def geometric_fusion(
    probabilities: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor] | torch.Tensor,
    *,
    strategy: str = "geometric",
    floor: float = 1e-6,
) -> torch.Tensor:
    """Fuse unimodal probability vectors.

    Args:
        probabilities: Mapping from modality name to a ``(B, C)`` probability
            tensor. Every tensor must share the same shape.
        weights: Reliability weights, either a mapping from modality name to a
            ``(B,)`` tensor or a ``(B, M)`` tensor ordered like
            ``probabilities``.
        strategy: ``arithmetic``, ``geometric``, ``hard``, ``video_only`` or
            ``audio_only``.
        floor: Lower bound applied before the logarithm.

    Returns:
        A ``(B, C)`` probability tensor that sums to one along classes.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown fusion strategy {strategy!r}")
    if not probabilities:
        raise ValueError("at least one modality is required")

    names = list(probabilities)
    tensors = [torch.as_tensor(probabilities[name]) for name in names]
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("all probability tensors must share the same shape")
    if shape[-1] < 2:
        raise ValueError("the class dimension must contain at least two classes")

    if strategy == "video_only":
        if "video" not in probabilities:
            raise KeyError("video_only requires a 'video' modality")
        return tensors[names.index("video")]
    if strategy == "audio_only":
        if "audio" not in probabilities:
            raise KeyError("audio_only requires an 'audio' modality")
        return tensors[names.index("audio")]

    weight_map = _as_weight_tensor(probabilities, weights)
    batch = shape[0]
    expanded: dict[str, torch.Tensor] = {}
    for name in names:
        weight = torch.as_tensor(weight_map[name], dtype=tensors[0].dtype)
        if weight.ndim == 0:
            weight = weight.expand(batch)
        if weight.shape != (batch,):
            raise ValueError("each modality weight must have shape (B,)")
        expanded[name] = weight

    if strategy == "hard":
        stacked = torch.stack([expanded[name] for name in names], dim=1)
        best = torch.argmax(stacked, dim=1)
        fused = torch.zeros_like(tensors[0])
        for index in range(len(names)):
            selected = (best == index).unsqueeze(1)
            fused = torch.where(selected, tensors[index], fused)
        return fused / fused.sum(dim=1, keepdim=True).clamp_min(floor)

    if strategy == "arithmetic":
        fused = torch.zeros_like(tensors[0])
        for name, tensor in zip(names, tensors, strict=True):
            fused = fused + expanded[name].unsqueeze(1) * tensor
    else:
        log_probabilities = torch.stack(
            [torch.log(tensor.clamp_min(floor)) for tensor in tensors], dim=1
        )
        stacked_weights = torch.stack([expanded[name] for name in names], dim=1)
        fused_log = (stacked_weights.unsqueeze(-1) * log_probabilities).sum(dim=1)
        fused = torch.exp(fused_log)

    return fused / fused.sum(dim=1, keepdim=True).clamp_min(floor)


def reliability_weights(reliability: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return video and audio weights from a ``(B, 2)`` reliability tensor."""
    if not isinstance(reliability, torch.Tensor):
        reliability = torch.as_tensor(reliability, dtype=torch.float32)
    if reliability.ndim == 1:
        if reliability.numel() != 2:
            raise ValueError("two reliability scores are required")
        reliability = reliability.unsqueeze(0)
    if reliability.shape[-1] != 2:
        raise ValueError("reliability must have two columns: video and audio")
    normalized = normalize_reliability(reliability)
    return {"video": normalized[..., 0], "audio": normalized[..., 1]}


__all__ = [
    "STRATEGIES",
    "geometric_fusion",
    "normalize_reliability",
    "reliability_weights",
]
