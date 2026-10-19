"""Classification and spotting metrics for SAVAT.

The article reports Top-1 and macro-F1 for linear probing, and average
precision (avg-mAP) for the SoccerNet transfer. Both families are implemented
without scikit-learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from savat.data import MultimodalDataset, discover_clips
from savat.fusion import geometric_fusion, reliability_weights
from savat.model import SAVAT, SAVATConfig
from savat.utils import resolve_device, write_json


def top1_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Fraction of exact matches."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.size == 0:
        raise ValueError("targets must not be empty")
    return float((targets == predictions).mean())


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Rows are targets, columns are predictions."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    pairs = zip(np.asarray(targets), np.asarray(predictions), strict=True)
    for target, prediction in pairs:
        matrix[int(target), int(prediction)] += 1
    return matrix


def macro_f1(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> float:
    """Unweighted mean of the per-class F1 scores."""
    matrix = confusion_matrix(targets, predictions, num_classes)
    scores: list[float] = []
    for index in range(num_classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def average_precision(
    targets: np.ndarray, scores: np.ndarray, num_classes: int
) -> list[float]:
    """One-vs-rest average precision for each class."""
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != num_classes:
        raise ValueError("scores must have shape (N, num_classes)")

    precisions: list[float] = []
    for index in range(num_classes):
        positive = targets == index
        support = int(positive.sum())
        if support == 0:
            precisions.append(0.0)
            continue
        order = np.argsort(-scores[:, index], kind="stable")
        ranked = positive[order]
        hits = np.cumsum(ranked)
        precision_at_k = hits / np.arange(1, ranked.size + 1)
        precisions.append(float(precision_at_k[ranked].mean()))
    return precisions


def mean_average_precision(
    targets: np.ndarray, scores: np.ndarray, num_classes: int
) -> float:
    """Mean of the per-class average precisions."""
    return float(np.mean(average_precision(targets, scores, num_classes)))


def fuse_output(
    output: torch.Tensor,
    audio_logits: torch.Tensor,
    reliability: torch.Tensor,
    *,
    strategy: str = "geometric",
) -> torch.Tensor:
    """Fuse video and audio probabilities with reliability weights."""
    weights = reliability_weights(reliability)
    probabilities = {
        "video": torch.softmax(output, dim=1),
        "audio": torch.softmax(audio_logits, dim=1),
    }
    return geometric_fusion(probabilities, weights, strategy=strategy)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    fusion: str = "joint",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return targets, predicted classes and class scores for a loader.

    Args:
        fusion: ``joint`` uses the joint head, ``geometric`` fuses the two
            unimodal heads with reliability weights, and ``video_only`` or
            ``audio_only`` select one branch.
    """
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for video, audio, _waveform, reliability, labels, _index in loader:
        video = video.to(device)
        audio = audio.to(device)
        reliability = reliability.to(device)
        output = model(video, audio)

        if fusion == "joint":
            logits = output.joint_logits
        elif fusion in {"geometric", "arithmetic", "hard"}:
            weights = reliability_weights(reliability)
            probabilities = {
                "video": torch.softmax(output.video_logits, dim=1),
                "audio": torch.softmax(output.audio_logits, dim=1),
            }
            fused = geometric_fusion(probabilities, weights, strategy=fusion)
            logits = torch.log(fused.clamp_min(1e-8))
        elif fusion == "video_only":
            logits = output.video_logits
        elif fusion == "audio_only":
            logits = output.audio_logits
        else:
            raise ValueError(f"unknown fusion mode: {fusion}")

        all_targets.append(labels.numpy())
        all_scores.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())

    return (
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        np.concatenate(all_scores),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    *,
    fusion: str = "joint",
) -> dict[str, Any]:
    """Evaluate a model and return a serialisable report."""
    targets, predictions, scores = collect_predictions(
        model, loader, device, fusion=fusion
    )
    num_classes = len(class_names)
    matrix = confusion_matrix(targets, predictions, num_classes)
    per_class_ap = average_precision(targets, scores, num_classes)
    return {
        "fusion": fusion,
        "clips": int(targets.size),
        "top1": top1_accuracy(targets, predictions),
        "macro_f1": macro_f1(targets, predictions, num_classes),
        "mean_average_precision": float(np.mean(per_class_ap)),
        "per_class_average_precision": {
            name: value for name, value in zip(class_names, per_class_ap, strict=True)
        },
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fusion",
        default="joint",
        choices=[
            "joint",
            "geometric",
            "arithmetic",
            "hard",
            "video_only",
            "audio_only",
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SAVATConfig(**checkpoint["config"])
    model = SAVAT(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    clips, class_names = discover_clips(args.data_root)
    dataset = MultimodalDataset(
        clips,
        num_frames=config.num_frames,
        image_size=config.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    report = evaluate_model(
        model, loader, device, class_names, fusion=args.fusion
    )
    report["checkpoint"] = str(args.checkpoint)
    report["data_root"] = str(args.data_root)
    write_json(args.output, report)
    print(
        f"clips={report['clips']} top1={report['top1']:.4f} "
        f"macro_f1={report['macro_f1']:.4f} "
        f"mAP={report['mean_average_precision']:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "average_precision",
    "collect_predictions",
    "confusion_matrix",
    "evaluate_model",
    "fuse_output",
    "macro_f1",
    "main",
    "mean_average_precision",
    "top1_accuracy",
]
