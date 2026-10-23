"""Frozen-encoder linear probing for SAVAT checkpoints.

The article evaluates the learned representation with a linear classifier while
the encoder is frozen. This module implements that protocol for a checkpoint
saved by :mod:`savat.train`: features are the reliability-weighted concatenation
of the two class tokens, the encoder is kept in evaluation mode, and only a
LayerNorm + linear head is trained.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from savat.config import parse_config_args
from savat.data import (
    MultimodalDataset,
    checkpoint_window_duration,
    discover_clips,
    grouped_split,
)
from savat.evaluate import (
    average_precision,
    confusion_matrix,
    macro_f1,
    top1_accuracy,
)
from savat.model import SAVAT, SAVATConfig
from savat.reliability import ReliabilityNormalizer
from savat.sampling import BalancedClassSampler
from savat.utils import resolve_device, seed_everything, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the linear-probe command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/savat_probe"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weights", choices=("none", "balanced"), default="balanced"
    )
    parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments, applying the shared YAML protocol where relevant."""
    parser = build_parser()
    args = parse_config_args(parser, argv, known_only=True)
    fractions = (
        1.0 - args.validation_fraction - args.test_fraction,
        args.validation_fraction,
        args.test_fraction,
    )
    if min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("probe train/validation/test fractions must sum to one")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    return args


@torch.no_grad()
def extract_features(
    model: SAVAT,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen reliability-weighted features and labels."""
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for video, audio, _waveform, reliability, targets, _index, _groups in loader:
        video = video.to(device)
        audio = audio.to(device)
        reliability = reliability.to(device)
        video_cls = model.video_branch(video, 0.0)[0]
        audio_cls = model.audio_branch(audio, 0.0)[0]
        weights = reliability.clamp(0.0, 1.0)
        fused = torch.cat(
            [video_cls * weights[:, 0:1], audio_cls * weights[:, 1:2]], dim=1
        )
        features.append(fused.cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def _class_weights(
    labels: torch.Tensor, num_classes: int, mode: str
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float64)
    if torch.any(counts == 0):
        raise ValueError("every class must have at least one training sample")
    return (counts.sum() / (num_classes * counts)).to(torch.float32)


def _report(
    targets: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    per_class = average_precision(targets, scores, num_classes)
    return {
        "clips": int(targets.size),
        "top1": top1_accuracy(targets, predictions),
        "macro_f1": macro_f1(targets, predictions, num_classes),
        "mean_average_precision": float(np.mean(per_class)),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(
            targets, predictions, num_classes
        ).tolist(),
    }


def linear_probe(args: argparse.Namespace) -> Path:
    """Extract frozen features, train a linear head and return its checkpoint."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = SAVATConfig(**checkpoint["config"])
    model = SAVAT(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    class_names = list(checkpoint["class_names"])
    normalizer = ReliabilityNormalizer.from_state_dict(
        checkpoint.get("reliability_normalizer", {})
    )

    clips, _labels = discover_clips(args.data_root, label_order=class_names)
    train_clips, validation_clips, test_clips = grouped_split(
        clips,
        fractions=(
            1.0 - args.validation_fraction - args.test_fraction,
            args.validation_fraction,
            args.test_fraction,
        ),
        seed=args.seed,
    )

    def dataset(selected: list) -> MultimodalDataset:
        return MultimodalDataset(
            selected,
            num_frames=config.num_frames,
            image_size=config.image_size,
            duration=checkpoint_window_duration(checkpoint),
            window_start=checkpoint.get("window_start"),
            time_frames=int(checkpoint.get("time_frames", config.audio_time)),
            normalizer=normalizer,
        )

    loaders = {
        "train": DataLoader(
            dataset(train_clips),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
        "validation": DataLoader(
            dataset(validation_clips),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
        "test": DataLoader(
            dataset(test_clips),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
    }
    features = {
        name: extract_features(model, loader, device)
        for name, loader in loaders.items()
    }

    train_features, train_labels = features["train"]
    validation_features, validation_labels = features["validation"]
    probe = nn.Sequential(
        nn.LayerNorm(train_features.shape[1]),
        nn.Linear(train_features.shape[1], len(class_names)),
    ).to(device)
    weights = _class_weights(train_labels, len(class_names), args.class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device)
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    sampler = (
        BalancedClassSampler(
            train_labels.tolist(),
            num_samples=len(train_labels),
            seed=args.seed,
        )
        if args.balanced_batches
        else None
    )
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        if isinstance(sampler, BalancedClassSampler):
            sampler.set_epoch(epoch)
        probe.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * batch_labels.numel()
            seen += batch_labels.numel()

        probe.eval()
        with torch.no_grad():
            validation_scores = torch.softmax(
                probe(validation_features.to(device)), dim=1
            ).cpu()
        validation_predictions = validation_scores.argmax(dim=1).numpy()
        validation_targets = validation_labels.numpy()
        score = macro_f1(validation_targets, validation_predictions, len(class_names))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "val_macro_f1": score,
                "seconds": time.perf_counter() - started,
            }
        )
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no finite validation score was produced")
    probe.load_state_dict(best_state)
    probe.eval()
    results: dict[str, Any] = {}
    for name in ("validation", "test"):
        with torch.no_grad():
            scores = (
                torch.softmax(probe(features[name][0].to(device)), dim=1).cpu().numpy()
            )
        predictions = scores.argmax(axis=1)
        results[name] = _report(
            features[name][1].numpy(), predictions, scores, class_names
        )
    results["checkpoint"] = str(args.checkpoint)
    results["data_root"] = str(args.data_root)
    results["probe_config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    results["best_val_macro_f1"] = best_score
    results["history"] = history
    payload = {
        "state_dict": probe.state_dict(),
        "class_names": class_names,
        "feature_dim": int(train_features.shape[1]),
        "source_checkpoint": str(args.checkpoint),
        "config": checkpoint["config"],
    }
    probe_path = args.output_dir / "probe.pt"
    torch.save(payload, probe_path)
    reloaded = torch.load(probe_path, map_location="cpu", weights_only=True)
    for key, value in payload["state_dict"].items():
        if not torch.equal(reloaded["state_dict"][key].cpu(), value.cpu()):
            raise RuntimeError(f"probe checkpoint verification failed: {key}")
    write_json(args.output_dir / "metrics.json", results)
    print(
        f"test top1={results['test']['top1']:.4f} "
        f"macro_f1={results['test']['macro_f1']:.4f}"
    )
    return args.output_dir / "probe.pt"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    linear_probe(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "extract_features",
    "linear_probe",
    "main",
    "parse_args",
]
