"""Training entry point for SAVAT self-supervised pretraining and transfer.

The manuscript objective is self-supervised: reliability-weighted masked
reconstruction for each modality plus a reliability-weighted audio-visual
contrastive loss. A supervised classification term is optional and is disabled
by the paper pretraining profile; it remains available for generic transfer
experiments.

The YAML configuration supplies defaults for every command-line option. An
explicit flag always overrides the corresponding YAML value.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from savat.alignment import alignment_accuracy, symmetric_infonce
from savat.config import (
    parse_config_args,
    resolve_legacy_loss_options,
    serializable_args,
    validate_train_args,
)
from savat.data import (
    AUDIO_TIME_FRAMES,
    WINDOW_DURATION,
    MultimodalDataset,
    discover_clips,
    fit_reliability_normalizer,
    grouped_split,
)
from savat.evaluate import evaluate_model
from savat.model import SAVAT, SAVATConfig, SAVATOutput, count_parameters
from savat.sampling import BalancedClassSampler
from savat.utils import resolve_device, seed_everything, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/savat"))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--window-duration", type=float, default=WINDOW_DURATION)
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--reliability-fit-limit", type=int, default=0)
    parser.add_argument("--audio-time", type=int, default=AUDIO_TIME_FRAMES)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--video-mask-ratio",
        type=float,
        default=None,
        help="masked fraction of video tokens (paper profile: 0.75)",
    )
    parser.add_argument(
        "--audio-mask-ratio",
        type=float,
        default=None,
        help="masked fraction of audio tokens (paper profile: 0.50)",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help=(
            "legacy alias applied to both modalities when the separate flags are absent"
        ),
    )
    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.30,
        help="weight of reliability-weighted symmetric InfoNCE",
    )
    parser.add_argument(
        "--video-reconstruction-weight",
        type=float,
        default=None,
        help="weight of video masked reconstruction (paper profile: 0.40)",
    )
    parser.add_argument(
        "--audio-reconstruction-weight",
        type=float,
        default=None,
        help="weight of audio masked reconstruction (paper profile: 0.30)",
    )
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=None,
        help=(
            "legacy alias applied to both reconstructions when the separate "
            "flags are absent"
        ),
    )
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=0.0,
        help="optional supervised CE weight; 0 for the paper pretraining objective",
    )
    parser.add_argument(
        "--unimodal-ce-weight",
        type=float,
        default=0.5,
        help=(
            "relative weight of the two unimodal CE terms when classification "
            "is enabled"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="modality-dropout probability; paper transfer uses 0.25, pretraining 0",
    )
    parser.add_argument(
        "--same-class-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict contrastive negatives to the same class (paper profile: true)",
    )
    parser.add_argument(
        "--balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample approximately the same number of clips per class",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("alignment_loss", "macro_f1"),
        default="alignment_loss",
        help="validation criterion; macro_f1 requires a positive classification weight",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments, applying YAML defaults before explicit flags."""
    parser = build_parser()
    args = parse_config_args(parser, argv)
    resolve_legacy_loss_options(args)
    validate_train_args(args)
    return args


def _weighted_masked_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-sample masked MSE, reduced with reliability weights."""
    squared = ((reconstruction - target) ** 2).mean(dim=-1)
    per_sample = (squared * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    weights = weights.clamp_min(0.0)
    if float(weights.sum()) <= 0.0:
        return torch.zeros((), device=reconstruction.device)
    return (per_sample * weights).sum() / weights.sum()


def reconstruction_losses(
    output: SAVATOutput,
    reliability: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the reliability-weighted masked MSE for each modality.

    Missing masks produce a differentiable zero, which keeps the objective
    valid for a branch that was not masked in an ablation.
    """
    device = output.joint_logits.device
    losses: dict[str, torch.Tensor] = {
        "video": torch.zeros((), device=device),
        "audio": torch.zeros((), device=device),
    }
    for index, (name, reconstruction, mask, target) in enumerate(
        (
            (
                "video",
                output.video_reconstruction,
                output.video_mask,
                output.video_target,
            ),
            (
                "audio",
                output.audio_reconstruction,
                output.audio_mask,
                output.audio_target,
            ),
        )
    ):
        if reconstruction is None or mask is None or target is None:
            continue
        if reliability is None:
            weights = torch.ones(reconstruction.shape[0], device=reconstruction.device)
        else:
            weights = reliability[:, index]
        losses[name] = _weighted_masked_mse(reconstruction, target, mask, weights)
    return losses


def reconstruction_loss(
    output: SAVATOutput,
    video: torch.Tensor,
    audio: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean of the available per-modality reconstruction losses.

    ``video`` and ``audio`` are kept in the signature for backward
    compatibility with v1.0 callers; the targets are carried by ``output``.
    """
    del video, audio
    terms = reconstruction_losses(output, reliability)
    available = [
        terms[name]
        for name in ("video", "audio")
        if getattr(output, f"{name}_reconstruction") is not None
    ]
    if not available:
        return torch.zeros((), device=output.joint_logits.device)
    return torch.stack(available).mean()


def training_loss(
    output: SAVATOutput,
    labels: torch.Tensor,
    reliability: torch.Tensor,
    *,
    alignment_weight: float = 0.30,
    video_reconstruction_weight: float | None = None,
    audio_reconstruction_weight: float | None = None,
    temperature: float = 0.07,
    same_class_negatives: bool = False,
    classification_weight: float = 0.0,
    unimodal_ce_weight: float = 0.5,
    reconstruction_weight: float | None = None,
    groups: torch.Tensor | None = None,
    video: torch.Tensor | None = None,
    audio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the SAVAT objective and its components.

    The default weights are the paper pretraining objective:
    ``0.40 * L_rec_video + 0.30 * L_rec_audio + 0.30 * L_align``.
    The supervised terms are multiplied by ``classification_weight`` and are
    absent when it is zero. This is a self-supervised loss by default; the
    optional supervised path is intended for generic transfer experiments.
    """
    del video, audio  # reconstruction targets are stored in ``output``
    if video_reconstruction_weight is None:
        video_reconstruction_weight = (
            0.40 if reconstruction_weight is None else reconstruction_weight
        )
    if audio_reconstruction_weight is None:
        audio_reconstruction_weight = (
            0.30 if reconstruction_weight is None else reconstruction_weight
        )
    for name, value in (
        ("alignment_weight", alignment_weight),
        ("video_reconstruction_weight", video_reconstruction_weight),
        ("audio_reconstruction_weight", audio_reconstruction_weight),
        ("classification_weight", classification_weight),
        ("unimodal_ce_weight", unimodal_ce_weight),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    weights = reliability[..., 0] * reliability[..., 1]
    alignment = symmetric_infonce(
        output.video_embedding,
        output.audio_embedding,
        temperature=temperature,
        sample_weights=weights,
        labels=labels,
        groups=groups,
        same_class_negatives=same_class_negatives,
    )

    reconstruction_terms = reconstruction_losses(output, reliability)
    reconstruction = (
        video_reconstruction_weight * reconstruction_terms["video"]
        + audio_reconstruction_weight * reconstruction_terms["audio"]
    )
    total = alignment_weight * alignment + reconstruction

    joint_ce = torch.zeros((), device=total.device)
    video_ce = torch.zeros((), device=total.device)
    audio_ce = torch.zeros((), device=total.device)
    if classification_weight > 0:
        joint_ce = nn.functional.cross_entropy(output.joint_logits, labels)
        video_ce = nn.functional.cross_entropy(output.video_logits, labels)
        audio_ce = nn.functional.cross_entropy(output.audio_logits, labels)
        total = total + classification_weight * (
            joint_ce + unimodal_ce_weight * (video_ce + audio_ce)
        )

    components = {
        "loss": float(total.detach()),
        "joint_ce": float(joint_ce.detach()),
        "video_ce": float(video_ce.detach()),
        "audio_ce": float(audio_ce.detach()),
        "alignment": float(alignment.detach()),
        "video_reconstruction": float(reconstruction_terms["video"].detach()),
        "audio_reconstruction": float(reconstruction_terms["audio"].detach()),
        "reconstruction": float(reconstruction.detach()),
        "alignment_accuracy": alignment_accuracy(
            output.video_embedding.detach(), output.audio_embedding.detach()
        ),
    }
    return total, components


@torch.no_grad()
def evaluate_alignment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    temperature: float = 0.07,
    same_class_negatives: bool = True,
) -> dict[str, float]:
    """Evaluate the self-supervised alignment objective on a loader."""
    model.eval()
    loss_sum = 0.0
    accuracy_sum = 0.0
    clips = 0
    for video, audio, _waveform, reliability, labels, _index, groups in loader:
        video = video.to(device)
        audio = audio.to(device)
        reliability = reliability.to(device)
        labels = labels.to(device)
        groups = groups.to(device)
        output = model(video, audio, reliability=reliability)
        weights = reliability[..., 0] * reliability[..., 1]
        loss = symmetric_infonce(
            output.video_embedding,
            output.audio_embedding,
            temperature=temperature,
            sample_weights=weights,
            labels=labels,
            groups=groups,
            same_class_negatives=same_class_negatives,
        )
        batch = int(labels.numel())
        loss_sum += float(loss) * batch
        accuracy_sum += (
            alignment_accuracy(output.video_embedding, output.audio_embedding) * batch
        )
        clips += batch
    return {
        "alignment_loss": loss_sum / max(clips, 1),
        "alignment_accuracy": accuracy_sum / max(clips, 1),
        "clips": float(clips),
    }


def _make_loader(
    dataset: MultimodalDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=shuffle and len(dataset) > batch_size,
    )


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        # Rank 0 evaluates the full validation split and saves the checkpoint
        # while the other ranks wait at a barrier; the default 10-minute NCCL
        # timeout is too short for a real audio-visual validation pass.
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=2),
        )
        return rank, world, local, True
    return rank, world, local, False


def _validation_score(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, dict[str, float]]:
    """Return the selection score and a report for one validation pass."""
    report: dict[str, float] = {}
    if args.classification_weight > 0:
        classification = evaluate_model(model, loader, device, class_names)
        report.update(
            {
                "val_top1": float(classification["top1"]),
                "val_macro_f1": float(classification["macro_f1"]),
                "val_map": float(classification["mean_average_precision"]),
            }
        )
    if args.classification_weight <= 0 or args.selection_metric == "alignment_loss":
        alignment = evaluate_alignment(
            model,
            loader,
            device,
            temperature=args.temperature,
            same_class_negatives=args.same_class_negatives,
        )
        report.update(
            {
                "val_alignment_loss": alignment["alignment_loss"],
                "val_alignment_accuracy": alignment["alignment_accuracy"],
            }
        )

    if args.selection_metric == "macro_f1":
        return report["val_macro_f1"], report
    return -report["val_alignment_loss"], report


def train(args: argparse.Namespace) -> Path:
    """Train SAVAT and return the path of the best checkpoint."""
    rank, world, local_rank, distributed = _distributed_context()
    seed_everything(args.seed + rank, args.deterministic)
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips, class_names = discover_clips(args.data_root)
    train_clips, validation_clips, test_clips = grouped_split(
        clips,
        fractions=(
            1.0 - args.validation_fraction - args.test_fraction,
            args.validation_fraction,
            args.test_fraction,
        ),
        seed=args.seed,
    )

    train_dataset = MultimodalDataset(
        train_clips,
        num_frames=args.num_frames,
        image_size=args.image_size,
        duration=args.window_duration,
        window_start=args.window_start,
        time_frames=args.audio_time,
    )
    # Bounds are fitted on training clips only; validation and test clips are
    # transformed with those fixed bounds.
    normalizer = fit_reliability_normalizer(
        train_dataset, limit=args.reliability_fit_limit
    )
    train_dataset.normalizer = normalizer
    validation_dataset = MultimodalDataset(
        validation_clips,
        num_frames=args.num_frames,
        image_size=args.image_size,
        duration=args.window_duration,
        window_start=args.window_start,
        time_frames=args.audio_time,
        normalizer=normalizer,
    )

    train_labels = [clip.label for clip in train_clips]
    if args.balanced_batches:
        # Keep the rank-local shards equal so that every DDP rank performs the
        # same number of optimizer steps. A global epoch length not divisible by
        # the world size would otherwise leave one rank with an extra batch and
        # hang the next collective.
        balanced_size = len(train_dataset)
        if world > 1:
            balanced_size -= balanced_size % world
            if balanced_size == 0:
                raise ValueError(
                    "balanced DDP training needs at least one sample per rank"
                )
        train_sampler: Sampler[int] | None = BalancedClassSampler(
            train_labels,
            num_samples=balanced_size,
            seed=args.seed,
            rank=rank,
            world_size=world,
        )
    elif distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world, rank=rank, shuffle=True
        )
    else:
        train_sampler = None
    train_loader = _make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
    )
    # Validation is evaluated in full on rank 0. A DistributedSampler would
    # give rank 0 only one shard, which would make checkpoint selection noisy.
    validation_loader = (
        _make_loader(
            validation_dataset,
            args.batch_size,
            args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        if rank == 0
        else None
    )

    config = SAVATConfig(
        image_size=args.image_size,
        num_frames=args.num_frames,
        audio_time=args.audio_time,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        decoder_depth=args.decoder_depth,
        dropout=args.dropout,
        num_classes=len(class_names),
    )
    model: nn.Module = SAVAT(config).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_score = -math.inf
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    training_metadata = {
        **serializable_args(args),
        "objective": {
            "alignment_weight": args.alignment_weight,
            "video_reconstruction_weight": args.video_reconstruction_weight,
            "audio_reconstruction_weight": args.audio_reconstruction_weight,
            "classification_weight": args.classification_weight,
            "unimodal_ce_weight": args.unimodal_ce_weight,
        },
        "mask_ratios": {
            "video": args.video_mask_ratio,
            "audio": args.audio_mask_ratio,
        },
        "selection_metric": args.selection_metric,
        "balanced_batches": args.balanced_batches,
        "same_class_negatives": args.same_class_negatives,
        "audio_time": args.audio_time,
    }
    step = 0

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        model.train()
        running: dict[str, float] = {}
        started = time.perf_counter()

        for (
            video,
            audio,
            _waveform,
            reliability,
            labels,
            _index,
            groups,
        ) in train_loader:
            video = video.to(device)
            audio = audio.to(device)
            reliability = reliability.to(device)
            labels = labels.to(device)
            groups = groups.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                video,
                audio,
                video_mask_ratio=args.video_mask_ratio,
                audio_mask_ratio=args.audio_mask_ratio,
                modality_dropout=args.modality_dropout,
                reliability=reliability,
            )
            loss, components = training_loss(
                output,
                labels,
                reliability,
                alignment_weight=args.alignment_weight,
                video_reconstruction_weight=args.video_reconstruction_weight,
                audio_reconstruction_weight=args.audio_reconstruction_weight,
                temperature=args.temperature,
                same_class_negatives=args.same_class_negatives,
                classification_weight=args.classification_weight,
                unimodal_ce_weight=args.unimodal_ce_weight,
                groups=groups,
                video=video,
                audio=audio,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            for name, value in components.items():
                running[name] = running.get(name, 0.0) + value

        if rank == 0:
            assert validation_loader is not None
            state_source = model.module if isinstance(model, DDP) else model
            score, report = _validation_score(
                args, state_source, validation_loader, device, class_names
            )
            epoch_report = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "selection_score": score,
                **report,
            }
            for name, value in running.items():
                epoch_report[f"train_{name}"] = value / max(1, len(train_loader))
            history.append(epoch_report)
            print(json.dumps(epoch_report))
            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "state_dict": state_source.state_dict(),
                        "config": asdict(config),
                        "class_names": class_names,
                        "reliability_normalizer": normalizer.state_dict(),
                        "window_duration": args.window_duration,
                        "window_start": args.window_start,
                        "time_frames": args.audio_time,
                        "epoch": epoch,
                        "selection_score": best_score,
                        "selection_metric": args.selection_metric,
                        "training": training_metadata,
                    },
                    best_path,
                )
        if distributed and dist.is_initialized():
            dist.barrier()

    test_report: dict[str, float] | None = None
    test_alignment: dict[str, float] | None = None
    if rank == 0 and test_clips and best_path.exists():
        test_dataset = MultimodalDataset(
            test_clips,
            num_frames=args.num_frames,
            image_size=args.image_size,
            duration=args.window_duration,
            window_start=args.window_start,
            time_frames=args.audio_time,
            normalizer=normalizer,
        )
        test_loader = _make_loader(
            test_dataset,
            args.batch_size,
            args.workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        state_source = model.module if isinstance(model, DDP) else model
        best_state = torch.load(best_path, map_location="cpu", weights_only=True)
        state_source.load_state_dict(best_state["state_dict"])
        if args.classification_weight > 0:
            test_report = evaluate_model(state_source, test_loader, device, class_names)
        else:
            test_alignment = evaluate_alignment(
                state_source,
                test_loader,
                device,
                temperature=args.temperature,
                same_class_negatives=args.same_class_negatives,
            )

    if rank == 0:
        write_json(
            args.output_dir / "history.json",
            {
                "config": asdict(config),
                "arguments": serializable_args(args),
                "parameters": count_parameters(model),
                "class_names": class_names,
                "train_clips": len(train_clips),
                "validation_clips": len(validation_clips),
                "test_clips": len(test_clips),
                "history": history,
                "best_score": best_score,
                "best_metric": args.selection_metric,
                "test": test_report,
                "test_alignment": test_alignment,
            },
        )

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return best_path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    checkpoint = train(args)
    print(f"best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "evaluate_alignment",
    "main",
    "parse_args",
    "reconstruction_loss",
    "reconstruction_losses",
    "train",
    "training_loss",
]
