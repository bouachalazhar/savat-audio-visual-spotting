"""Training entry point for SAVAT pretraining and SoccerNet transfer.

The objective combines supervised classification, masked reconstruction and
reliability-weighted audio-visual alignment. A YAML configuration can supply
defaults for every command-line option.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from savat.alignment import alignment_accuracy, symmetric_infonce
from savat.data import (
    AUDIO_TIME_FRAMES,
    MultimodalDataset,
    discover_clips,
    fit_reliability_normalizer,
    grouped_split,
)
from savat.evaluate import evaluate_model
from savat.model import SAVAT, SAVATConfig, SAVATOutput, count_parameters
from savat.utils import load_yaml, resolve_device, seed_everything, write_json


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
    parser.add_argument("--window-duration", type=float, default=5.0)
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--reliability-fit-limit", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--same-class-negatives", action="store_true")
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
    preliminary, _unknown = parser.parse_known_args(argv)
    if preliminary.config is not None:
        defaults = load_yaml(preliminary.config)
        valid = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = set(defaults) - valid
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


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


def reconstruction_loss(
    output: SAVATOutput,
    video: torch.Tensor,
    audio: torch.Tensor,
    reliability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reliability-weighted mean squared error over masked patches only."""
    losses: list[torch.Tensor] = []
    for index, (reconstruction, mask, target) in enumerate(
        (
            (output.video_reconstruction, output.video_mask, output.video_target),
            (output.audio_reconstruction, output.audio_mask, output.audio_target),
        )
    ):
        if reconstruction is None or mask is None or target is None:
            continue
        if reliability is None:
            weights = torch.ones(reconstruction.shape[0], device=reconstruction.device)
        else:
            weights = reliability[:, index]
        losses.append(_weighted_masked_mse(reconstruction, target, mask, weights))
    if not losses:
        return torch.zeros((), device=video.device)
    return torch.stack(losses).mean()


def training_loss(
    output: SAVATOutput,
    labels: torch.Tensor,
    reliability: torch.Tensor,
    *,
    alignment_weight: float,
    reconstruction_weight: float,
    temperature: float,
    same_class_negatives: bool,
    video: torch.Tensor | None = None,
    audio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the SAVAT objective and its components."""
    joint = nn.functional.cross_entropy(output.joint_logits, labels)
    video_ce = nn.functional.cross_entropy(output.video_logits, labels)
    audio_ce = nn.functional.cross_entropy(output.audio_logits, labels)
    weights = reliability[..., 0] * reliability[..., 1]
    alignment = symmetric_infonce(
        output.video_embedding,
        output.audio_embedding,
        temperature=temperature,
        sample_weights=weights,
        labels=labels,
        same_class_negatives=same_class_negatives,
    )
    total = joint + 0.5 * (video_ce + audio_ce) + alignment_weight * alignment

    reconstruction = torch.zeros((), device=joint.device)
    if reconstruction_weight > 0 and video is not None and audio is not None:
        reconstruction = reconstruction_loss(output, video, audio, reliability)
        total = total + reconstruction_weight * reconstruction

    components = {
        "loss": float(total.detach()),
        "joint_ce": float(joint.detach()),
        "video_ce": float(video_ce.detach()),
        "audio_ce": float(audio_ce.detach()),
        "alignment": float(alignment.detach()),
        "reconstruction": float(reconstruction.detach()),
        "alignment_accuracy": alignment_accuracy(
            output.video_embedding.detach(), output.audio_embedding.detach()
        ),
    }
    return total, components


def _make_loader(
    dataset: MultimodalDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    sampler: DistributedSampler | None = None,
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
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return rank, world, local, True
    return rank, world, local, False


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
        time_frames=AUDIO_TIME_FRAMES,
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
        time_frames=AUDIO_TIME_FRAMES,
        normalizer=normalizer,
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True)
        if distributed
        else None
    )
    train_loader = _make_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        shuffle=True,
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
        audio_time=AUDIO_TIME_FRAMES,
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
    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running: dict[str, float] = {}
        started = time.perf_counter()

        for video, audio, _waveform, reliability, labels, _index in train_loader:
            video = video.to(device)
            audio = audio.to(device)
            reliability = reliability.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                video,
                audio,
                video_mask_ratio=args.mask_ratio,
                audio_mask_ratio=args.mask_ratio,
                modality_dropout=args.modality_dropout,
                reliability=reliability,
            )
            loss, components = training_loss(
                output,
                labels,
                reliability,
                alignment_weight=args.alignment_weight,
                reconstruction_weight=args.reconstruction_weight,
                temperature=args.temperature,
                same_class_negatives=args.same_class_negatives,
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

        report: dict[str, float] | None = None
        if rank == 0:
            assert validation_loader is not None
            report = evaluate_model(
                model.module if isinstance(model, DDP) else model,
                validation_loader,
                device,
                class_names,
            )
            epoch_report = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "val_top1": report["top1"],
                "val_macro_f1": report["macro_f1"],
                "val_map": report["mean_average_precision"],
            }
            for name, value in running.items():
                epoch_report[f"train_{name}"] = value / max(1, len(train_loader))
            history.append(epoch_report)
            print(json.dumps(epoch_report))
            if report["macro_f1"] > best_macro_f1:
                best_macro_f1 = report["macro_f1"]
                state_source = model.module if isinstance(model, DDP) else model
                torch.save(
                    {
                        "state_dict": state_source.state_dict(),
                        "config": asdict(config),
                        "class_names": class_names,
                        "reliability_normalizer": normalizer.state_dict(),
                        "window_duration": args.window_duration,
                        "window_start": args.window_start,
                        "time_frames": AUDIO_TIME_FRAMES,
                        "epoch": epoch,
                        "val_macro_f1": best_macro_f1,
                    },
                    best_path,
                )
        if distributed and dist.is_initialized():
            dist.barrier()

    test_report: dict[str, float] | None = None
    if rank == 0 and test_clips and best_path.exists():
        test_dataset = MultimodalDataset(
            test_clips,
            num_frames=args.num_frames,
            image_size=args.image_size,
            duration=args.window_duration,
            window_start=args.window_start,
            time_frames=AUDIO_TIME_FRAMES,
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
        test_report = evaluate_model(state_source, test_loader, device, class_names)

    if rank == 0:
        write_json(
            args.output_dir / "history.json",
            {
                "config": asdict(config),
                "arguments": vars(args),
                "parameters": count_parameters(model),
                "class_names": class_names,
                "train_clips": len(train_clips),
                "validation_clips": len(validation_clips),
                "test_clips": len(test_clips),
                "history": history,
                "best_val_macro_f1": best_macro_f1,
                "test": test_report,
            },
        )

    if distributed and dist.is_initialized():
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
    "main",
    "parse_args",
    "reconstruction_loss",
    "train",
    "training_loss",
]
