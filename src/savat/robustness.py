"""Stress tests for noise, blur and audio-visual desynchronisation.

The three corruptions probe different failure modes and must not be confused:
pink noise degrades the acoustic signal only, Gaussian blur degrades spatial
detail only, and a temporal shift breaks the correspondence between the two
streams while leaving each stream intact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import DataLoader

from savat.config import parse_config_args
from savat.data import (
    AUDIO_TIME_FRAMES,
    MultimodalDataset,
    discover_clips,
    waveform_to_logmel,
)
from savat.evaluate import (
    average_precision,
    confusion_matrix,
    macro_f1,
    top1_accuracy,
)
from savat.model import SAVAT, SAVATConfig
from savat.reliability import ReliabilityNormalizer, reliability_scores
from savat.utils import resolve_device, seed_everything, write_json


def add_pink_noise(
    waveform: torch.Tensor, snr_db: float = 10.0, seed: int = 17
) -> torch.Tensor:
    """Add 1/f noise at a target signal-to-noise ratio.

    Args:
        waveform: ``(1, N)`` or ``(N,)`` audio tensor.
        snr_db: Signal-to-noise ratio in decibels.
        seed: Generator seed.
    """
    squeeze = waveform.ndim == 1
    signal = waveform.unsqueeze(0) if squeeze else waveform
    if signal.ndim != 2:
        raise ValueError("waveform must have shape (1, N) or (N,)")
    generator = torch.Generator(device=signal.device).manual_seed(seed)
    white = torch.randn(signal.shape, generator=generator, device=signal.device)
    spectrum = torch.fft.rfft(white, dim=-1)
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0, device=signal.device)
    shaping = 1.0 / torch.sqrt(frequencies.clamp_min(1.0 / signal.shape[-1]))
    noise = torch.fft.irfft(spectrum * shaping, n=signal.shape[-1], dim=-1)

    signal_power = signal.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * torch.sqrt(target / noise_power)
    result = signal + noise
    return result.squeeze(0) if squeeze else result


def gaussian_blur(
    video: torch.Tensor, kernel_size: int = 7, sigma: float = 1.0
) -> torch.Tensor:
    """Apply a Gaussian blur to every frame of a ``(C, T, H, W)`` clip."""
    if video.ndim != 4:
        raise ValueError("video must have shape (C, T, H, W)")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    frames = video.permute(1, 0, 2, 3)
    blurred = F.gaussian_blur(frames, kernel_size=kernel_size, sigma=sigma)
    return blurred.permute(1, 0, 2, 3)


def shift_audio(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    """Shift audio in time, filling the vacated samples with zeros."""
    if samples == 0:
        return waveform
    shifted = torch.zeros_like(waveform)
    if samples > 0:
        shifted[..., samples:] = waveform[..., :-samples]
    else:
        shifted[..., :samples] = waveform[..., -samples:]
    return shifted


def shift_video(video: torch.Tensor, frames: int) -> torch.Tensor:
    """Shift a clip in time, filling the vacated frames with zeros."""
    if frames == 0:
        return video
    shifted = torch.zeros_like(video)
    if frames > 0:
        shifted[:, frames:] = video[:, :-frames]
    else:
        shifted[:, :frames] = video[:, -frames:]
    return shifted


@dataclass(frozen=True)
class Corruption:
    """One stress-test condition."""

    name: str
    kind: str
    severity: float


def apply_corruption(
    video: torch.Tensor,
    spectrogram: torch.Tensor,
    waveform: torch.Tensor,
    corruption: Corruption,
    *,
    sample_rate: int = 16_000,
    seed: int = 17,
    time_frames: int = AUDIO_TIME_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one corruption to a batch and return video and spectrogram.

    Audio corruptions are applied to the waveform and the log-Mel map is
    recomputed, so that the model receives a physically consistent input.
    """
    if corruption.kind == "clean":
        return video, spectrogram
    if corruption.kind == "pink_noise":
        waveforms = [
            add_pink_noise(
                waveform[index], snr_db=corruption.severity, seed=seed + index
            )
            for index in range(waveform.shape[0])
        ]
        noisy = torch.stack(waveforms, dim=0)
        spectrograms = torch.stack(
            [
                waveform_to_logmel(
                    noisy[index].reshape(-1), sample_rate, time_frames=time_frames
                )
                for index in range(noisy.shape[0])
            ]
        )
        return video, spectrograms
    if corruption.kind == "blur":
        blurred = torch.stack(
            [gaussian_blur(clip, sigma=corruption.severity) for clip in video]
        )
        return blurred, spectrogram
    if corruption.kind == "desync":
        samples = int(round(corruption.severity * sample_rate))
        shifted = torch.stack([shift_audio(item, samples=samples) for item in waveform])
        spectrograms = torch.stack(
            [
                waveform_to_logmel(
                    shifted[index].reshape(-1), sample_rate, time_frames=time_frames
                )
                for index in range(shifted.shape[0])
            ]
        )
        return video, spectrograms
    raise ValueError(f"unknown corruption kind: {corruption.kind}")


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    corruption: Corruption,
    *,
    sample_rate: int = 16_000,
    seed: int = 17,
    time_frames: int = AUDIO_TIME_FRAMES,
    normalizer: ReliabilityNormalizer | None = None,
) -> dict[str, Any]:
    """Evaluate one corruption and return metrics.

    Reliability is recomputed on the corrupted signals, so a degraded modality
    receives a lower weight instead of reusing the clean score.
    """
    model.eval()
    targets: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []

    for video, audio, waveform, _reliability, labels, _index, _groups in loader:
        video = video.to(device)
        audio = audio.to(device)
        waveform = waveform.to(device)
        video, audio = apply_corruption(
            video,
            audio,
            waveform,
            corruption,
            sample_rate=sample_rate,
            seed=seed,
            time_frames=time_frames,
        )
        reliability_rows = []
        for index in range(video.shape[0]):
            scores = reliability_scores(video[index], audio[index])
            if normalizer is not None:
                scores = normalizer.transform(scores)
            reliability_rows.append(scores.as_tensor())
        reliability = torch.stack(reliability_rows).to(device)
        logits = model(video, audio, reliability=reliability).joint_logits
        targets.append(labels.numpy())
        score_batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    target_array = np.concatenate(targets)
    score_array = np.concatenate(score_batches)
    predictions = score_array.argmax(axis=1)
    per_class = average_precision(target_array, score_array, len(class_names))
    return {
        "condition": corruption.name,
        "kind": corruption.kind,
        "severity": corruption.severity,
        "seed": seed,
        "clips": int(target_array.size),
        "top1": top1_accuracy(target_array, predictions),
        "macro_f1": macro_f1(target_array, predictions, len(class_names)),
        "mean_average_precision": float(np.mean(per_class)),
        "confusion_matrix": confusion_matrix(
            target_array, predictions, len(class_names)
        ).tolist(),
    }


def run_harness(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    corruptions: list[Corruption],
    *,
    seeds: tuple[int, ...] = (17,),
    sample_rate: int = 16_000,
    time_frames: int = AUDIO_TIME_FRAMES,
    normalizer: ReliabilityNormalizer | None = None,
) -> dict[str, Any]:
    """Evaluate every condition over seeds and report the clean-reference drop.

    Each condition is evaluated once per seed. Metrics are averaged over seeds
    and the per-seed runs are kept in the report for inspection.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    reports: list[dict[str, Any]] = []
    for corruption in corruptions:
        runs = [
            evaluate_condition(
                model,
                loader,
                device,
                class_names,
                corruption,
                sample_rate=sample_rate,
                seed=seed,
                time_frames=time_frames,
                normalizer=normalizer,
            )
            for seed in seeds
        ]
        metric_names = ("top1", "macro_f1", "mean_average_precision")
        summary: dict[str, Any] = {
            "condition": corruption.name,
            "kind": corruption.kind,
            "severity": corruption.severity,
            "seeds": list(seeds),
            "clips": runs[0]["clips"],
            "runs": runs,
        }
        for name in metric_names:
            values = np.asarray([run[name] for run in runs], dtype=np.float64)
            summary[name] = float(values.mean())
            summary[f"{name}_std"] = float(values.std(ddof=0))
        reports.append(summary)

    clean = next(
        (report for report in reports if report["kind"] == "clean"),
        reports[0] if reports else None,
    )
    reference = clean["mean_average_precision"] if clean is not None else 0.0
    for report in reports:
        report["delta_map"] = report["mean_average_precision"] - reference
    return {"reference": clean, "conditions": reports}


def default_corruptions(
    noise_snr: float = 10.0,
    blur_sigma: float = 1.0,
    offsets: tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0),
) -> list[Corruption]:
    """Return the clean, pink noise, blur and desynchronisation conditions.

    Defaults follow the article's robustness protocol: 10 dB pink noise,
    Gaussian blur with sigma 1.0, and offsets of -2, -1, +1 and +2 seconds.
    """
    conditions = [
        Corruption("clean", "clean", 0.0),
        Corruption(f"pink_noise_{noise_snr:g}db", "pink_noise", noise_snr),
        Corruption(f"blur_sigma_{blur_sigma:g}", "blur", blur_sigma),
    ]
    conditions.extend(
        Corruption(f"desync_{offset:+g}s", "desync", offset) for offset in offsets
    )
    return conditions


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise-snr", type=float, default=10.0)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument(
        "--offsets",
        default="-2,-1,1,2",
        help="comma-separated seconds, signed offsets allowed",
    )
    parser.add_argument(
        "--seeds", default="17,29,43", help="comma-separated generator seeds"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("robustness.json"))
    args = parse_config_args(parser, argv, known_only=True)

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = SAVATConfig(**checkpoint["config"])
    model = SAVAT(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    normalizer = ReliabilityNormalizer.from_state_dict(
        checkpoint.get("reliability_normalizer", {})
    )

    class_names = list(checkpoint["class_names"])
    clips, _data_labels = discover_clips(args.data_root, label_order=class_names)
    dataset = MultimodalDataset(
        clips,
        num_frames=config.num_frames,
        image_size=config.image_size,
        duration=float(checkpoint.get("window_duration", 4.0)),
        window_start=checkpoint.get("window_start"),
        time_frames=int(checkpoint.get("time_frames", config.audio_time)),
        normalizer=normalizer,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    offsets = tuple(float(value) for value in args.offsets.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    report = run_harness(
        model,
        loader,
        device,
        class_names,
        default_corruptions(args.noise_snr, args.blur_sigma, offsets),
        seeds=seeds,
        time_frames=int(checkpoint.get("time_frames", config.audio_time)),
        normalizer=normalizer,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["data_root"] = str(args.data_root)
    write_json(args.output, report)
    for condition in report["conditions"]:
        print(
            f"{condition['condition']}: mAP={condition['mean_average_precision']:.4f} "
            f"delta={condition['delta_map']:+.4f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Corruption",
    "add_pink_noise",
    "apply_corruption",
    "default_corruptions",
    "evaluate_condition",
    "gaussian_blur",
    "run_harness",
    "shift_audio",
    "shift_video",
]
