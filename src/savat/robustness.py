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

from savat.data import MultimodalDataset, discover_clips, waveform_to_logmel
from savat.evaluate import (
    average_precision,
    confusion_matrix,
    macro_f1,
    top1_accuracy,
)
from savat.model import SAVAT, SAVATConfig
from savat.utils import resolve_device, seed_everything, write_json


def add_pink_noise(
    waveform: torch.Tensor, snr_db: float = 5.0, seed: int = 17
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
    video: torch.Tensor, kernel_size: int = 7, sigma: float = 1.5
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
                waveform_to_logmel(noisy[index].reshape(-1), sample_rate)
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
        shifted = torch.stack(
            [shift_audio(item, samples=samples) for item in waveform]
        )
        spectrograms = torch.stack(
            [
                waveform_to_logmel(shifted[index].reshape(-1), sample_rate)
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
) -> dict[str, Any]:
    """Evaluate one corruption and return metrics."""
    model.eval()
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []

    for video, audio, waveform, _reliability, labels, _index in loader:
        video = video.to(device)
        audio = audio.to(device)
        waveform = waveform.to(device)
        video, audio = apply_corruption(
            video, audio, waveform, corruption, sample_rate=sample_rate, seed=seed
        )
        logits = model(video, audio).joint_logits
        targets.append(labels.numpy())
        scores.append(torch.softmax(logits, dim=1).cpu().numpy())

    target_array = np.concatenate(targets)
    score_array = np.concatenate(scores)
    predictions = score_array.argmax(axis=1)
    per_class = average_precision(target_array, score_array, len(class_names))
    return {
        "condition": corruption.name,
        "kind": corruption.kind,
        "severity": corruption.severity,
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
    seed: int = 17,
) -> dict[str, Any]:
    """Evaluate every condition and report the drop from the clean reference."""
    reports = [
        evaluate_condition(model, loader, device, class_names, corruption, seed=seed)
        for corruption in corruptions
    ]
    reference = reports[0]["mean_average_precision"] if reports else 0.0
    for report in reports:
        report["delta_map"] = report["mean_average_precision"] - reference
    return {"reference": reports[0] if reports else None, "conditions": reports}


def default_corruptions(
    noise_snr: float = 5.0, blur_sigma: float = 1.5, offsets: tuple[float, ...] = (0.5,)
) -> list[Corruption]:
    """Return the clean, pink noise, blur and desynchronisation conditions."""
    conditions = [
        Corruption("clean", "clean", 0.0),
        Corruption(f"pink_noise_{noise_snr:g}db", "pink_noise", noise_snr),
        Corruption(f"blur_sigma_{blur_sigma:g}", "blur", blur_sigma),
    ]
    conditions.extend(
        Corruption(f"desync_{offset:g}s", "desync", offset) for offset in offsets
    )
    return conditions


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise-snr", type=float, default=5.0)
    parser.add_argument("--blur-sigma", type=float, default=1.5)
    parser.add_argument(
        "--offsets", default="0.5,1.0,1.5,2.0", help="comma-separated seconds"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("robustness.json"))
    args = parser.parse_args(argv)

    seed_everything(args.seed)
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
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    offsets = tuple(float(value) for value in args.offsets.split(",") if value)
    report = run_harness(
        model,
        loader,
        device,
        class_names,
        default_corruptions(args.noise_snr, args.blur_sigma, offsets),
        seed=args.seed,
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
