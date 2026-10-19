"""Per-modality reliability indicators.

The article aggregates normalized indicators with a geometric mean, so that a
modality whose single indicator collapses receives a low score. The raw
indicators below are bounded in ``[0, 1]`` by construction and are grouped in
three families: availability, quality and temporal stability. They are
technical descriptors, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

AUDIO_INDICATORS = ("presence", "stability", "no_clipping", "tonality")
VIDEO_INDICATORS = ("sharpness", "exposure", "stability")


def _geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("at least one indicator is required")
    if min(values) < 0:
        raise ValueError("indicators must be non-negative")
    if min(values) == 0:
        return 0.0
    product = 1.0
    for value in values:
        product *= value
    return float(product ** (1.0 / len(values)))


def audio_indicators(spectrogram: torch.Tensor) -> dict[str, float]:
    """Compute audio indicators from a ``(1, F, T)`` normalized log-Mel map.

    Args:
        spectrogram: Values in ``[0, 1]`` as produced by
            :func:`savat.data.waveform_to_logmel`.

    Returns:
        ``presence`` (mean energy), ``stability`` (1 minus the coefficient of
        variation of frame energy), ``no_clipping`` (fraction of saturated
        bins) and ``tonality`` (1 minus spectral flatness).
    """
    if spectrogram.ndim != 3 or spectrogram.shape[0] != 1:
        raise ValueError("spectrogram must have shape (1, F, T)")
    values = spectrogram.clamp(0, 1)
    frame_energy = values.mean(dim=1).reshape(-1)
    mean_energy = float(frame_energy.mean())
    presence = min(1.0, mean_energy * 5.0)

    variation = float(frame_energy.std()) / (mean_energy + 1e-6)
    stability = max(0.0, 1.0 - min(1.0, variation))

    clipping = float((values > 0.99).float().mean())
    no_clipping = 1.0 - clipping

    bins = values.mean(dim=-1).clamp_min(1e-4)
    flatness = float(torch.exp(torch.log(bins).mean()) / bins.mean())
    tonality = max(0.0, 1.0 - min(1.0, flatness))

    return {
        "presence": presence,
        "stability": stability,
        "no_clipping": no_clipping,
        "tonality": tonality,
    }


def video_indicators(video: torch.Tensor) -> dict[str, float]:
    """Compute video indicators from a ``(C, T, H, W)`` normalized clip.

    Args:
        video: ImageNet-normalized clip, channel first.

    Returns:
        ``sharpness`` (Laplacian energy), ``exposure`` (distance of the mean
        intensity to the centre of the normalized range) and ``stability``
        (regularity of frame-to-frame differences).
    """
    if video.ndim != 4:
        raise ValueError("video must have shape (C, T, H, W)")
    gray = video.mean(dim=0)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=gray.dtype
    ).view(1, 1, 3, 3)
    gray4 = gray.unsqueeze(1)
    laplacian = torch.nn.functional.conv2d(gray4, kernel, padding=1)
    sharpness = float(laplacian.var()) / (float(laplacian.var()) + 0.01)
    sharpness = min(1.0, sharpness)

    exposure = max(0.0, 1.0 - min(1.0, abs(float(gray.mean()))))

    if gray.shape[0] > 1:
        differences = (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2))
        mean = float(differences.mean())
        variation = float(differences.std()) / (mean + 1e-6)
        stability = max(0.0, 1.0 - min(1.0, variation))
    else:
        stability = 1.0

    return {
        "sharpness": sharpness,
        "exposure": exposure,
        "stability": stability,
    }


@dataclass
class ReliabilityScores:
    """Aggregated reliability of one audio-visual pair."""

    video: float
    audio: float
    video_indicators: dict[str, float] = field(default_factory=dict)
    audio_indicators: dict[str, float] = field(default_factory=dict)

    def as_tensor(self) -> torch.Tensor:
        """Return ``[video, audio]`` as a float tensor."""
        return torch.tensor([self.video, self.audio], dtype=torch.float32)


def reliability_scores(
    video: torch.Tensor, spectrogram: torch.Tensor
) -> ReliabilityScores:
    """Compute the geometric-mean reliability of both modalities."""
    video_raw = video_indicators(video)
    audio_raw = audio_indicators(spectrogram)
    video_score = _geometric_mean([video_raw[name] for name in VIDEO_INDICATORS])
    audio_score = _geometric_mean([audio_raw[name] for name in AUDIO_INDICATORS])
    return ReliabilityScores(
        video=video_score,
        audio=audio_score,
        video_indicators=video_raw,
        audio_indicators=audio_raw,
    )


class ReliabilityNormalizer:
    """Min-max normalisation fitted on the training partition only."""

    def __init__(self) -> None:
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, samples: list[dict[str, dict[str, float]]]) -> ReliabilityNormalizer:
        """Fit bounds from raw indicator dictionaries.

        Args:
            samples: One dictionary per clip, with ``video`` and ``audio``
                keys, each mapping indicator names to raw values.
        """
        for modality, names in (
            ("video", VIDEO_INDICATORS),
            ("audio", AUDIO_INDICATORS),
        ):
            for name in names:
                values = [
                    float(sample[modality][name])
                    for sample in samples
                    if name in sample.get(modality, {})
                ]
                if values:
                    self.bounds[f"{modality}.{name}"] = (min(values), max(values))
        return self

    def _normalise(self, modality: str, name: str, value: float) -> float:
        key = f"{modality}.{name}"
        if key not in self.bounds:
            return max(0.0, min(1.0, value))
        low, high = self.bounds[key]
        if high - low < 1e-9:
            return 1.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def transform(self, scores: ReliabilityScores) -> ReliabilityScores:
        """Apply the fitted bounds and recompute both geometric means."""
        video = {
            name: self._normalise("video", name, value)
            for name, value in scores.video_indicators.items()
        }
        audio = {
            name: self._normalise("audio", name, value)
            for name, value in scores.audio_indicators.items()
        }
        return ReliabilityScores(
            video=_geometric_mean([video[name] for name in VIDEO_INDICATORS]),
            audio=_geometric_mean([audio[name] for name in AUDIO_INDICATORS]),
            video_indicators=video,
            audio_indicators=audio,
        )


__all__ = [
    "AUDIO_INDICATORS",
    "VIDEO_INDICATORS",
    "ReliabilityNormalizer",
    "ReliabilityScores",
    "audio_indicators",
    "reliability_scores",
    "video_indicators",
]
