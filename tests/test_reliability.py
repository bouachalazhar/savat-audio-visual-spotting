"""Tests for reliability indicators and normalisation."""

from __future__ import annotations

import torch

from savat.reliability import (
    ReliabilityNormalizer,
    audio_indicators,
    reliability_scores,
    video_indicators,
)


def make_spectrogram(energy: float = 0.5) -> torch.Tensor:
    spectrogram = torch.zeros((1, 16, 32))
    spectrogram[:, 4:12, :] = energy
    return spectrogram


def make_video() -> torch.Tensor:
    video = torch.zeros((3, 8, 32, 32))
    video[:, :, 8:24, 8:24] = 1.0
    for frame in range(8):
        video[:, frame, 4 + frame : 8 + frame, 16:20] = 2.0
    return video


def test_audio_indicators_are_bounded() -> None:
    indicators = audio_indicators(make_spectrogram())
    assert set(indicators) == {"presence", "stability", "no_clipping", "tonality"}
    assert all(0.0 <= value <= 1.0 for value in indicators.values())


def test_silent_audio_has_zero_presence() -> None:
    indicators = audio_indicators(torch.zeros((1, 16, 32)))
    assert indicators["presence"] == 0.0


def test_video_indicators_are_bounded() -> None:
    indicators = video_indicators(make_video())
    assert set(indicators) == {"sharpness", "exposure", "stability"}
    assert all(0.0 <= value <= 1.0 for value in indicators.values())


def test_reliability_scores_shapes() -> None:
    scores = reliability_scores(make_video(), make_spectrogram())
    tensor = scores.as_tensor()
    assert tensor.shape == (2,)
    assert torch.all(tensor >= 0)
    assert torch.all(tensor <= 1)


def test_normalizer_fit_and_transform() -> None:
    samples = [
        {
            "video": {"sharpness": 0.1, "exposure": 0.2, "stability": 0.3},
            "audio": {
                "presence": 0.2,
                "stability": 0.3,
                "no_clipping": 0.4,
                "tonality": 0.5,
            },
        },
        {
            "video": {"sharpness": 0.9, "exposure": 0.8, "stability": 0.7},
            "audio": {
                "presence": 0.8,
                "stability": 0.7,
                "no_clipping": 0.6,
                "tonality": 0.9,
            },
        },
    ]
    normalizer = ReliabilityNormalizer().fit(samples)
    transformed = normalizer.transform(
        reliability_scores(make_video(), make_spectrogram())
    )
    assert 0.0 <= transformed.video <= 1.0
    assert 0.0 <= transformed.audio <= 1.0
