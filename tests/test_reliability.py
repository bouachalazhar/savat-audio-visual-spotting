"""Tests for reliability indicators and normalisation."""

from __future__ import annotations

import torch

from savat.reliability import (
    ReliabilityNormalizer,
    ReliabilityScores,
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

    restored = ReliabilityNormalizer.from_state_dict(normalizer.state_dict())
    assert restored.bounds == normalizer.bounds
    assert restored.medians == normalizer.medians


def test_constant_indicator_still_penalises_degradation() -> None:
    normalizer = ReliabilityNormalizer().fit(
        [
            {
                "video": {"sharpness": 1.0, "exposure": 1.0, "stability": 1.0},
                "audio": {
                    "presence": 1.0,
                    "stability": 1.0,
                    "no_clipping": 1.0,
                    "tonality": 1.0,
                },
            }
        ]
    )
    perfect = normalizer.transform(
        ReliabilityScores(
            video=1.0,
            audio=1.0,
            video_indicators={"sharpness": 1.0, "exposure": 1.0, "stability": 1.0},
            audio_indicators={
                "presence": 1.0,
                "stability": 1.0,
                "no_clipping": 1.0,
                "tonality": 1.0,
            },
        )
    )
    clipped = normalizer.transform(
        ReliabilityScores(
            video=1.0,
            audio=1.0,
            video_indicators={"sharpness": 1.0, "exposure": 1.0, "stability": 1.0},
            audio_indicators={
                "presence": 1.0,
                "stability": 1.0,
                "no_clipping": 0.5,
                "tonality": 1.0,
            },
        )
    )
    assert perfect.audio == 1.0
    assert clipped.audio < perfect.audio


def test_normalizer_imputes_missing_indicators_with_training_median() -> None:
    normalizer = ReliabilityNormalizer().fit(
        [
            {
                "video": {"sharpness": 0.2, "exposure": 0.4, "stability": 0.6},
                "audio": {
                    "presence": 0.2,
                    "stability": 0.4,
                    "no_clipping": 0.6,
                    "tonality": 0.8,
                },
            },
            {
                "video": {"sharpness": 0.4, "exposure": 0.6, "stability": 0.8},
                "audio": {
                    "presence": 0.4,
                    "stability": 0.6,
                    "no_clipping": 0.8,
                },
            },
        ]
    )
    scores = ReliabilityScores(
        video=0.5,
        audio=0.5,
        video_indicators={"sharpness": 0.4},
        audio_indicators={"presence": 0.4},
    )
    transformed = normalizer.transform(scores)
    assert 0.0 <= transformed.video <= 1.0
    assert 0.0 <= transformed.audio <= 1.0
    assert normalizer.medians["audio.tonality"] == 0.8
