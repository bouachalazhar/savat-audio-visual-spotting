"""Tests for corruption operators and the robustness harness."""

from __future__ import annotations

import torch

from savat.robustness import (
    Corruption,
    add_pink_noise,
    apply_corruption,
    default_corruptions,
    gaussian_blur,
    shift_audio,
    shift_video,
)


def test_pink_noise_preserves_shape_and_changes_signal() -> None:
    waveform = torch.sin(torch.linspace(0, 40, 16_000)).unsqueeze(0)
    noisy = add_pink_noise(waveform, snr_db=0.0, seed=17)
    assert noisy.shape == waveform.shape
    assert not torch.allclose(noisy, waveform)


def test_pink_noise_is_reproducible() -> None:
    waveform = torch.randn(1, 1000)
    first = add_pink_noise(waveform, seed=3)
    second = add_pink_noise(waveform, seed=3)
    assert torch.allclose(first, second)


def test_gaussian_blur_preserves_shape() -> None:
    video = torch.randn(3, 4, 32, 32)
    blurred = gaussian_blur(video, kernel_size=5, sigma=1.0)
    assert blurred.shape == video.shape
    assert not torch.allclose(blurred, video)


def test_shift_audio_zero_fills() -> None:
    waveform = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    shifted = shift_audio(waveform, 3)
    assert torch.equal(shifted[0, :3], torch.zeros(3))
    assert torch.equal(shifted[0, 3:], waveform[0, :-3])


def test_shift_video_zero_fills() -> None:
    video = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    shifted = shift_video(video, 1)
    assert torch.equal(shifted[:, 0], torch.zeros_like(shifted[:, 0]))
    assert torch.equal(shifted[:, 1:], video[:, :-1])


def test_apply_corruption_dispatch() -> None:
    video = torch.randn(2, 3, 4, 32, 32)
    spectrogram = torch.rand(2, 1, 64, 320)
    waveform = torch.randn(2, 16_000)
    noisy_video, noisy_spectrogram = apply_corruption(
        video, spectrogram, waveform, Corruption("noise", "pink_noise", 5.0)
    )
    assert noisy_video.shape == video.shape
    assert noisy_spectrogram.shape == spectrogram.shape
    assert torch.equal(noisy_video, video)
    assert not torch.allclose(noisy_spectrogram, spectrogram)

    blurred_video, same_spectrogram = apply_corruption(
        video, spectrogram, waveform, Corruption("blur", "blur", 1.0)
    )
    assert not torch.allclose(blurred_video, video)
    assert torch.equal(same_spectrogram, spectrogram)


def test_default_conditions_cover_all_families() -> None:
    conditions = default_corruptions()
    kinds = [condition.kind for condition in conditions]
    assert kinds[0] == "clean"
    assert "pink_noise" in kinds
    assert "blur" in kinds
    assert kinds.count("desync") == 4
    assert {
        condition.severity for condition in conditions if condition.kind == "desync"
    } == {
        -2.0,
        -1.0,
        1.0,
        2.0,
    }
    assert conditions[1].severity == 10.0
    assert conditions[2].severity == 1.0
