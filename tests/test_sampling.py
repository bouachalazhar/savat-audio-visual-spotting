"""Tests for class-balanced sampling."""

from __future__ import annotations

import numpy as np

from savat.sampling import BalancedClassSampler


def test_balanced_sampler_visits_each_class_roughly_equally() -> None:
    sampler = BalancedClassSampler([0, 0, 0, 0, 1, 1], num_samples=8, seed=3)
    indices = list(sampler)
    counts = np.bincount([[0, 0, 0, 0, 1, 1][index] for index in indices], minlength=2)
    assert counts[0] == counts[1] == 4


def test_balanced_sampler_is_deterministic_per_epoch() -> None:
    sampler = BalancedClassSampler([0, 0, 1, 1, 2, 2], seed=7)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_balanced_sampler_rank_slices_are_disjoint() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    rank_zero = list(BalancedClassSampler(labels, seed=7, rank=0, world_size=2))
    rank_one = list(BalancedClassSampler(labels, seed=7, rank=1, world_size=2))
    assert set(rank_zero).isdisjoint(rank_one)
    assert len(rank_zero) + len(rank_one) == len(labels)


def test_balanced_sampler_has_equal_rank_lengths_when_divisible() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    lengths = {
        len(
            list(
                BalancedClassSampler(
                    labels, num_samples=6, seed=7, rank=rank, world_size=3
                )
            )
        )
        for rank in range(3)
    }
    assert lengths == {2}
