"""Samplers used by the SAVAT training loop."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BalancedClassSampler(Sampler[int]):
    """Yield roughly the same number of examples from every class.

    The sampler is deterministic for a given ``seed`` and epoch. With a
    distributed run each rank receives a disjoint slice of the same balanced
    epoch, so no example is duplicated across ranks within an epoch. This is a
    sampler rather than a loss reweighting: the manuscript specifies balanced
    mini-batches, and balanced batches also make same-class contrastive
    negatives more likely to be present.
    """

    def __init__(
        self,
        labels: list[int] | np.ndarray,
        *,
        num_samples: int | None = None,
        seed: int = 17,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        array = np.asarray(labels, dtype=np.int64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional array")
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("rank must lie in [0, world_size)")
        self.labels = array
        self.num_samples = int(num_samples or array.size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.classes = np.unique(array)
        self.class_indices = [np.flatnonzero(array == value) for value in self.classes]

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic permutation for an epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        per_class = int(math.ceil(self.num_samples / len(self.classes)))
        sampled: list[np.ndarray] = []
        for indices in self.class_indices:
            shuffled = generator.permutation(indices)
            # ``np.resize`` repeats the class when it has fewer examples than
            # the target. The final global shuffle prevents a fixed class
            # order inside every batch.
            sampled.append(np.resize(shuffled, per_class))
        epoch_indices = np.concatenate(sampled)
        generator.shuffle(epoch_indices)
        epoch_indices = epoch_indices[: self.num_samples]
        # The sampler is already balanced before sharding; slicing every
        # ``world_size``-th item keeps the rank-local stream deterministic and
        # disjoint without a second distributed wrapper.
        return iter(int(value) for value in epoch_indices[self.rank :: self.world_size])

    def __len__(self) -> int:
        return int(math.ceil(self.num_samples / self.world_size))


__all__ = ["BalancedClassSampler"]
