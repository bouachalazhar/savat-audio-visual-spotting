"""Samplers used by the SAVAT training loop."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BalancedClassSampler(Sampler[int]):
    """Yield class-balanced batches with guaranteed same-class negatives.

    Each batch draws a small set of classes and at least two examples from
    every selected class whenever ``batch_size >= 2``. This matters for the
    same-class contrastive objective: a batch with one example per class leaves
    an anchor with no admissible negative. Classes are rotated across batches
    and the sampler is deterministic for a given ``seed`` and epoch. With a
    distributed run each rank receives a disjoint set of complete batches, so
    the ranks perform the same number of optimizer steps.
    """

    def __init__(
        self,
        labels: list[int] | np.ndarray,
        *,
        num_samples: int | None = None,
        batch_size: int = 1,
        seed: int = 17,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        array = np.asarray(labels, dtype=np.int64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional array")
        if num_samples is not None and num_samples < 1:
            raise ValueError("num_samples must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("rank must lie in [0, world_size)")
        self.labels = array
        self.num_samples = int(num_samples or array.size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.classes = np.unique(array)
        self.class_indices = [np.flatnonzero(array == value) for value in self.classes]

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic permutation for an epoch."""
        self.epoch = int(epoch)

    def _num_batches(self) -> int:
        batches = int(math.ceil(self.num_samples / self.batch_size))
        if self.world_size == 1:
            return batches
        # Keep every rank on the same number of optimizer steps.
        return max(
            self.world_size,
            int(math.ceil(batches / self.world_size)) * self.world_size,
        )

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        num_classes = len(self.classes)
        classes_per_batch = min(num_classes, max(1, self.batch_size // 2))
        base, extra = divmod(self.batch_size, classes_per_batch)
        class_order = generator.permutation(num_classes)
        pools = [generator.permutation(indices) for indices in self.class_indices]
        cursors = [0] * num_classes
        batches: list[list[int]] = []

        def draw(class_index: int, count: int) -> np.ndarray:
            pool = pools[class_index]
            if count > len(pool):
                return generator.choice(pool, size=count, replace=True)
            start = cursors[class_index]
            if start + count <= len(pool):
                chosen = pool[start : start + count]
                cursors[class_index] = start + count
                return chosen
            head = pool[start:]
            tail_count = count - len(head)
            pools[class_index] = generator.permutation(pool)
            chosen = np.concatenate((head, pools[class_index][:tail_count]))
            cursors[class_index] = tail_count
            return chosen

        for batch_index in range(self._num_batches()):
            positions = [
                (batch_index * classes_per_batch + offset) % num_classes
                for offset in range(classes_per_batch)
            ]
            batch: list[int] = []
            for position, offset in enumerate(positions):
                class_index = int(class_order[offset])
                count = base + (1 if position < extra else 0)
                batch.extend(int(value) for value in draw(class_index, count))
            order = generator.permutation(len(batch))
            batches.append([batch[int(index)] for index in order])

        local_batches = batches[self.rank :: self.world_size]
        return iter(index for batch in local_batches for index in batch)

    def __len__(self) -> int:
        return (self._num_batches() // self.world_size) * self.batch_size


__all__ = ["BalancedClassSampler"]
