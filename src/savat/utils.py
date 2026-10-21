"""Shared helpers: device selection, seeding and serialisation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from yaml.nodes import MappingNode


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects repeated mapping keys."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        keys: list[Any] = []
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f"duplicate YAML key: {key!r}")
            keys.append(key)
        return super().construct_mapping(node, deep=deep)


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the best available device for the requested policy."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(content, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


__all__ = ["load_yaml", "resolve_device", "seed_everything", "write_json"]
