"""Warm-start initialisation from third-party and prior checkpoints.

The paper pretraining starts the two modality encoders from CAV-MAE Sync
(BSD-2-Clause upstream, fetched with ``savat-fetch-init``). The architectures
differ in tokenisers, token budgets and decoders, so only shape-compatible
transformer encoder layers are transferred, layer by layer; everything else
keeps its own initialisation and is reported explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

# Parameter map between the upstream CAV-MAE ViT block and the
# ``nn.TransformerEncoderLayer`` used by SAVAT (both pre-norm residual
# wiring). The upstream ``attn.qkv`` is already the q, k, v concatenation
# that ``nn.MultiheadAttention`` stores as ``in_proj_weight``/``in_proj_bias``.
_LAYER_PARAMETER_MAP = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "attn.qkv.weight": "self_attn.in_proj_weight",
    "attn.qkv.bias": "self_attn.in_proj_bias",
    "attn.proj.weight": "self_attn.out_proj.weight",
    "attn.proj.bias": "self_attn.out_proj.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "linear1.weight",
    "mlp.fc1.bias": "linear1.bias",
    "mlp.fc2.weight": "linear2.weight",
    "mlp.fc2.bias": "linear2.bias",
}

_LAYER_KEY = re.compile(r"^blocks_([avu])\.(\d+)\.(.+)$")

# Upstream branch letter to the SAVAT branch that initialises from it. The
# upstream shared layer ``blocks_u`` has no counterpart in the compact
# branches and is skipped by design.
_UPSTREAM_BRANCH = {"v": "video_branch", "a": "audio_branch", "u": None}


@dataclass
class InitReport:
    """What a warm-start transferred and what it deliberately left alone."""

    source: Path
    transferred: int = 0
    transferred_names: list[str] = field(default_factory=list)
    skipped_shapes: int = 0
    skipped_names: int = 0

    def summary(self) -> str:
        """One-line human summary for training logs."""
        return (
            f"initialisation from {self.source}: {self.transferred} tensors"
            f" transferred, {self.skipped_shapes} skipped by shape,"
            f" {self.skipped_names} skipped by name"
        )


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint as a flat tensor mapping, tolerating wrappers."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} does not contain a tensor mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        if isinstance(value, torch.Tensor):
            state[key.removeprefix("module.")] = value
    return state


def _looks_like_cavmae(state: dict[str, torch.Tensor]) -> bool:
    """Detect the upstream CAV-MAE / CAV-MAE Sync key layout."""
    return any(
        key.startswith(("blocks_", "video_encoder", "audio_encoder")) for key in state
    )


def _warm_start_savat(
    model: nn.Module, state: dict[str, torch.Tensor], source: Path
) -> InitReport:
    """Copy every same-name, same-shape parameter from a SAVAT checkpoint."""
    report = InitReport(source=source)
    with torch.no_grad():
        for name, parameter in model.state_dict().items():
            value = state.get(name)
            if value is None:
                report.skipped_names += 1
                continue
            if value.shape != parameter.shape:
                report.skipped_shapes += 1
                continue
            parameter.copy_(value)
            report.transferred += 1
            report.transferred_names.append(name)
    return report


def _group_encoder_layers(
    state: dict[str, torch.Tensor],
) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    """Index upstream encoder layer tensors by ``(branch letter, layer)``."""
    layers: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        match = _LAYER_KEY.match(key)
        if match is None or match.group(3) not in _LAYER_PARAMETER_MAP:
            continue
        index = (match.group(1), int(match.group(2)))
        layers.setdefault(index, {})[match.group(3)] = value
    return layers


def _warm_start_cavmae(
    model: nn.Module, state: dict[str, torch.Tensor], source: Path
) -> InitReport:
    """Transfer shape-compatible upstream encoder layers into both branches."""
    report = InitReport(source=source)
    upstream = _group_encoder_layers(state)
    for letter, branch in _UPSTREAM_BRANCH.items():
        if branch is None:
            continue
        downstream = dict(model.named_parameters())
        branch_layers = max(
            (
                layer
                for (upstream_letter, layer) in upstream
                if upstream_letter == letter
            ),
            default=-1,
        )
        for layer_index in range(branch_layers + 1):
            sources = upstream.get((letter, layer_index), {})
            for suffix, target_suffix in _LAYER_PARAMETER_MAP.items():
                value = sources.get(suffix)
                if value is None:
                    continue
                target_name = f"{branch}.encoder.layers.{layer_index}.{target_suffix}"
                parameter = downstream.get(target_name)
                if parameter is None:
                    report.skipped_names += 1
                    continue
                if value.shape != parameter.shape:
                    report.skipped_shapes += 1
                    continue
                with torch.no_grad():
                    parameter.copy_(value)
                report.transferred += 1
                report.transferred_names.append(target_name)
    # Upstream tokenisers, positional and register tokens, decoders and CLI
    # heads have no SAVAT counterpart; they intentionally keep their own
    # initialisation. The shared ``blocks_u`` branch is skipped by design.
    return report


def apply_init_from(model: nn.Module, path: Path) -> InitReport:
    """Warm-start ``model`` from ``path`` and report what was transferred.

    Two layouts are supported: SAVAT checkpoints (including those saved by
    this package) and upstream CAV-MAE / CAV-MAE Sync files, whose encoder
    layers are remapped. Token budgets and architectures that do not match
    are intentionally not transferred.
    """
    state = _load_state(path)
    if _looks_like_cavmae(state):
        return _warm_start_cavmae(model, state, path)
    return _warm_start_savat(model, state, path)
