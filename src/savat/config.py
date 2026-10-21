"""Strict YAML configuration handling shared by SAVAT entry points.

Configuration files are flat mappings whose keys are the command-line
destinations. They are deliberately simple: a protocol can be inspected without
knowing the Python argument parser, while unknown keys and invalid ranges fail
before a long training run starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from savat.utils import load_yaml


def _config_path(argv: list[str] | None) -> Path | None:
    """Find ``--config`` before normal parsing so it can supply required values."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, argument in enumerate(arguments):
        if argument == "--config" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1])
    return None


def parse_config_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    known_only: bool = False,
) -> argparse.Namespace:
    """Parse arguments, applying YAML defaults before explicit flags.

    Args:
        parser: Command parser defining the accepted destinations.
        argv: Optional argument list, mainly for tests.
        known_only: Ignore keys that belong to another SAVAT command. This is
            useful when a single protocol file is shared by training and
            evaluation. Training uses the strict mode.
    """
    config_path = _config_path(argv)
    config = load_yaml(config_path) if config_path is not None else {}
    apply_config(parser, config, known_only=known_only)
    return parser.parse_args(argv)


def apply_config(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    known_only: bool = False,
) -> None:
    """Apply configuration values as parser defaults.

    Unknown keys are rejected to catch typos in protocol files. Explicit
    command-line flags still take precedence because ``argparse`` only uses
    these values as defaults.
    """
    valid = {action.dest for action in parser._actions}  # noqa: SLF001
    unknown = set(config) - valid
    if unknown and not known_only:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    filtered = {key: value for key, value in config.items() if key in valid}
    for action in parser._actions:  # noqa: SLF001
        if (
            action.dest in filtered
            and filtered[action.dest] is not None
            and action.required
        ):
            action.required = False
    parser.set_defaults(**filtered)


def resolve_legacy_loss_options(args: argparse.Namespace) -> None:
    """Resolve the v1.0 single-mask options into the separate modality values.

    The public v1.0 profiles used one ``mask_ratio`` and one
    ``reconstruction_weight``. They remain accepted as aliases for custom
    configurations, but the paper profile always sets the two modalities
    explicitly.
    """
    if args.mask_ratio is not None:
        if args.video_mask_ratio is None:
            args.video_mask_ratio = args.mask_ratio
        if args.audio_mask_ratio is None:
            args.audio_mask_ratio = args.mask_ratio
    if args.reconstruction_weight is not None:
        if args.video_reconstruction_weight is None:
            args.video_reconstruction_weight = args.reconstruction_weight
        if args.audio_reconstruction_weight is None:
            args.audio_reconstruction_weight = args.reconstruction_weight

    if args.video_mask_ratio is None:
        args.video_mask_ratio = 0.75
    if args.audio_mask_ratio is None:
        args.audio_mask_ratio = 0.50
    if args.video_reconstruction_weight is None:
        args.video_reconstruction_weight = 0.40
    if args.audio_reconstruction_weight is None:
        args.audio_reconstruction_weight = 0.30


def validate_train_args(args: argparse.Namespace) -> None:
    """Validate the protocol before allocating a model or decoding media."""
    fractions = (
        1.0 - args.validation_fraction - args.test_fraction,
        args.validation_fraction,
        args.test_fraction,
    )
    if min(fractions) < 0 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(
            "train/validation/test fractions must be non-negative and sum to one"
        )
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.window_duration <= 0:
        raise ValueError("window_duration must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if not 0.0 <= args.video_mask_ratio <= 1.0:
        raise ValueError("video_mask_ratio must lie in [0, 1]")
    if not 0.0 <= args.audio_mask_ratio <= 1.0:
        raise ValueError("audio_mask_ratio must lie in [0, 1]")
    if not 0.0 <= args.modality_dropout < 1.0:
        raise ValueError("modality_dropout must lie in [0, 1)")
    weights = (
        args.alignment_weight,
        args.video_reconstruction_weight,
        args.audio_reconstruction_weight,
        args.classification_weight,
    )
    if min(weights) < 0:
        raise ValueError("loss weights must be non-negative")
    if max(weights) <= 0:
        raise ValueError("at least one loss weight must be positive")
    if args.classification_weight <= 0 and args.selection_metric == "macro_f1":
        raise ValueError(
            "macro_f1 selection requires a positive classification_weight; "
            "use --selection-metric alignment_loss for self-supervised pretraining"
        )


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return a checkpoint-friendly argument mapping."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


__all__ = [
    "apply_config",
    "parse_config_args",
    "resolve_legacy_loss_options",
    "serializable_args",
    "validate_train_args",
]
