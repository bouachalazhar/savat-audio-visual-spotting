"""Tests for the SAVAT protocol configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from savat.config import (
    apply_config,
    parse_config_args,
    resolve_legacy_loss_options,
    validate_train_args,
)
from savat.train import build_parser, parse_args
from savat.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("epochs: 1\nepochs: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_yaml(path)


def test_pretraining_profile_matches_the_manuscript_objective() -> None:
    args = parse_args(
        [
            "--config",
            str(ROOT / "configs" / "pretrain_sportvista.yaml"),
            "--data-root",
            str(ROOT),
        ]
    )
    assert args.video_mask_ratio == pytest.approx(0.75)
    assert args.audio_mask_ratio == pytest.approx(0.50)
    assert args.alignment_weight == pytest.approx(0.30)
    assert args.video_reconstruction_weight == pytest.approx(0.40)
    assert args.audio_reconstruction_weight == pytest.approx(0.30)
    assert args.classification_weight == 0.0
    assert args.modality_dropout == 0.0
    assert args.window_duration == pytest.approx(4.0)
    assert args.audio_time == 256
    assert args.same_class_negatives is True
    assert args.balanced_batches is True
    assert args.selection_metric == "alignment_loss"


def test_transfer_profile_is_generic_and_not_calf() -> None:
    args = parse_args(
        [
            "--config",
            str(ROOT / "configs" / "transfer_soccernet.yaml"),
            "--data-root",
            str(ROOT),
        ]
    )
    assert args.classification_weight > 0
    assert args.selection_metric == "macro_f1"
    assert args.modality_dropout == pytest.approx(0.25)
    # The public profile does not claim the CALF lambda_m branch.
    assert args.alignment_weight == pytest.approx(0.05)


def test_cli_flags_override_yaml(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "epochs: 3\nbatch_size: 2\nvideo_mask_ratio: 0.6\nalignment_weight: 0.2\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--config",
            str(path),
            "--data-root",
            str(tmp_path),
            "--epochs",
            "9",
        ]
    )
    assert args.epochs == 9
    assert args.batch_size == 2
    assert args.video_mask_ratio == pytest.approx(0.6)
    assert args.alignment_weight == pytest.approx(0.2)


def test_yaml_can_supply_values_required_by_the_command(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        "data_root: /tmp\nepochs: 2\n",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(path)])
    assert args.data_root == Path("/tmp")
    assert args.epochs == 2


def test_unknown_keys_are_rejected_but_known_only_ignores_other_commands(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text("epochs: 3\ntypo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration keys"):
        parse_args(["--config", str(path), "--data-root", str(tmp_path)])

    parser = build_parser()
    ignored = parse_config_args(
        parser, ["--config", str(path), "--data-root", str(tmp_path)], known_only=True
    )
    assert ignored.epochs == 3


def test_legacy_single_mask_and_reconstruction_options_are_resolved() -> None:
    args = parse_args(["--data-root", "/tmp", "--mask-ratio", "0.6"])
    assert args.video_mask_ratio == pytest.approx(0.6)
    assert args.audio_mask_ratio == pytest.approx(0.6)

    args = parse_args(["--data-root", "/tmp", "--reconstruction-weight", "0.2"])
    assert args.video_reconstruction_weight == pytest.approx(0.2)
    assert args.audio_reconstruction_weight == pytest.approx(0.2)

    args = parse_args(
        [
            "--data-root",
            "/tmp",
            "--reconstruction-weight",
            "0.2",
            "--video-reconstruction-weight",
            "0.4",
            "--audio-reconstruction-weight",
            "0.3",
        ]
    )
    assert args.video_reconstruction_weight == pytest.approx(0.4)
    assert args.audio_reconstruction_weight == pytest.approx(0.3)


def test_macro_f1_selection_requires_supervised_classification(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="macro_f1 selection"):
        parse_args(
            [
                "--data-root",
                str(tmp_path),
                "--classification-weight",
                "0",
                "--selection-metric",
                "macro_f1",
            ]
        )


def test_apply_config_does_not_mutate_the_input_mapping() -> None:
    parser = build_parser()
    config = {"epochs": 4}
    apply_config(parser, config)
    assert config == {"epochs": 4}
    assert parser.parse_args(["--data-root", "/tmp"]).epochs == 4


def test_validate_train_args_rejects_invalid_masks() -> None:
    parser = build_parser()
    args = parse_config_args(parser, ["--data-root", "/tmp"])
    resolve_legacy_loss_options(args)
    args.video_mask_ratio = 1.1
    with pytest.raises(ValueError, match="video_mask_ratio"):
        validate_train_args(args)
