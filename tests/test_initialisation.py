"""Warm-start initialisation tests."""

from __future__ import annotations

from pathlib import Path

import torch

from savat.initialisation import _load_state, apply_init_from
from savat.model import SAVAT, SAVATConfig


def _tiny_model() -> SAVAT:
    config = SAVATConfig(
        image_size=32,
        num_frames=4,
        video_patch=4,
        tubelet=2,
        audio_time=64,
        audio_mels=16,
        audio_patch=4,
        embed_dim=32,
        depth=2,
        heads=2,
        decoder_depth=1,
    )
    torch.manual_seed(17)
    return SAVAT(config)


def _upstream_state(embed_dim: int = 32) -> dict[str, torch.Tensor]:
    """Craft an upstream CAV-MAE Sync encoder layout (fused qkv attention)."""
    mlp = 4 * embed_dim
    video: dict[str, torch.Tensor] = {}
    for layer in range(2):
        root = f"blocks_v.{layer}."
        state = {
            f"{root}norm1.weight": torch.full((embed_dim,), 1.5 + layer),
            f"{root}norm1.bias": torch.zeros(embed_dim),
            f"{root}attn.qkv.weight": torch.eye(embed_dim).repeat(3, 1),
            f"{root}attn.qkv.bias": torch.tensor(
                [10.0, 20.0, 30.0], dtype=torch.float32
            ).repeat_interleave(embed_dim),
            f"{root}attn.proj.weight": torch.eye(embed_dim) * 2,
            f"{root}attn.proj.bias": torch.zeros(embed_dim),
            f"{root}norm2.weight": torch.full((embed_dim,), 1.25),
            f"{root}norm2.bias": torch.zeros(embed_dim),
            f"{root}mlp.fc1.weight": torch.ones(mlp, embed_dim),
            f"{root}mlp.fc1.bias": torch.zeros(mlp),
            f"{root}mlp.fc2.weight": torch.ones(embed_dim, mlp),
            f"{root}mlp.fc2.bias": torch.zeros(embed_dim),
        }
        video.update(state)
    audio = {
        key.replace("blocks_v.", "blocks_a."): value for key, value in video.items()
    }
    # The shared upstream branch has no SAVAT counterpart.
    shared = {
        "blocks_u.0.norm1.weight": torch.zeros(embed_dim),
    }
    token_budget_keys = {
        "pos_embed_v": torch.zeros(1, 12, embed_dim),
        "patch_embed_v.proj.weight": torch.zeros(768, 3, 16, 16),
    }
    return video | audio | shared | token_budget_keys


def test_savat_checkpoint_warm_start(tmp_path: Path) -> None:
    model = _tiny_model()
    source_state = {name: value.clone() for name, value in model.state_dict().items()}
    source_state["video_branch.encoder.layers.0.self_attn.in_proj_weight"] = (
        torch.full_like(
            source_state["video_branch.encoder.layers.0.self_attn.in_proj_weight"],
            3.75,
        )
    )
    path = tmp_path / "prior.pt"
    torch.save({"state_dict": source_state}, path)

    before = model.state_dict()[  # type: ignore[index]
        "video_branch.encoder.layers.0.self_attn.in_proj_weight"
    ].clone()
    model = _tiny_model()
    report = apply_init_from(model, path)
    after = model.state_dict()[  # type: ignore[index]
        "video_branch.encoder.layers.0.self_attn.in_proj_weight"
    ]
    assert after[0, 0] == 3.75
    assert not torch.equal(before, after)
    assert report.transferred == len(source_state)


def test_cavmae_qkv_split_maps_to_in_proj(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "cav_mae_sync.pth"
    torch.save(_upstream_state(), path)
    report = apply_init_from(model, path)

    layer = model.video_branch.encoder.layers[0]
    assert layer.self_attn.in_proj_weight[0, 0] == 1.0  # q block
    assert layer.self_attn.in_proj_weight[32, 0] == 1.0  # k block
    assert torch.equal(
        layer.self_attn.in_proj_bias,
        torch.tensor([10.0] * 32 + [20.0] * 32 + [30.0] * 32),
    )
    assert layer.linear1.bias.sum() == 0  # transferred without nan
    assert layer.norm1.weight[0] == 1.5
    assert layer.linear1.weight.abs().sum() == 128 * 32

    # The layer per-branch transfer is per-layer; depth-2 model only sees the
    # provided two layers of each branch.
    assert report.transferred > 0
    assert "video_branch.encoder.layers.0.self_attn.in_proj_weight" in (
        report.transferred_names
    )


def test_shape_mismatch_is_skipped_not_raised(tmp_path: Path) -> None:
    model = _tiny_model()
    upstream = _upstream_state()
    upstream["blocks_v.0.attn.qkv.weight"] = torch.zeros(7, 11)
    path = tmp_path / "broken.pth"
    torch.save(upstream, path)
    report = apply_init_from(model, path)
    assert "video_branch.encoder.layers.0.self_attn.in_proj_weight" not in (
        report.transferred_names
    )
    assert report.skipped_shapes >= 1


def test_load_state_strips_module_prefix(tmp_path: Path) -> None:
    path = tmp_path / "dp.pth"
    torch.save({"module.cls": torch.zeros(3), "real": torch.ones(2)}, path)
    state = _load_state(path)
    assert set(state) == {"cls", "real"}
