"""SAVAT: Sports Audio-Visual Alignment Transformer.

The model encodes video tubelets and log-Mel patches with two transformer
encoders, aligns their class tokens with a contrastive objective, and
reconstructs masked patches. A joint head and two unimodal heads expose the
logits used by reliability-aware fusion.

This is a compact reimplementation of the method described in the article. It
does not ship the CAV-MAE Sync initialisation used for the reported
experiments; see the README for the scope of the reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from savat.fusion import geometric_fusion, normalize_reliability


@dataclass(frozen=True)
class SAVATConfig:
    """Architecture hyper-parameters."""

    image_size: int = 112
    num_frames: int = 16
    video_patch: int = 16
    tubelet: int = 4
    audio_mels: int = 64
    audio_time: int = 256
    audio_patch: int = 16
    embed_dim: int = 256
    depth: int = 4
    heads: int = 8
    decoder_depth: int = 2
    dropout: float = 0.1
    num_classes: int = 11


@dataclass
class SAVATOutput:
    """Outputs of one forward pass."""

    joint_logits: torch.Tensor
    video_logits: torch.Tensor
    audio_logits: torch.Tensor
    video_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    video_reconstruction: torch.Tensor | None = None
    audio_reconstruction: torch.Tensor | None = None
    video_mask: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    video_target: torch.Tensor | None = None
    audio_target: torch.Tensor | None = None


class MaskedBranch(nn.Module):
    """Patch tokenizer, transformer encoder, decoder and reconstruction head."""

    def __init__(
        self,
        tokenizer: PatchEmbed3D | PatchEmbed2D,
        patch_dim: int,
        token_count: int,
        *,
        embed_dim: int,
        depth: int,
        heads: int,
        decoder_depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.patch_dim = patch_dim
        self.token_count = token_count

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruction_head = nn.Linear(embed_dim, patch_dim)

    def tokenize(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embedded tokens and raw patch targets, both ``(B, N, D)``."""
        embedded = self.tokenizer(inputs)
        tokens = embedded.flatten(2).transpose(1, 2)
        target = self.tokenizer.raw_patches(inputs)
        return tokens, target

    def _mask(self, batch: int, ratio: float, device: torch.device) -> torch.Tensor:
        count = max(1, int(round(self.token_count * ratio)))
        scores = torch.rand(batch, self.token_count, device=device)
        indices = scores.topk(count, dim=1).indices
        mask = torch.zeros(batch, self.token_count, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)
        return mask

    def forward(
        self, inputs: torch.Tensor, mask_ratio: float = 0.0
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return the class token, reconstruction, mask and target patches."""
        tokens, target = self.tokenize(inputs)
        batch = tokens.shape[0]

        if mask_ratio <= 0.0:
            sequence = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
            encoded = self.encoder(sequence + self.position)
            return self.norm(encoded[:, 0]), None, None, None

        mask = self._mask(batch, mask_ratio, tokens.device)
        visible = ~mask
        keep = int(visible.sum(dim=1)[0])
        visible_tokens = tokens[visible].reshape(batch, keep, -1)
        positions = self.position[:, 1:].expand(batch, -1, -1)
        visible_position = positions[visible].reshape(
            batch, keep, self.position.shape[-1]
        )

        sequence = torch.cat(
            [self.cls_token.expand(batch, -1, -1), visible_tokens], dim=1
        )
        position = torch.cat(
            [self.position[:, :1].expand(batch, -1, -1), visible_position], dim=1
        )
        encoded = self.encoder(sequence + position)
        class_token = self.norm(encoded[:, 0])

        full = self.mask_token.expand(batch, self.token_count, -1).clone()
        full = full.masked_scatter(
            visible.unsqueeze(-1).expand_as(full), encoded[:, 1:]
        )
        decoded = self.decoder(full + self.position[:, 1:])
        reconstruction = self.reconstruction_head(self.norm(decoded))
        return class_token, reconstruction, mask, target


class PatchEmbed3D(nn.Module):
    """Tubelet embedding for a ``(B, C, T, H, W)`` video."""

    def __init__(self, patch: int, tubelet: int, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.projection(video)

    def raw_patches(self, video: torch.Tensor) -> torch.Tensor:
        """Return non-overlapping patches as ``(B, N, C * tubelet * p * p)``."""
        patch = self.projection.kernel_size[1]
        tubelet = self.projection.kernel_size[0]
        batch, channels = video.shape[:2]
        patches = (
            video.unfold(2, tubelet, tubelet)
            .unfold(3, patch, patch)
            .unfold(4, patch, patch)
        )
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7)
        return patches.reshape(batch, -1, channels * tubelet * patch * patch)


class PatchEmbed2D(nn.Module):
    """Patch embedding for a ``(B, 1, F, T)`` spectrogram."""

    def __init__(self, patch: int, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.projection(spectrogram)

    def raw_patches(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Return non-overlapping patches as ``(B, N, p * p)``."""
        patch = self.projection.kernel_size[0]
        batch = spectrogram.shape[0]
        patches = spectrogram.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(batch, -1, patch * patch)


class SAVAT(nn.Module):
    """Two-encoder audio-visual transformer with masked reconstruction."""

    def __init__(self, config: SAVATConfig | None = None) -> None:
        super().__init__()
        self.config = config or SAVATConfig()
        cfg = self.config
        if cfg.num_frames % cfg.tubelet != 0:
            raise ValueError("num_frames must be divisible by tubelet")
        if cfg.image_size % cfg.video_patch != 0:
            raise ValueError("image_size must be divisible by video_patch")
        if cfg.audio_mels % cfg.audio_patch != 0:
            raise ValueError("audio_mels must be divisible by audio_patch")
        if cfg.audio_time % cfg.audio_patch != 0:
            raise ValueError("audio_time must be divisible by audio_patch")

        video_tokens = (cfg.num_frames // cfg.tubelet) * (
            cfg.image_size // cfg.video_patch
        ) ** 2
        audio_tokens = (cfg.audio_mels // cfg.audio_patch) * (
            cfg.audio_time // cfg.audio_patch
        )
        video_patch_dim = 3 * cfg.tubelet * cfg.video_patch * cfg.video_patch
        audio_patch_dim = cfg.audio_patch * cfg.audio_patch

        self.video_branch = MaskedBranch(
            PatchEmbed3D(cfg.video_patch, cfg.tubelet, cfg.embed_dim),
            video_patch_dim,
            video_tokens,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            decoder_depth=cfg.decoder_depth,
            dropout=cfg.dropout,
        )
        self.audio_branch = MaskedBranch(
            PatchEmbed2D(cfg.audio_patch, cfg.embed_dim),
            audio_patch_dim,
            audio_tokens,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            decoder_depth=cfg.decoder_depth,
            dropout=cfg.dropout,
        )

        self.video_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.audio_head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.joint_head = nn.Linear(2 * cfg.embed_dim, cfg.num_classes)
        self.video_null = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        self.audio_null = nn.Parameter(torch.zeros(1, cfg.embed_dim))

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        *,
        video_mask_ratio: float = 0.0,
        audio_mask_ratio: float = 0.0,
        modality_dropout: float = 0.0,
        reliability: torch.Tensor | None = None,
    ) -> SAVATOutput:
        """Run both branches and return logits, embeddings and reconstructions.

        When ``reliability`` is provided, the joint head receives reliability-
        weighted audio and video features. This is the feature-level fusion of
        the article; the decision-level strategies in :mod:`savat.fusion`
        remain available as ablations.
        """
        video_cls, video_recon, video_mask, video_target = self.video_branch(
            video, video_mask_ratio
        )
        audio_cls, audio_recon, audio_mask, audio_target = self.audio_branch(
            audio, audio_mask_ratio
        )

        if self.training and modality_dropout > 0.0:
            video_cls, audio_cls = self._drop_modalities(
                video_cls, audio_cls, modality_dropout
            )

        joint_video = video_cls
        joint_audio = audio_cls
        if reliability is not None:
            if reliability.ndim != 2 or reliability.shape[1] != 2:
                raise ValueError("reliability must have shape (B, 2)")
            if reliability.shape[0] != video_cls.shape[0]:
                raise ValueError("reliability batch size does not match the input")
            weights = reliability.to(
                device=video_cls.device, dtype=video_cls.dtype
            ).clamp(0.0, 1.0)
            joint_video = video_cls * weights[:, 0:1]
            joint_audio = audio_cls * weights[:, 1:2]

        joint = self.joint_head(torch.cat([joint_video, joint_audio], dim=1))
        return SAVATOutput(
            joint_logits=joint,
            video_logits=self.video_head(video_cls),
            audio_logits=self.audio_head(audio_cls),
            video_embedding=video_cls,
            audio_embedding=audio_cls,
            video_reconstruction=video_recon,
            audio_reconstruction=audio_recon,
            video_mask=video_mask,
            audio_mask=audio_mask,
            video_target=video_target,
            audio_target=audio_target,
        )

    def _drop_modalities(
        self, video: torch.Tensor, audio: torch.Tensor, probability: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= probability < 1.0:
            raise ValueError("modality_dropout must lie in [0, 1)")
        batch = video.shape[0]
        device = video.device
        # Independent Bernoulli masks, with at least one modality kept.
        drop_video = torch.rand(batch, 1, device=device) < probability
        drop_audio = torch.rand(batch, 1, device=device) < probability
        both = drop_video & drop_audio
        drop_audio = torch.where(both, ~drop_audio, drop_audio)
        video = torch.where(drop_video, self.video_null.expand(batch, -1), video)
        audio = torch.where(drop_audio, self.audio_null.expand(batch, -1), audio)
        return video, audio

    @torch.no_grad()
    def fused_probabilities(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        reliability: torch.Tensor,
        *,
        strategy: str = "geometric",
    ) -> torch.Tensor:
        """Return reliability-weighted fused probabilities at inference."""
        output = self.forward(video, audio, reliability=reliability)
        weights = normalize_reliability(reliability)
        probabilities = {
            "video": torch.softmax(output.video_logits, dim=1),
            "audio": torch.softmax(output.audio_logits, dim=1),
        }
        return geometric_fusion(probabilities, weights, strategy=strategy)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


__all__ = [
    "MaskedBranch",
    "SAVAT",
    "SAVATConfig",
    "SAVATOutput",
    "count_parameters",
]
