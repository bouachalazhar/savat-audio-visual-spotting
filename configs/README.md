# Protocol configurations

Each YAML file is a complete, readable profile for the general implementation.
`--config` supplies defaults and an explicit command-line flag overrides them.
Training and linear probing reject unknown keys; evaluation and robustness use
known-only mode so one shared file can contain options for several commands.

| File | Purpose |
| --- | --- |
| `pretrain_sportvista.yaml` | Self-supervised SportVISTA protocol: 4 s window, 16 video frames, 256 log-Mel frames, masks `0.75/0.50`, loss weights `0.40/0.30/0.30`, no supervised CE, balanced batches, group-aware same-class negatives, selection on alignment loss. |
| `transfer_soccernet.yaml` | Generic supervised clip-level adaptation for SoccerNet-style data. It enables cross-entropy, selects on macro-F1 and uses modality dropout. It is **not** the CALF protocol. |
| `linear_probe_sportvista.yaml` | Frozen-encoder probe: reliability-weighted audio-video features, `LayerNorm + Linear` head, balanced batches and class weights, selection on validation macro-F1. |

## Manuscript versus public code

| Item | Manuscript claim | Public profile / implementation |
| --- | --- | --- |
| Pretraining objective | Masked reconstruction for both modalities plus reliability-weighted audio-visual alignment | `0.40 L_rec_video + 0.30 L_rec_audio + 0.30 L_align`; `classification_weight: 0.0` |
| Video mask ratio | `0.75` | `video_mask_ratio: 0.75` |
| Audio mask ratio | `0.50` | `audio_mask_ratio: 0.50` |
| Temporal window | 4 s for both streams | `window_duration: 4.0` |
| Video sampling | 16 frames | `num_frames: 16` |
| Audio representation | 256 log-Mel frames | `audio_time: 256` |
| Batch sampling | Class-balanced | `balanced_batches: true` with `BalancedClassSampler` |
| Contrastive negatives | Same class, excluding the same source group | `same_class_negatives: true`; the sampler receives group IDs |
| Pretraining selection | Alignment objective, not a random classifier | `selection_metric: alignment_loss` |
| Linear probing | Frozen encoder, linear classifier | `savat-linear-probe`, `LayerNorm + Linear` |
| Transfer protocol | CALF with temporal targets and official evaluation | **Not implemented.** `transfer_soccernet.yaml` is a generic clip-level supervised baseline. |
| Initialisation | CAV-MAE Sync weights | **Not distributed.** Training starts randomly unless the user provides compatible weights. |
| `lambda_m = 0.20` modality branch | CALF-specific claim | **Not claimed.** The transfer profile has a generic `modality_dropout: 0.25` regulariser only. |

### What the public profiles do not establish

- No numerical result in the article is reproduced bit-for-bit. The private
  corpus, splits, CAV-MAE Sync initialisation and CALF evaluation server are
  not part of this repository.
- The 70/15/15 split default is an implementation default. Change
  `validation_fraction` and `test_fraction` to match the split used in an
  experiment; groups are always kept in one partition.
- The probe learning rate, weight decay, epoch count and batch size are
  implementation choices, not reported manuscript constants.
- Reliability scores are technical indicators, not calibrated probabilities.

## Pretraining

```bash
uv run savat-train --config configs/pretrain_sportvista.yaml
```

The profile contains no `classification_weight` above zero. The classification
heads are therefore untrained and `history.json` reports `test_alignment`, not a
classification score. Use the linear probe for representation quality.

## Generic transfer

```bash
uv run savat-train --config configs/transfer_soccernet.yaml
```

This run enables the supervised term and selects on macro-F1. It is useful for
testing adaptation on a labelled dataset, but it must not be presented as CALF:
the loss, temporal targets, non-maximum suppression and tolerance-based
average mAP of the official protocol are absent.

## Frozen-encoder probe

```bash
uv run savat-linear-probe --config configs/linear_probe_sportvista.yaml
```

The probe writes `probe.pt` and `metrics.json`. The encoder is never updated.
The source checkpoint supplies the class order, architecture, reliability
normaliser, window duration and log-Mel length.

## Aliases and precedence

The legacy single-value options are still accepted for compatibility:

- `--mask-ratio` sets both `video_mask_ratio` and `audio_mask_ratio`;
- `--reconstruction-weight` sets both `video_reconstruction_weight` and
  `audio_reconstruction_weight`.

When a separate value is provided, it takes priority over the alias. In YAML,
prefer the separate keys so the protocol remains explicit.

The order of precedence is:

1. explicit command-line flag;
2. YAML value;
3. command default.

## Checkpoint compatibility

New checkpoints store `window_duration`, `time_frames`, the reliability
normaliser, the class order and the architecture configuration. Evaluation,
robustness and linear probing read those values from the checkpoint rather than
guessing new defaults. A checkpoint trained with the earlier 5 s / 320-frame
configuration therefore keeps its own temporal metadata; it is not silently
reinterpreted with the 4 s / 256-frame defaults. If a legacy checkpoint lacks
the window metadata entirely, evaluation and probing use the historical 5 s
fallback rather than assuming the new 4 s default.
