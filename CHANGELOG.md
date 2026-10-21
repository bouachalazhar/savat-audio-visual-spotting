# Changelog

## 1.1.0 — 2026-10-21

Protocol-fidelity release.

- Made self-supervised pretraining the default: separate video/audio mask
  ratios (`0.75/0.50`) and loss weights (`0.40/0.30/0.30`), with no supervised
  cross-entropy term.
- Decoded one common 4 s audio-visual window with 16 video frames and 256
  log-Mel frames; older checkpoints keep their own temporal metadata.
- Added class-balanced batches and same-class contrastive negatives that exclude
  the same source group.
- Added a frozen-encoder `savat-linear-probe` command and checkpoint persistence
  of the reliability normaliser, class order and protocol metadata.
- Added grouped train/validation/test evaluation, including the held-out test
  report, and explicit YAML profiles.
- Kept `configs/transfer_soccernet.yaml` as a generic supervised baseline; it
  is not CALF and does not claim the article's `lambda_m = 0.20` branch.
- Updated the README, provenance notes and configuration documentation.

## 1.0.2 — 2026-10-20

- Synchronised package metadata and release packaging.

## 1.0.1 — 2026-10-20

- Aligned audio-visual windows and reliability weighting.

## 1.0.0 — 2026-10-19

- Initial public release.
