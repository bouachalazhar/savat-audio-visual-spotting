# SAVAT: Sports Audio-Visual Alignment Transformer

Reference code for the audio-visual representation learning method described in:

> L. Bouacha, L. Wendling, C. Kurtz. *SAVAT: Sports Audio-Visual Alignment
> Transformer for Action Spotting.* Advanced Concepts for Intelligent Vision
> Systems (ACIVS 2026), Lecture Notes in Computer Science, in press.

SAVAT learns a joint audio-visual representation from sports video. Two
transformer encoders process video tubelets and log-Mel patches, a contrastive
objective aligns their embeddings, and per-modality reliability scores
down-weight degraded pairs. The representation is then transferred to an
action spotting task.

## Method

1. **Dual encoding** (`savat.model`): a 3D tubelet embedding and a spectrogram
   patch embedding feed two transformer encoders. Each branch exposes a class
   token, a unimodal head and a reconstruction head.
2. **Masked reconstruction** (`savat.model`): a configurable fraction of
   tokens is masked; a lightweight decoder reconstructs the masked patches.
   Masking is applied independently to each modality, and the reconstruction
   error is weighted by each modality's reliability.
3. **Reliability indicators** (`savat.reliability`): audio presence, temporal
   stability, clipping and tonality; video sharpness, exposure and temporal
   stability. They are aggregated with a geometric mean, so one collapsed
   indicator lowers the whole score. Min-max bounds and missing-value medians
   are fitted on the training partition only and stored in the checkpoint.
4. **Reliability-weighted InfoNCE** (`savat.alignment`): the symmetric
   contrastive loss is weighted by the product of the two reliability scores.
   With labels, negatives can be restricted to the same normalised class.
5. **Fusion** (`savat.model`): the joint head concatenates reliability-weighted
   video and audio class tokens, which is the feature-level fusion used by the
   default evaluation. `savat.fusion` additionally provides arithmetic,
   geometric, hard-selection and unimodal decision-level ablations.
6. **Modality dropout** (`savat.model`): during training, one branch can be
   replaced by a learned null embedding, which prepares the transfer stage for
   missing audio or video.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/bouachalazhar/savat-audio-visual-spotting.git
cd savat-audio-visual-spotting
uv sync --extra dev
```

## Data layout

Each split is a directory of labelled clips:

```text
sportvista/
  football/clip_001.mp4
  football/clip_002.mp4
  tennis/clip_003.mp4
  ...
```

The label is the first directory below the data root. The group used by the
grouped split is the relative parent directory when clips are nested:

```text
sportvista/
  football/match_001/clip_001.mp4
  football/match_001/clip_002.mp4
  football/match_002/clip_003.mp4
```

For a flat label directory, the group falls back to the file stem with a
trailing index removed, so `match_001` and `match_002` both map to `match`.
Nested directories are the unambiguous option and are recommended for a
match-level protocol.

The default split is 70/15/15, stratified by label and grouped by source. The
manuscript uses seeds 17, 29 and 43; the split rejects an empty partition
instead of silently leaking a group.

Video and audio are decoded from the same temporal window. By default the
window is five seconds, centred on the clip. `--window-duration` changes its
length and `--window-start` fixes its start for an ablation. The audio branch
uses 320 log-Mel frames, matching the model's default `audio_time`; the video
branch uniformly samples 16 frames from the same window.

## Training

Pretraining on an eleven-class corpus:

```bash
uv run savat-train --config configs/pretrain_sportvista.yaml
```

Transfer to a three-class SoccerNet setup:

```bash
uv run savat-train --config configs/transfer_soccernet.yaml
```

Every YAML key maps to a command-line option, and explicit flags override the
file. The default objective is

$$
\mathcal{L} = \mathcal{L}_{\text{joint}}
+ \tfrac{1}{2}\left(\mathcal{L}_{v} + \mathcal{L}_{a}\right)
+ \lambda_{\text{align}} \mathcal{L}_{\text{InfoNCE}}
+ \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} .
$$

The contrastive term uses weights $w_i = r_v(i)\,r_a(i)$, and the masked
reconstruction term $\mathcal{L}_{\text{rec}}$ is the reliability-weighted
mean of the per-sample video and audio patch errors. Each reliability score is
the geometric mean of its indicators:

$$
r_v = \left(\prod_{k=1}^{p} v_k\right)^{1/p},
\qquad
r_a = \left(\prod_{k=1}^{q} a_k\right)^{1/q}.
$$

Checkpoints are selected on validation macro-F1.

## Evaluation

```bash
uv run savat-evaluate \
  --data-root data/sportvista \
  --checkpoint runs/savat_pretrain/best.pt \
  --fusion joint \
  --output metrics.json
```

`joint` is the default and evaluates the reliability-weighted feature fusion
used for training. `geometric`, `arithmetic`, `hard`, `video_only` and
`audio_only` are decision-level ablations. The report contains Top-1,
macro-F1, per-class average precision, mean average precision and the confusion
matrix. The checkpoint's class order and reliability normaliser are reused, so
evaluation data are never used to refit normalisation.

## Robustness

```bash
uv run savat-robustness \
  --data-root data/sportvista \
  --checkpoint runs/savat_pretrain/best.pt \
  --noise-snr 10 \
  --blur-sigma 1.0 \
  --offsets=-2,-1,1,2 \
  --seeds 17,29,43 \
  --output robustness.json
```

The defaults follow the manuscript protocol: 10 dB pink noise, Gaussian blur
with sigma 1.0, signed audio shifts of -2, -1, +1 and +2 seconds, and seeds 17,
29 and 43. Each condition is evaluated for every seed and the report contains
both the runs and their mean/standard deviation. Reliability is recomputed on
the corrupted signals before fusion. Noise and blur alter one modality only;
the shift leaves both signals intact but breaks their correspondence.

## Scope and differences from the article

This repository is a compact, runnable reimplementation of the method. The
following points delimit what it reproduces:

- The reported experiments initialise the encoders from CAV-MAE Sync. Those
  weights are not redistributed here, so `savat-train` starts from random
  initialisation by default.
- The public joint head applies reliability weights to the two class tokens
  before concatenation. It is a feature-level analogue of the article's
  weighted fusion, not a line-by-line reproduction of its internal
  projection blocks.
- The reported SoccerNet number follows the CALF protocol (temporal targets,
  non-maximum suppression and tolerance-based avg-mAP). This repository
  provides clip-level transfer training and classification average precision,
  not the official CALF evaluation script. The two metrics are not
  interchangeable.
- No SportVISTA clip, SoccerNet clip or trained checkpoint is distributed.
  Broadcast and platform rights do not allow redistribution.
- Reliability indicators are technical descriptors. They are not calibrated
  probabilities and do not measure semantic synchronisation.

These differences mean that the numbers in the article cannot be reproduced
bit-for-bit from this repository alone. The architecture, objectives, fusion
rules and stress-test protocol are provided so that the method can be
reimplemented and extended on data the user is allowed to process.

## Repository layout

```text
src/savat/
  data.py         media decoding, dataset, grouped split
  model.py        SAVAT encoders, masked reconstruction, heads
  reliability.py  indicators and train-set normalisation
  fusion.py       reliability-weighted fusion strategies
  alignment.py    symmetric weighted InfoNCE
  train.py        training loop and objective
  evaluate.py     Top-1, macro-F1, average precision
  robustness.py   pink noise, blur, desynchronisation
configs/          pretraining and transfer configurations
tests/            unit and smoke tests
```

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

The tests build synthetic audio-visual clips with PyAV, so no external data is
needed. CI runs lint, format, unit tests and a wheel build.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{bouacha2026savat,
  title     = {{SAVAT}: Sports Audio-Visual Alignment Transformer for Action Spotting},
  author    = {Bouacha, Lazhar and Wendling, Laurent and Kurtz, Camille},
  booktitle = {Proceedings of Advanced Concepts for Intelligent Vision Systems (ACIVS)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {In press}
}
```

The software archive DOI is listed in `CITATION.cff` once deposited on Zenodo.
