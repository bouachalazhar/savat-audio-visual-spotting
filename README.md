# SAVAT: Sports Audio-Visual Alignment Transformer

Reference code for the audio-visual representation learning method described in:

> L. Bouacha, L. Wendling, C. Kurtz. *SAVAT: Sports Audio-Visual Alignment
> Transformer for Action Spotting.* Advanced Concepts for Intelligent Vision
> Systems (ACIVS 2026), Lecture Notes in Computer Science, in press.

SAVAT learns a joint audio-visual representation from sports video. Two
transformer encoders process video tubelets and log-Mel patches, a
reliability-weighted contrastive objective aligns their class tokens, and
masked reconstruction regularises each modality. The representation can then be
transferred to action spotting or evaluated with a frozen-encoder linear probe.

## Method

1. **Common audio-visual window** (`savat.data`): a clip is decoded once over a
   centred 4 s window. The video branch uniformly samples 16 frames from that
   window and the audio branch produces 256 log-Mel frames from the same
   interval. `--window-start` fixes the window for an ablation.
2. **Dual encoding** (`savat.model`): a 3D tubelet embedding and a spectrogram
   patch embedding feed two transformer encoders. Each branch exposes a class
   token, a unimodal classification head and a reconstruction head.
3. **Masked reconstruction**: 75 % of the video tokens and 50 % of the audio
   tokens are masked by default. A lightweight decoder reconstructs the masked
   patches, and the per-sample error is weighted by the reliability of the
   corresponding modality.
4. **Reliability indicators** (`savat.reliability`): audio presence, temporal
   stability, clipping and tonality; video sharpness, exposure and temporal
   stability. They are aggregated with a geometric mean. Min-max bounds and
   missing-value medians are fitted on the training partition only and stored
   in the checkpoint.
5. **Reliability-weighted InfoNCE** (`savat.alignment`): the symmetric
   contrastive loss uses $w_i = r_v(i)\,r_a(i)$ as sample weights. With
   `same_class_negatives: true`, negatives must share the anchor's label and
   must not come from the same source group; when no such negative exists in a
   batch, the implementation falls back to cross-group clips and ultimately to
   all other clips rather than producing an empty mask.
6. **Feature-level fusion** (`savat.model`): the joint head concatenates the
   reliability-weighted video and audio class tokens. `savat.fusion` provides
   arithmetic, geometric, hard-selection and unimodal decision-level ablations;
   those are not the default training path.
7. **Modality dropout** (`savat.model`): one branch can be replaced by a learned
   null embedding. The paper pretraining profile disables it; the generic
   transfer profile uses it as a robustness regulariser.

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
SportVISTA/
  football/match_001/clip_001.mp4
  football/match_001/clip_002.mp4
  football/match_002/clip_003.mp4
  tennis/match_003/clip_004.mp4
  ...
```

The label is the first directory below the data root. The source group is the
relative parent directory, so every clip from one match stays in one partition.
For a flat label directory, the group falls back to the file stem with a
trailing index removed (`match_001` and `match_002` both map to `match`).
Nested directories are the unambiguous option and are recommended for a
match-level protocol.

Directories without any media file are ignored, which lets an output directory
live next to the data without becoming a class.

The default split is 70/15/15, stratified by label and grouped by source. The
fractions are configurable. The split rejects an empty partition instead of
silently leaking a group. The test partition is evaluated after training and
its metrics are written to `history.json`.

Video and audio are decoded from the same temporal window. The current
implementation uses a 4 s window and 256 log-Mel frames. Older checkpoints keep
their own `window_duration` and `time_frames` metadata and remain loadable at
evaluation and probing time; a checkpoint that predates this metadata uses the
historical 5 s window fallback.

## Pretraining (paper objective)

```bash
uv run savat-train --config configs/pretrain_sportvista.yaml
```

The paper pretraining objective has no supervised cross-entropy term:

$$
\mathcal{L}
= 0.40\,\mathcal{L}^{v}_{\mathrm{rec}}
+ 0.30\,\mathcal{L}^{a}_{\mathrm{rec}}
+ 0.30\,\mathcal{L}_{\mathrm{align}}.
$$

The contrastive term uses weights $w_i = r_v(i)\,r_a(i)$. The reconstruction
term is the reliability-weighted mean of the per-sample video and audio patch
errors. Each reliability score is the geometric mean of its indicators:

$$
r_v = \left(\prod_{k=1}^{p} v_k\right)^{1/p},
\qquad
r_a = \left(\prod_{k=1}^{q} a_k\right)^{1/q}.
$$

The profile uses AdamW, batch size 8, learning rate $10^{-4}$, 75 epochs, a
five-epoch warm-up and cosine decay. Validation selection uses the alignment
loss, not a randomly initialised classification head. The balanced sampler
constructs each batch from a small set of classes with at least two examples
per selected class, and same-class, group-aware negatives are enabled.

The test split is evaluated with the best checkpoint after training. For the
self-supervised profile, `history.json` records `test_alignment` (loss and
paired clip retrieval accuracy); no classification metric is reported because
the classification heads were not trained.

## Generic supervised transfer

```bash
uv run savat-train --config configs/transfer_soccernet.yaml
```

This profile is a **generic clip-level transfer baseline**, not the article's
CALF protocol. It enables the supervised term:

$$
\mathcal{L}
= \lambda_v \mathcal{L}^{v}_{\mathrm{rec}}
+ \lambda_a \mathcal{L}^{a}_{\mathrm{rec}}
+ \lambda_{\mathrm{align}} \mathcal{L}_{\mathrm{align}}
+ \lambda_{\mathrm{cls}}
\left(
\mathcal{L}_{\mathrm{joint}}
+ \lambda_{\mathrm{uni}}
\left(\mathcal{L}_{v} + \mathcal{L}_{a}\right)
\right),
$$

where the joint and unimodal logits are trained with a standard clip-level
cross-entropy.
The transfer profile uses $\lambda_{\mathrm{cls}}=1$, $\lambda_{\mathrm{uni}}=0.5$,
$\lambda_{\mathrm{align}}=0.05$, $\lambda_v=\lambda_a=0.05$ and selects on
validation macro-F1. It is provided to adapt a checkpoint to a new labelled
dataset; it does not implement CALF temporal targets, SoccerNet's official
evaluation server or the article's $\lambda_m=0.20$ modality-dropout branch.

## Frozen-encoder linear probe

The classification heads are not trained during self-supervised pretraining.
Use the dedicated probe rather than `savat-evaluate` if the goal is to measure
the learned representation:

```bash
uv run savat-linear-probe \
  --config configs/linear_probe_sportvista.yaml
```

The encoder is frozen and kept in evaluation mode. Features are the
reliability-weighted concatenation of the two class tokens, and only a
`LayerNorm + Linear` head is trained. The command writes `probe.pt` and
`metrics.json` with validation and test Top-1, macro-F1, mean average precision
and confusion matrices. The probe uses the training reliability normaliser
stored in the source checkpoint.

## Evaluation

```bash
uv run savat-evaluate \
  --data-root data/sportvista \
  --checkpoint runs/savat_pretrain/best.pt \
  --fusion joint \
  --output metrics.json
```

`joint` is the default and evaluates the reliability-weighted feature fusion.
`geometric`, `arithmetic`, `hard`, `video_only` and `audio_only` are
decision-level ablations. The report contains Top-1, macro-F1, per-class
average precision, mean average precision and the confusion matrix. The
checkpoint's class order and reliability normaliser are reused, so evaluation
data are never used to refit normalisation.

For a self-supervised checkpoint, the classification numbers from
`savat-evaluate` are not meaningful because the heads were never trained; use
`savat-linear-probe`. `savat-evaluate` is appropriate for a checkpoint produced
by a supervised transfer run.

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

The defaults follow the manuscript stress-test protocol: 10 dB pink noise,
Gaussian blur with sigma 1.0, signed audio shifts of -2, -1, +1 and +2 seconds,
and seeds 17, 29 and 43. Each condition is evaluated for every seed and the
report contains both the runs and their mean and standard deviation.
Reliability is recomputed on the corrupted signals before fusion. Noise and
blur alter one modality only; the shift leaves both signals intact but breaks
their correspondence.

## Configuration

The YAML files under `configs/` are complete, readable profiles. Every key maps
to a command-line option, and an explicit flag overrides the YAML value.
Unknown keys are rejected by the training and probe commands; shared profiles
are read in known-only mode by evaluation and robustness. See
[`configs/README.md`](configs/README.md) for the profile table and the legacy
aliases.

## Scope and differences from the article

This repository is a compact, runnable reimplementation of the method. The
following points delimit what it reproduces:

- The reported experiments initialise the encoders from CAV-MAE Sync. Those
  weights are not redistributed here, so `savat-train` starts from random
  initialisation by default.
- The public joint head applies reliability weights to the two class tokens
  before concatenation. It is a feature-level analogue of the article's
  weighted fusion, not a line-by-line reproduction of its internal projection
  blocks.
- The reported SoccerNet number follows the CALF protocol (temporal targets,
  non-maximum suppression and tolerance-based avg-mAP). This repository
  provides a generic clip-level transfer baseline and classification average
  precision, not the official CALF evaluation script. The two metrics are not
  interchangeable.
- No SportVISTA clip, SoccerNet clip or trained checkpoint is distributed.
  Broadcast and platform rights do not allow redistribution.
- Reliability indicators are technical descriptors. They are not calibrated
  probabilities and do not measure semantic synchronisation.
- The pretraining, transfer and probe hyper-parameters are declared explicitly
  in YAML, but tuning choices such as the probe learning rate are
  implementation choices, not constants reported in the manuscript.

These differences mean that the numbers in the article cannot be reproduced
bit-for-bit from this repository alone. The architecture, objectives, fusion
rules and stress-test protocol are provided so that the method can be
reimplemented and extended on data the user is allowed to process.

## Repository layout

```text
src/savat/
  config.py        YAML loading and command-line precedence
  data.py          media decoding, dataset, grouped split
  model.py         SAVAT encoders, masked reconstruction, heads
  reliability.py   indicators and train-set normalisation
  fusion.py        reliability-weighted fusion strategies
  alignment.py     symmetric weighted InfoNCE
  sampling.py      balanced class sampler
  train.py         training loop and objective
  evaluate.py      Top-1, macro-F1, average precision
  linear_probe.py  frozen-encoder probe
  robustness.py    pink noise, blur, desynchronisation
configs/           pretraining, transfer and probe profiles
tests/             unit and smoke tests
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

## Release history

See [CHANGELOG.md](CHANGELOG.md).

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

@software{bouacha2026savat-archive,
  title   = {{SAVAT}: Sports Audio-Visual Alignment Transformer for Action Spotting},
  author  = {Bouacha, Lazhar and Wendling, Laurent and Kurtz, Camille},
  version = {1.1.1},
  year    = {2026},
  doi     = {10.5281/zenodo.23605021}
}
```

The `@inproceedings` entry identifies the conference paper. The `@software`
entry identifies the archived software version that accompanies it.
