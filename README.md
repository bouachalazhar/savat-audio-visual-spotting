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
   Masking is applied independently to each modality.
3. **Reliability indicators** (`savat.reliability`): audio presence, temporal
   stability, clipping and tonality; video sharpness, exposure and temporal
   stability. They are aggregated with a geometric mean, so one collapsed
   indicator lowers the whole score. A normaliser can be fitted on the
   training partition only.
4. **Reliability-weighted InfoNCE** (`savat.alignment`): the symmetric
   contrastive loss is weighted by the product of the two reliability scores.
   With labels, negatives can be restricted to the same normalised class.
5. **Fusion** (`savat.fusion`): arithmetic, geometric, hard-selection and
   unimodal strategies. Geometric fusion is the default because a modality
   with zero weight disappears without discarding the sample.
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
grouped split is the file stem with a trailing index removed: `match_001` and
`match_002` belong to the group `match`. Re-encodings and fragments of the same
source should therefore share a stem prefix. When that convention is not
enough, provide one directory per source group.

The default split is 70/15/15, stratified by label and grouped by source, with
seeds 17, 29 or 43 as in the manuscript.

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
+ \lambda_{\text{rec}} \mathcal{L}_{\text{MSE}} .
$$

The contrastive term uses weights $w_i = r_v(i)\,r_a(i)$, where each
reliability score is the geometric mean of its indicators:

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
  --fusion geometric \
  --output metrics.json
```

The report contains Top-1, macro-F1, per-class average precision, mean average
precision and the confusion matrix.

## Robustness

```bash
uv run savat-robustness \
  --data-root data/sportvista \
  --checkpoint runs/savat_pretrain/best.pt \
  --noise-snr 5 \
  --blur-sigma 1.5 \
  --offsets 0.5,1.0,1.5,2.0 \
  --output robustness.json
```

The harness reports clean, pink-noise, Gaussian-blur and audio-shift
conditions, with the difference in mean average precision against the clean
reference. Noise and blur alter one modality only; the shift leaves both
signals intact but breaks their correspondence.

## Scope and differences from the article

This repository is a compact, runnable reimplementation of the method. The
following points delimit what it reproduces:

- The reported experiments initialise the encoders from CAV-MAE Sync. Those
  weights are not redistributed here, so `savat-train` starts from random
  initialisation by default.
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
uv run pytest
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{bouacha2025savat,
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
