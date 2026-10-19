# Provenance

This repository was assembled from the private `sportiq` research codebase at
commit `d1ec656e` (2026-10-19). The table below lists the correspondence between
the public modules and their private sources. The public code was cleaned of
local paths, cluster identifiers, project-specific settings and experiment
orchestration; the algorithms and their tests were kept.

| Public path | Private source |
| --- | --- |
| `src/savat/data.py` | `src/sportiq/training/training_multimodal.py` (`MultimodalDataset`, media decoding), extended with grouped splits and reliability output |
| `src/savat/model.py` | new: SAVAT dual transformer encoders, masked reconstruction, heads and modality dropout |
| `src/savat/reliability.py` | new: reliability indicators and train-set normalisation |
| `src/savat/fusion.py` | new: reliability-weighted fusion strategies |
| `src/savat/alignment.py` | new: reliability-weighted symmetric InfoNCE |
| `src/savat/train.py` | `src/sportiq/training/training_multimodal.py` (training loop, distributed execution), extended with the SAVAT objective |
| `src/savat/evaluate.py` | new: Top-1, macro-F1 and average precision |
| `src/savat/robustness.py` | new: pink noise, blur and desynchronisation harness |

The private repository remains the development source. This public repository
is the citable artefact associated with the article. No checkpoint, no
SportVISTA clip and no SoccerNet clip is included. The CAV-MAE Sync
initialisation used in the reported experiments is not redistributed; see the
README for the exact scope.
