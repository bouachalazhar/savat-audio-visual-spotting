# Provenance

This repository is a cleaned, standalone release of research code developed for
the associated article. It was extracted from an internal research codebase;
local filesystem paths, cluster identifiers, private dataset locations,
experiment orchestration and project-specific configuration were removed. The
algorithms, public APIs and tests were retained and adapted for this release.

## Relationship to the article

The public modules implement the described architecture and training
objective:

- dual video and audio transformer encoders with tubelet and log-Mel patches;
- masked reconstruction and reliability-weighted contrastive alignment;
- train-only reliability normalisation and reliability-weighted feature fusion;
- clip-level transfer training and evaluation;
- noise, blur and desynchronisation stress tests.

The article's reported experiments use CAV-MAE Sync initialisation and the
official CALF evaluation protocol. Those weights and the official evaluation
scripts are not redistributed. The public code therefore provides the
architecture and training objective, not a bit-for-bit reproduction of the
reported numbers.

## What is not included

No SportVISTA clip, SoccerNet clip, trained checkpoint or CAV-MAE Sync weight is
distributed. Broadcast and platform rights do not allow redistribution. The
README documents the remaining scope differences.
