"""Audio-visual action spotting with reliability-aware fusion.

This package accompanies the article:

    L. Bouacha, L. Wendling, C. Kurtz. SAVAT: Sports Audio-Visual Alignment
    Transformer for Action Spotting. ACIVS 2026, LNCS, in press.

Modules
-------
config       Strict YAML loading and command-line precedence.
data         Media decoding, clip datasets and grouped splits.
sampling     Balanced class sampler.
model        SAVAT transformer encoders, masked reconstruction and heads.
reliability  Per-modality reliability indicators and train-set normalisation.
fusion       Arithmetic, geometric and hard reliability-weighted fusion.
alignment    Reliability-weighted contrastive audio-visual alignment.
train        Training with reconstruction, alignment and reliability terms.
evaluate     Top-1, macro-F1 and average precision reports.
linear_probe Frozen-encoder linear probing of a checkpoint.
robustness   Pink noise, blur and desynchronisation stress tests.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.2.0"
