# Deterministic demo transcript

The demo proves that the packaged audio-visual pipeline runs end to end and
reproduces its outputs inside the environment pinned by `uv.lock` (same
platform, same bundled codec build). The fixture uses closed-form content
with no random source anywhere in the chain.

Fixture generation: `demo/generate_fixture.py` (24 clips: 2 labels ×
3 groups × 4 clips; 16 frames of 32 px at 8 fps, 2 s of mono 16 kHz audio;
codecs `mpeg4`/`aac`, the ones the test suite already relies on). Every clip
varies its block level, block size, texture phase, tone and volume through
closed-form formulas so that the reliability indicators seen by the min-max
normaliser are continuous and never collapse to a constant bound. Tree
layout:

```
demo/data/sportvista/<label>/<group>/clip_***.mp4
```

Reproduce with `./demo/run_demo.sh` from the repository root. Last verified
locally on a macOS arm CPU host:

```bash
uv run demo/generate_fixture.py
uv run savat-train --config demo/demo_pretrain_sportvista.yaml
# epoch 1: val_alignment_loss 0.70004785, train_loss 0.97375166,
#          train_alignment 1.43410474, train_video_reconstruction 0.80498266,
#          train_audio_reconstruction 0.73842376
# epoch 2: val_alignment_loss 0.69558185, train_loss 0.98345190,
#          train_alignment 1.54097897, train_video_reconstruction 0.76660544,
#          train_audio_reconstruction 0.71505344
# best checkpoint: demo/runs/savat_demo/best.pt
uv run savat-evaluate --data-root demo/data/sportvista --checkpoint demo/runs/savat_demo/best.pt --fusion joint --output demo/metrics.json
# clips=24 top1=0.5000 macro_f1=0.3333 mAP=0.6326
```

Determinism check: two consecutive executions of `./demo/run_demo.sh`
produced the identical evaluation file.

```
7e01bc74114a432254563db86bfeb72ea71abdfab239872c5ce86cf247ecbf78  demo/metrics.json
```

The recorded evaluation is byte-identical to the committed reference at
`demo/expected/metrics.json`; `shasum -a 256 demo/expected/metrics.json`
gives the same hash above.

Scope: the per-epoch wall clock recorded in
`demo/runs/savat_demo/history.json` is not a deterministic artefact (it is
timing metadata by design); the demo profile is a tiny CPU configuration
(embed dim 32, depth 1, two epochs), so its numbers are demo numbers only
and are not paper results — with two epochs the classifier head stays at
chance level, which demonstrates runnability and reproducibility, not
accuracy. On hosts outside the pinned `uv.lock` environment (different
platforms or unbundled codec builds), float rounding or codec differences
may change the last digits — the environment pin, not the demo data,
provides the reproducibility contract.
