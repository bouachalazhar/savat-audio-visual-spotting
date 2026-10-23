#!/usr/bin/env zsh
# Reproduce the deterministic audio-visual demo end to end and print the
# SHA-256 of the recorded evaluation. Run from the repository root with uv.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
uv run demo/generate_fixture.py
uv run savat-train --config demo/demo_pretrain_sportvista.yaml
uv run savat-evaluate --data-root demo/data/sportvista --checkpoint demo/runs/savat_demo/best.pt --fusion joint --output demo/metrics.json

echo "--- SHA-256 of the deterministic output ---"
shasum -a 256 demo/metrics.json
