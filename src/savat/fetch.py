"""Fetch the third-party CAV-MAE Sync initialisation checkpoint.

The upstream authors distribute the checkpoint through their project page
(edsonroteia/cav-mae-sync, BSD-2-Clause). This command downloads that file to
a user-controlled location, verifies a pinned SHA-256 and refuses anything
that does not match. The repository itself never redistributes weights.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# Direct-download mirroring of the upstream ``get_pretrained_model.sh``:
# https://github.com/edsonroteia/cav-mae-sync/blob/main/pretrained_models/get_pretrained_model.sh
DEFAULT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1KBCvDcEJ5l7WtRXVQZC6TJNUpY4L6UZo&export=download&confirm=t"
)
DEFAULT_SHA256 = "a035b04cf615fe4c85d6aa1e74b26d8223a85945957664e66abea16e183629bd"
DEFAULT_DESTINATION = Path("downloads/cav_mae_sync.pth")
CHUNK = 8 * 1024 * 1024

UPSTREAM_LICENSE = (
    "CAV-MAE Sync is BSD-2-Clause-licensed upstream; keep the upstream notice"
    " when you reuse the file and comply with their terms."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path, sha256: str) -> str:
    """Download ``url`` to ``destination`` and verify the SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = urlopen(url, timeout=120)
    except URLError as error:
        message = f"download failed: {error}"
        raise RuntimeError(message) from error
    with destination.open("wb") as handle, response:
        transferred = 0
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            transferred += len(block)
    observed = _sha256(destination)
    if observed != sha256:
        message = (
            f"integrity check failed for {destination}:"
            f" sha256 {observed} != expected {sha256}"
        )
        raise ValueError(message)
    print(f"verified {destination} ({transferred} bytes, sha256 {observed})")
    return observed


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``savat-fetch-init``."""
    parser = argparse.ArgumentParser(
        prog="savat-fetch-init",
        description=(
            "Download the CAV-MAE Sync initialisation checkpoint used by the"
            " paper pretraining and verify its integrity."
        ),
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    print(UPSTREAM_LICENSE)
    destination = fetch(args.url, args.destination, args.sha256)
    print(f"run: savat-train --init-from {destination} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
