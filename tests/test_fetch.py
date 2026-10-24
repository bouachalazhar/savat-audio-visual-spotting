"""Fetch-integrity tests use a local file:// source."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from savat.fetch import fetch


def test_fetch_writes_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    payload = b"approved payload"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "downloads" / "copy.bin"
    observed = fetch(source.as_uri(), destination, digest)
    assert observed == digest
    assert destination.read_bytes() == payload


def test_fetch_rejects_wrong_digest(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"unexpected payload")
    destination = tmp_path / "downloads" / "copy.bin"
    with pytest.raises(ValueError, match="integrity check failed"):
        fetch(source.as_uri(), destination, "0" * 64)
