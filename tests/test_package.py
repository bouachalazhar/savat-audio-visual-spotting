"""Package metadata consistency."""

from __future__ import annotations

from importlib.metadata import version

from savat import __version__


def test_package_version_matches_distribution_metadata() -> None:
    assert __version__ == version("savat")
