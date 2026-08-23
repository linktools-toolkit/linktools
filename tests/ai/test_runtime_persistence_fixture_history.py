#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical persistence fixtures are immutable release evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=False,
        capture_output=True,
        text=True,
    )


def _historical_bytes(path: Path) -> bytes | None:
    root = Path(__file__).resolve().parents[2]
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    verified = _git(root, "rev-parse", "--verify", "origin/master^{commit}")
    if verified.returncode != 0:
        return None
    base = _git(root, "merge-base", "HEAD", "origin/master")
    if base.returncode != 0 or not base.stdout.strip():
        return None
    historical = subprocess.run(
        ("git", "-C", str(root), "show", f"{base.stdout.strip()}:{relative}"),
        check=False,
        capture_output=True,
    )
    return None if historical.returncode != 0 else historical.stdout


def test_model_message_v1_fixture_is_append_only() -> None:
    fixture = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai/scripts/build/matrix/runtime_model_messages_v1.json"
    )
    historical = _historical_bytes(fixture)
    if historical is None:
        pytest.skip("fixture has no released merge-base version yet")
    assert fixture.read_bytes() == historical
