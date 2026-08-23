#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-gate history selection must fail closed and use the target head."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.build import persistence


def _result(returncode: int, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("git",), returncode, stdout, "")


def test_feature_branch_uses_target_head_not_merge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "a" * 40
    head = "b" * 40

    def fake_git(_repository: Path, *args: str):
        if args[:2] == ("rev-parse", "--verify"):
            return _result(0, target + "\n")
        if args == ("rev-parse", "HEAD"):
            return _result(0, head + "\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    assert persistence._baseline_commit(Path("."), base_ref="origin/master") == target


def test_target_branch_uses_first_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40
    parent = "c" * 40

    def fake_git(_repository: Path, *args: str):
        if args[:2] == ("rev-parse", "--verify"):
            return _result(0, head + "\n")
        if args == ("rev-parse", "HEAD"):
            return _result(0, head + "\n")
        if args == ("rev-parse", "HEAD^"):
            return _result(0, parent + "\n")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    assert persistence._baseline_commit(Path("."), base_ref="origin/master") == parent


def test_missing_target_ref_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        persistence,
        "_git",
        lambda _repository, *_args: _result(1),
    )
    with pytest.raises(ValueError, match="baseline ref is unavailable"):
        persistence._baseline_commit(Path("."), base_ref="origin/master")
