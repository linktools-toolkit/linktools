#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-gate history selection must fail closed and use the target head."""

from __future__ import annotations

import subprocess
import sys
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


def test_current_head_unavailable_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "a" * 40

    def fake_git(_repository: Path, *args: str):
        if args[:2] == ("rev-parse", "--verify"):
            return _result(0, target + "\n")
        if args == ("rev-parse", "HEAD"):
            return _result(1)
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    with pytest.raises(ValueError, match="current Git HEAD is unavailable"):
        persistence._baseline_commit(Path("."), base_ref="origin/master")


def test_target_branch_parent_unavailable_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40

    def fake_git(_repository: Path, *args: str):
        if args[:2] == ("rev-parse", "--verify"):
            return _result(0, head + "\n")
        if args == ("rev-parse", "HEAD"):
            return _result(0, head + "\n")
        if args == ("rev-parse", "HEAD^"):
            return _result(1)
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    with pytest.raises(ValueError, match="baseline parent is unavailable"):
        persistence._baseline_commit(Path("."), base_ref="origin/master")


def test_missing_git_repository_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    script = repository / "linktools-ai/scripts/build/persistence.py"
    candidate = repository / "linktools-ai/scripts/build/matrix/runtime.json"
    monkeypatch.setattr(persistence, "__file__", str(script))

    with pytest.raises(ValueError, match="Git repository is unavailable"):
        persistence._repository_path(candidate)


def test_contract_path_outside_repository_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    script = repository / "linktools-ai/scripts/build/persistence.py"
    monkeypatch.setattr(persistence, "__file__", str(script))

    with pytest.raises(ValueError, match="contract path is outside repository"):
        persistence._repository_path(tmp_path / "outside.json")


def _patch_baseline_location(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline: str,
    relative: str,
) -> None:
    monkeypatch.setattr(
        persistence,
        "_repository_path",
        lambda _path: (Path("."), relative),
    )
    monkeypatch.setattr(
        persistence,
        "_baseline_commit",
        lambda _repository, *, base_ref: baseline,
    )


def test_baseline_tree_read_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = "a" * 40
    relative = "matrix/runtime-persistence-v1.json"
    _patch_baseline_location(
        monkeypatch,
        baseline=baseline,
        relative=relative,
    )
    monkeypatch.setattr(
        persistence,
        "_git",
        lambda _repository, *_args: _result(128),
    )

    with pytest.raises(ValueError, match="baseline tree is unreadable"):
        persistence.load_git_json_baseline("candidate.json")


def test_missing_baseline_path_is_valid_first_introduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = "a" * 40
    relative = "matrix/new-contract.json"
    _patch_baseline_location(
        monkeypatch,
        baseline=baseline,
        relative=relative,
    )

    def fake_git(_repository: Path, *args: str):
        assert args == (
            "ls-tree",
            "--full-tree",
            baseline,
            "--",
            relative,
        )
        return _result(0)

    monkeypatch.setattr(persistence, "_git", fake_git)
    assert persistence.load_git_json_baseline("candidate.json") is None


def test_existing_baseline_path_read_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = "a" * 40
    relative = "matrix/runtime-persistence-v1.json"
    _patch_baseline_location(
        monkeypatch,
        baseline=baseline,
        relative=relative,
    )

    def fake_git(_repository: Path, *args: str):
        if args[0] == "ls-tree":
            return _result(
                0,
                f"100644 blob {'b' * 40}\t{relative}\n",
            )
        if args[0] == "show":
            return _result(128)
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    with pytest.raises(ValueError, match="baseline file is unreadable"):
        persistence.load_git_json_baseline("candidate.json")


def test_invalid_historical_json_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = "a" * 40
    relative = "matrix/runtime-persistence-v1.json"
    _patch_baseline_location(
        monkeypatch,
        baseline=baseline,
        relative=relative,
    )

    def fake_git(_repository: Path, *args: str):
        if args[0] == "ls-tree":
            return _result(
                0,
                f"100644 blob {'b' * 40}\t{relative}\n",
            )
        if args[0] == "show":
            return _result(0, "{")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    with pytest.raises(ValueError, match="historical JSON contract is invalid"):
        persistence.load_git_json_baseline("candidate.json")


def test_existing_historical_json_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = "a" * 40
    relative = "matrix/runtime-persistence-v1.json"
    _patch_baseline_location(
        monkeypatch,
        baseline=baseline,
        relative=relative,
    )

    def fake_git(_repository: Path, *args: str):
        if args[0] == "ls-tree":
            return _result(
                0,
                f"100644 blob {'b' * 40}\t{relative}\n",
            )
        if args[0] == "show":
            return _result(0, '{"wire_version": 1}')
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(persistence, "_git", fake_git)
    assert persistence.load_git_json_baseline("candidate.json") == {
        "wire_version": 1
    }


def test_checked_in_runtime_persistence_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["persistence"])
    assert persistence.main() == 0
