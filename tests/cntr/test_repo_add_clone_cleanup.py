#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``RepoService.add()`` must never leave a half-cloned directory behind when
Git clone fails partway through, and must never write INSTALLED_REPOS for a
repository that never finished being added.

Regression: `add()` called `self.git.clone(...)` with no try/except at all --
a clone that created files on disk (a real partial checkout, e.g. after a
network drop mid-clone) before raising left that directory permanently
stranded, and the next `add()` of the same repo picked a new `-0`/`-1`
suffix instead of reusing the original name.
"""
import os
from typing import TYPE_CHECKING

import pytest

from linktools.errors import GitError

if TYPE_CHECKING:
    from pathlib import Path
    from linktools.cntr import ContainerManager


def test_failed_clone_removes_partial_directory(fresh_manager, monkeypatch):
    created_paths = []

    def broken_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)
        created_paths.append(repo_path)
        with open(os.path.join(repo_path, "partial"), "w") as file:
            file.write("partial")
        raise GitError("clone failed")

    monkeypatch.setattr(fresh_manager.repos.git, "clone", broken_clone)

    with pytest.raises(GitError):
        fresh_manager.repos.add("https://example.com/repo.git")

    assert len(created_paths) == 1
    assert not os.path.lexists(created_paths[0])
    assert "https://example.com/repo.git" not in fresh_manager.repos.get_all()


def test_failed_clone_does_not_bump_repo_path_suffix(fresh_manager, monkeypatch):
    """After a cleaned-up failed add, the next add of the same URL must
    reuse the original repo directory name -- not skip to `-0`/`-1` because
    the stale (never-cleaned) directory was still occupying it."""
    attempts = []

    def broken_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)
        attempts.append(repo_path)
        raise GitError("clone failed")

    monkeypatch.setattr(fresh_manager.repos.git, "clone", broken_clone)
    with pytest.raises(GitError):
        fresh_manager.repos.add("https://example.com/repo.git")

    def working_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)
        attempts.append(repo_path)

    monkeypatch.setattr(fresh_manager.repos.git, "clone", working_clone)
    monkeypatch.setattr(fresh_manager.repos, "_validate_new_repo_requirement", lambda repo_path: None)
    fresh_manager.repos.add("https://example.com/repo.git")

    assert attempts[0] == attempts[1]


def test_failed_requirement_validation_removes_cloned_directory(fresh_manager, monkeypatch):
    """clone() itself succeeds, but the post-clone requirement check fails
    -- the just-cloned directory must still be removed, and INSTALLED_REPOS
    must still never be written."""
    created_paths = []

    def fake_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)
        created_paths.append(repo_path)

    def fake_validate(repo_path):
        from linktools.cntr.container import ContainerError
        raise ContainerError("not usable")

    monkeypatch.setattr(fresh_manager.repos.git, "clone", fake_clone)
    monkeypatch.setattr(fresh_manager.repos, "_validate_new_repo_requirement", fake_validate)

    from linktools.cntr.container import ContainerError
    with pytest.raises(ContainerError):
        fresh_manager.repos.add("https://example.com/repo.git")

    assert len(created_paths) == 1
    assert not os.path.lexists(created_paths[0])
    assert "https://example.com/repo.git" not in fresh_manager.repos.get_all()


@pytest.mark.parametrize("replace", [False, True])
def test_interrupted_clone_cleans_staging_and_preserves_existing_repo(
        fresh_manager: "ContainerManager", monkeypatch: "pytest.MonkeyPatch", replace: bool) -> None:
    from pathlib import Path

    url = "https://example.com/repo.git"
    monkeypatch.setattr(fresh_manager.repos.git, "clone", lambda url, path, branch: os.makedirs(path))
    if replace:
        fresh_manager.repos.add(url)
    before = fresh_manager.repos.get_all()
    attempts = []

    def interrupted_clone(url: str, path: str, branch: "str | None" = None) -> None:
        root = Path(path)
        root.mkdir()
        (root / "partial").write_text("incomplete")
        attempts.append(root)
        raise KeyboardInterrupt()

    monkeypatch.setattr(fresh_manager.repos.git, "clone", interrupted_clone)
    with pytest.raises(KeyboardInterrupt):
        fresh_manager.repos.add(url, replace=replace)

    assert fresh_manager.repos.get_all() == before
    assert not attempts[0].exists()
    if replace:
        assert Path(before[url]["repo_path"]).is_dir()


def test_interrupted_local_validation_removes_only_new_link(
        fresh_manager: "ContainerManager", tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch") -> None:
    repo = tmp_path / "local-source"
    repo.mkdir()
    original = repo / "original.txt"
    original.write_text("preserved")
    staged = []

    def interrupted_validation(path: str) -> None:
        staged.append(path)
        raise KeyboardInterrupt()

    monkeypatch.setattr(fresh_manager.repos, "_validate_new_repo_requirement", interrupted_validation)
    with pytest.raises(KeyboardInterrupt):
        fresh_manager.repos.add(str(repo))

    assert fresh_manager.repos.get_all() == {}
    assert not os.path.lexists(staged[0])
    assert original.read_text() == "preserved"
