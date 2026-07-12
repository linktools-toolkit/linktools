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

import pytest

from linktools.errors import GitError


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
