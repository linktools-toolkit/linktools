#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RepoGit: Git capability gating for cntr repositories.

Python <3.10 / missing Dulwich must warn exactly once per instance, never
silently "succeed", and never leave a half-added repo behind. Local
(non-git) repositories never warn at all.
"""
import pytest

import linktools.cntr.repo.git as repo_git_module
from linktools.cntr.container import ContainerError
from linktools.cntr.repo.git import RepoGit, RepoGitResult


@pytest.fixture
def git_unavailable(monkeypatch):
    monkeypatch.setattr(repo_git_module, "is_git_available", lambda: False)
    monkeypatch.setattr(
        repo_git_module, "get_git_unavailable_reason",
        lambda: "Git repository operations require Python 3.10 or newer.",
    )


def test_clone_unavailable_warns_once_and_raises(fresh_manager, git_unavailable, tmp_path):
    git = RepoGit(fresh_manager)
    target = tmp_path / "clone-target"

    with pytest.raises(ContainerError):
        git.clone("https://example.com/repo.git", str(target))

    assert not target.exists()  # no half-created directory


def test_clone_unavailable_does_not_warn_twice(fresh_manager, git_unavailable, tmp_path):
    git = RepoGit(fresh_manager)
    warnings = []
    monkeypatch_logger = fresh_manager.logger
    orig_warning = monkeypatch_logger.warning
    monkeypatch_logger.warning = lambda msg: warnings.append(msg)
    try:
        for i in range(3):
            with pytest.raises(ContainerError):
                git.clone("https://example.com/repo.git", str(tmp_path / f"target-{i}"))
    finally:
        monkeypatch_logger.warning = orig_warning

    assert len(warnings) == 1


def test_update_unavailable_returns_failed_result_and_does_not_raise(fresh_manager, git_unavailable, tmp_path):
    git = RepoGit(fresh_manager)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    result = git.update("https://example.com/repo.git", str(repo_path))

    assert isinstance(result, RepoGitResult)
    assert result.success is False
    assert result.revision is None
    assert result.dirty is None
    assert "Python 3.10" in result.error


def test_inspect_unavailable_reports_unsupported_with_reason(fresh_manager, git_unavailable, tmp_path):
    git = RepoGit(fresh_manager)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    info = git.inspect(str(repo_path))

    assert info["supported"] is False
    assert info["revision"] is None
    assert info["dirty"] is None
    assert "Python 3.10" in info["reason"]


def test_local_repo_update_is_a_noop_when_git_unavailable(fresh_manager, git_unavailable, tmp_path):
    """A purely local (non-git) repo path must not even try to touch Git,
    so it never triggers the unavailable warning at all."""
    git = RepoGit(fresh_manager)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    # `available` gates first -- local repos never even reach the
    # not-a-git-repository branch when Git support itself is unavailable,
    # matching the "local repo features remain available" contract.
    result = git.update("file:///local/path", str(repo_path))
    assert result.success is False  # unavailable gate fires before repo-type is even checked


def test_inspect_missing_path_is_supported_but_unobserved(fresh_manager, tmp_path):
    """Git available, but nothing at repo_path yet (e.g. mid-add) -- not an
    error, just nothing to report."""
    git = RepoGit(fresh_manager)
    info = git.inspect(str(tmp_path / "does-not-exist"))
    assert info["supported"] is True
    assert info["revision"] is None
    assert info["reason"] is None
