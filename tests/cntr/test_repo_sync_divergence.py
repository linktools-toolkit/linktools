#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Without --force, a diverged repo must fail and leave local commits alone.

Regression: RepoGit.update() must not catch GitDivergedError from a plain
STASH_AND_RESTORE/fast-forward sync and silently retry with RESET_TO_REMOTE
when the caller never asked for --force, discarding unpushed local commits.
"""
import pytest

import linktools.cntr.repo.git as repo_git_module
from linktools.cntr.repo.git import RepoGit
from linktools.errors import GitDivergedError
from linktools.git import GitSyncPolicy


class _FakeGitRepository:
    def __init__(self):
        self.sync_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def is_dirty(self):
        return False

    def head_sha(self):
        return "deadbeef"

    def sync(self, policy):
        self.sync_calls.append(policy)
        if policy == GitSyncPolicy.STASH_AND_RESTORE:
            raise GitDivergedError("diverged")


@pytest.fixture
def fake_repo(monkeypatch):
    holder = {}

    def fake_open_if_valid(environ, repo_path):
        instance = _FakeGitRepository()
        holder["instance"] = instance
        return instance

    fake_module = type("FakeGitRepositoryModule", (), {
        "open_if_valid": staticmethod(fake_open_if_valid),
    })
    monkeypatch.setattr(repo_git_module, "GitRepository", fake_module)
    return holder


def test_update_without_force_raises_on_divergence_and_does_not_reset(fresh_manager, tmp_path, fake_repo):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with pytest.raises(GitDivergedError):
        RepoGit(fresh_manager).update("https://example.com/repo.git", str(repo_path), reset=False)

    assert fake_repo["instance"].sync_calls == [GitSyncPolicy.STASH_AND_RESTORE]


def test_update_with_force_resets_on_divergence(fresh_manager, tmp_path, fake_repo):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    RepoGit(fresh_manager).update("https://example.com/repo.git", str(repo_path), reset=True)

    assert fake_repo["instance"].sync_calls == [GitSyncPolicy.RESET_TO_REMOTE]
