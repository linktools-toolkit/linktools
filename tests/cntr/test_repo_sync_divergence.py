#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Without --force, a diverged repo must fail and leave local commits alone.

Regression: RepoSync.sync() caught GitDivergedError from a plain
FAST_FORWARD_ONLY sync and silently retried with RESET_TO_REMOTE even when
the caller never asked for --force, discarding unpushed local commits.
"""
import pytest

import linktools.cntr.repo.sync as repo_sync_module
from linktools.cntr.repo.sync import RepoSync
from linktools.errors import GitDivergedError
from linktools.git import GitSyncPolicy


class _FakeGit:
    def stash(self, *args, **kwargs):
        pass

    def reset(self, hard=True):
        pass

    def checkout(self, branch):
        pass


class _FakeGitRepository:
    def __init__(self, environ, repo_path):
        self.git = _FakeGit()
        self.heads = []
        self.sync_calls = []

    @staticmethod
    def clone(environ, url, repo_path, branch=None):
        pass

    def is_dirty(self):
        return False

    def sync(self, policy):
        self.sync_calls.append(policy)
        if policy == GitSyncPolicy.FAST_FORWARD_ONLY:
            raise GitDivergedError("diverged")


@pytest.fixture
def fake_repo(monkeypatch):
    holder = {}

    def fake_ctor(environ, repo_path):
        instance = _FakeGitRepository(environ, repo_path)
        holder["instance"] = instance
        return instance

    fake_ctor.clone = _FakeGitRepository.clone
    monkeypatch.setattr(repo_sync_module, "GitRepository", fake_ctor)
    return holder


def test_update_without_force_raises_on_divergence_and_does_not_reset(fresh_manager, tmp_path, fake_repo):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    meta = {"type": "git", "repo_path": str(repo_path)}

    with pytest.raises(GitDivergedError):
        RepoSync(fresh_manager).sync("https://example.com/repo.git", meta, reset=False)

    assert fake_repo["instance"].sync_calls == [GitSyncPolicy.FAST_FORWARD_ONLY]


def test_update_with_force_resets_on_divergence(fresh_manager, tmp_path, fake_repo):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    meta = {"type": "git", "repo_path": str(repo_path)}

    RepoSync(fresh_manager).sync("https://example.com/repo.git", meta, reset=True)

    assert fake_repo["instance"].sync_calls == [GitSyncPolicy.RESET_TO_REMOTE]
