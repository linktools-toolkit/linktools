#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RepoService: local vs Git repository type routing.

Regression: RepoService.update()/describe() used to call RepoGit
unconditionally for every repository, so a purely local (non-git) repo
triggered Git-unavailable warnings/failures on Python <3.10 even though it
never needs Git at all. Business type (git vs local) must be decided by
RepoService, and RepoGit must only ever see real Git repositories.
"""
import pytest

from linktools.cntr.container import ContainerError
from linktools.cntr.repo.service import RepoUpdateResult, _is_remote_git_url


@pytest.fixture
def local_repo(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "container.py").write_text("# placeholder\n")
    return str(source)


def test_local_repo_update_does_not_call_git(fresh_manager, monkeypatch, local_repo):
    fresh_manager.repos.add(local_repo, force=True)

    def fail(*args, **kwargs):
        raise AssertionError("RepoGit.update must not be called for local repos")

    monkeypatch.setattr(fresh_manager.repos.git, "update", fail)

    results = fresh_manager.repos.update()

    assert len(results) == 1
    assert isinstance(results[0], RepoUpdateResult)
    assert results[0].updated is True
    assert results[0].compatible is True
    assert results[0].error is None
    assert results[0].revision is None


def test_local_repo_describe_does_not_inspect_git(fresh_manager, monkeypatch, local_repo):
    fresh_manager.repos.add(local_repo, force=True)
    url, meta = next(iter(fresh_manager.repos.get_all().items()))

    def fail(*args, **kwargs):
        raise AssertionError("RepoGit.inspect must not be called for local repos")

    monkeypatch.setattr(fresh_manager.repos.git, "inspect", fail)

    info = fresh_manager.repos.describe(url, meta)

    assert info["git"]["applicable"] is False
    assert info["git"]["supported"] is True


def test_local_repo_status_produces_no_git_warning(fresh_manager, local_repo):
    fresh_manager.repos.add(local_repo, force=True)
    warnings = []
    fresh_manager.repos.git.logger.warning = lambda msg: warnings.append(msg)

    url, meta = next(iter(fresh_manager.repos.get_all().items()))
    fresh_manager.repos.describe(url, meta)
    fresh_manager.repos.update()

    assert warnings == []


def test_git_repo_update_still_calls_git(fresh_manager, monkeypatch, local_repo):
    # Reuse the local fixture path as a stand-in git repo URL/meta -- only
    # the type routing (not real cloning) is under test here.
    calls = []

    def fake_update(url, repo_path, branch=None, reset=False):
        calls.append(url)
        from linktools.cntr.repo.git import RepoGitResult
        return RepoGitResult(success=True, revision="deadbeef", dirty=False, error=None)

    monkeypatch.setattr(fresh_manager.repos.git, "update", fake_update)
    fresh_manager.repos._dump({
        "https://example.com/repo.git": dict(type="git", repo_path=local_repo, repo_name="repo"),
    })

    results = fresh_manager.repos.update()

    assert calls == ["https://example.com/repo.git"]
    assert results[0].updated is True
    assert results[0].revision == "deadbeef"


def test_git_repo_describe_still_inspects_git(fresh_manager, monkeypatch, local_repo):
    calls = []

    def fake_inspect(repo_path):
        calls.append(repo_path)
        return {"applicable": True, "supported": True, "revision": "deadbeef", "dirty": False, "reason": None}

    monkeypatch.setattr(fresh_manager.repos.git, "inspect", fake_inspect)
    fresh_manager.repos._dump({
        "https://example.com/repo.git": dict(type="git", repo_path=local_repo, repo_name="repo"),
    })

    url, meta = next(iter(fresh_manager.repos.get_all().items()))
    info = fresh_manager.repos.describe(url, meta)

    assert calls == [local_repo]
    assert info["git"]["applicable"] is True
    assert info["git"]["revision"] == "deadbeef"


def test_unknown_repo_type_update_fails_explicitly_without_crashing_others(fresh_manager, local_repo):
    fresh_manager.repos._dump({
        "bad": dict(type="mystery", repo_path=local_repo, repo_name="bad"),
        "good": dict(type="local", repo_path=local_repo, repo_name="good"),
    })

    results = fresh_manager.repos.update()

    by_url = {r.url: r for r in results}
    assert by_url["bad"].updated is False
    assert "unsupported type" in by_url["bad"].error
    assert by_url["good"].updated is True


def test_unknown_repo_type_describe_reports_incompatible_without_raising(fresh_manager, local_repo):
    meta = dict(type="mystery", repo_path=local_repo, repo_name="bad")
    info = fresh_manager.repos.describe("bad", meta)

    assert info["compatible"] is False
    assert "unsupported type" in info["local_config_error"]


def test_missing_repo_path_update_fails_explicitly(fresh_manager):
    fresh_manager.repos._dump({"broken": dict(type="local", repo_name="broken")})

    results = fresh_manager.repos.update()

    assert results[0].updated is False
    assert "path is missing" in results[0].error.lower()


@pytest.mark.parametrize("value", [
    "https://host/repo.git",
    "http://host/repo.git",
    "ssh://host/repo.git",
    "git://host/repo.git",
    "file:///tmp/repo.git",
])
def test_remote_git_url_recognized(value):
    assert _is_remote_git_url(value)


@pytest.mark.parametrize("value", [
    "/tmp/repo",
    "./repo",
    "../repo",
    r"C:\repo",
    "C:/repo",
    r"\\server\share\repo",
    "git@host:repo.git",
    "user@host:path/repo.git",
])
def test_local_path_not_recognized_as_remote(value):
    assert not _is_remote_git_url(value)
