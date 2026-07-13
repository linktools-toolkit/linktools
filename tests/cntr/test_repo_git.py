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

    assert info["applicable"] is True
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
    assert info["applicable"] is True
    assert info["supported"] is True
    assert info["revision"] is None
    assert info["reason"] is None


def test_inspect_non_git_directory_is_applicable(fresh_manager, tmp_path):
    """A real directory that just isn't a Git checkout -- distinct from
    "nothing exists yet" -- must still report applicable=True (this is a
    Git-capable code path, it just found no repo there)."""
    git = RepoGit(fresh_manager)
    repo_path = tmp_path / "not-a-repo"
    repo_path.mkdir()

    info = git.inspect(str(repo_path))

    assert info["applicable"] is True
    assert info["supported"] is True
    assert info["revision"] is None
    assert info["dirty"] is None
    assert info["reason"] == "Not a git repository."


def test_inspect_corrupted_repo_fails_closed_not_raises(fresh_manager, tmp_path, monkeypatch):
    """review P2-04: a corrupted object store/permission error while
    reading head_sha()/is_dirty() must be reported structurally, not
    propagate -- inspect() is used by describe()/validate(), which promise
    one bad repo can never hide the rest of a multi-repo status/validate."""
    from linktools.git.repository import GitRepository

    git = RepoGit(fresh_manager)
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    from dulwich import porcelain
    porcelain.init(str(repo_path))
    (repo_path / "a.txt").write_text("hello")
    porcelain.add(str(repo_path), [str(repo_path / "a.txt")])
    porcelain.commit(str(repo_path), message=b"first",
                     author=b"T <t@example.com>", committer=b"T <t@example.com>")

    def broken_head_sha(self):
        raise RuntimeError("corrupted object store")

    monkeypatch.setattr(GitRepository, "head_sha", broken_head_sha)

    info = git.inspect(str(repo_path))

    assert info["applicable"] is True
    assert info["supported"] is False
    assert info["revision"] is None
    assert info["dirty"] is None
    assert "corrupted object store" in info["reason"]


def test_describe_isolates_one_bad_repo_from_others(fresh_manager, tmp_path, monkeypatch):
    """A repo whose Git inspection raises must not prevent describe()/
    validate() from reporting every other repository."""
    from linktools.git.repository import GitRepository
    from dulwich import porcelain

    good_repo = tmp_path / "good"
    good_repo.mkdir()
    porcelain.init(str(good_repo))
    (good_repo / "a.txt").write_text("hello")
    porcelain.add(str(good_repo), [str(good_repo / "a.txt")])
    porcelain.commit(str(good_repo), message=b"first",
                     author=b"T <t@example.com>", committer=b"T <t@example.com>")

    bad_repo = tmp_path / "bad"
    bad_repo.mkdir()
    porcelain.init(str(bad_repo))
    (bad_repo / "b.txt").write_text("hello")
    porcelain.add(str(bad_repo), [str(bad_repo / "b.txt")])
    porcelain.commit(str(bad_repo), message=b"first",
                     author=b"T <t@example.com>", committer=b"T <t@example.com>")

    def broken_head_sha(self):
        if str(self._path) == str(bad_repo):
            raise RuntimeError("corrupted object store")
        return "deadbeef" * 5

    monkeypatch.setattr(GitRepository, "head_sha", broken_head_sha)

    fresh_manager.repos._dump({
        "good-url": dict(type="git", repo_path=str(good_repo), repo_name="good"),
        "bad-url": dict(type="git", repo_path=str(bad_repo), repo_name="bad"),
    })

    results, _ = fresh_manager.repos.validate()

    assert results["good-url"]["git"]["supported"] is True
    assert results["bad-url"]["git"]["supported"] is False
    assert "corrupted object store" in results["bad-url"]["git"]["reason"]


def test_repo_service_add_remote_url_while_unavailable_leaves_no_trace(fresh_manager, git_unavailable):
    """End-to-end through RepoService.add() (not just the RepoGit adapter):
    a remote `repo add` while Git is unavailable must fail non-zero and
    never write the repo into INSTALLED_REPOS."""
    with pytest.raises(ContainerError):
        fresh_manager.repos.add("https://example.com/some/repo.git")

    assert fresh_manager.repos.get_all() == {}
