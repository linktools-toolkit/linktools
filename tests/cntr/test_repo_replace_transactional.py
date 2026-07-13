#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`repo add --replace` must not destroy the old repository until the new
one has fully succeeded (review P1-06): clone/symlink, requirement check,
AND the Store write must all succeed before the old directory is ever
touched -- and a failure removing the old directory afterward is a warning,
not a rollback of the now-successful replace. `repo remove` must not drop
the Store entry unless the directory deletion itself succeeded.
"""
import os

import pytest

from linktools.errors import GitError
from linktools.cntr.container import ContainerError

_URL = "https://example.com/repo.git"


def _working_clone(url, repo_path, branch=None):
    os.makedirs(repo_path)


def _add_working_repo(manager, monkeypatch):
    """Add a repo via a mocked, always-succeeding clone/requirement check
    -- the initial setup step every test here builds on before exercising a
    *second* add(replace=True) that fails or succeeds differently."""
    monkeypatch.setattr(manager.repos.git, "clone", _working_clone)
    monkeypatch.setattr(manager.repos, "_validate_new_repo_requirement", lambda repo_path: None)
    manager.repos.add(_URL)


def test_replace_clone_failure_preserves_old_repo(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    old_entry = dict(fresh_manager.repos.get_all()[_URL])
    old_path = old_entry["repo_path"]
    assert os.path.lexists(old_path)

    def broken_clone(url, repo_path, branch=None):
        os.makedirs(repo_path)
        raise GitError("clone failed")

    monkeypatch.setattr(fresh_manager.repos.git, "clone", broken_clone)

    with pytest.raises(GitError):
        fresh_manager.repos.add(_URL, replace=True)

    # Old repo untouched, new (never-finished) directory cleaned up.
    current = fresh_manager.repos.get_all()[_URL]
    assert current == old_entry
    assert os.path.lexists(old_path)


def test_replace_requirement_failure_preserves_old_repo(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    old_entry = dict(fresh_manager.repos.get_all()[_URL])
    old_path = old_entry["repo_path"]

    def fake_validate(repo_path):
        raise ContainerError("not usable")

    monkeypatch.setattr(fresh_manager.repos, "_validate_new_repo_requirement", fake_validate)

    with pytest.raises(ContainerError):
        fresh_manager.repos.add(_URL, replace=True)

    current = fresh_manager.repos.get_all()[_URL]
    assert current == old_entry
    assert os.path.lexists(old_path)


def test_replace_store_write_failure_preserves_old_repo_and_cleans_new_dir(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    old_entry = dict(fresh_manager.repos.get_all()[_URL])
    old_path = old_entry["repo_path"]

    new_paths = []
    original_dump = fresh_manager.repos._dump

    def broken_dump(repos):
        # Capture the new path before "failing" so the test can assert it
        # was cleaned up.
        new_paths.append(repos[_URL]["repo_path"])
        raise OSError("disk full")

    monkeypatch.setattr(fresh_manager.repos, "_dump", broken_dump)

    with pytest.raises(OSError):
        fresh_manager.repos.add(_URL, replace=True)

    monkeypatch.setattr(fresh_manager.repos, "_dump", original_dump)
    current = fresh_manager.repos.get_all()[_URL]
    assert current == old_entry
    assert os.path.lexists(old_path)
    assert not os.path.lexists(new_paths[0])


def test_replace_success_removes_old_directory_and_updates_store(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    old_path = fresh_manager.repos.get_all()[_URL]["repo_path"]

    fresh_manager.repos.add(_URL, replace=True)

    new_path = fresh_manager.repos.get_all()[_URL]["repo_path"]
    assert new_path != old_path
    assert not os.path.lexists(old_path)
    assert os.path.lexists(new_path)


def test_replace_old_repo_with_uncommitted_files_preserved_on_failure(fresh_manager, monkeypatch):
    """The old directory containing real (e.g. uncommitted) content must
    still be untouched when the replace attempt fails."""
    _add_working_repo(fresh_manager, monkeypatch)
    old_path = fresh_manager.repos.get_all()[_URL]["repo_path"]
    marker = os.path.join(old_path, "uncommitted.txt")
    with open(marker, "w") as f:
        f.write("important")

    def broken_clone(url, repo_path, branch=None):
        raise GitError("network drop")

    monkeypatch.setattr(fresh_manager.repos.git, "clone", broken_clone)

    with pytest.raises(GitError):
        fresh_manager.repos.add(_URL, replace=True)

    assert os.path.exists(marker)
    with open(marker) as f:
        assert f.read() == "important"


def test_add_without_replace_fails_when_repo_exists(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    with pytest.raises(ContainerError, match="already exists"):
        fresh_manager.repos.add(_URL)


# -- remove(): directory deletion must succeed before the Store forgets it --

def test_remove_deletion_failure_preserves_store_entry(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    before = dict(fresh_manager.repos.get_all())

    def broken_remove(repo):
        raise OSError("permission denied")

    monkeypatch.setattr(fresh_manager.repos, "_remove_repo_file", broken_remove)

    with pytest.raises(OSError):
        fresh_manager.repos.remove(_URL)

    assert fresh_manager.repos.get_all() == before


def test_get_all_does_not_expose_the_internal_mutable_dict(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)

    first = fresh_manager.repos.get_all()
    first[_URL]["repo_path"] = "/tampered"
    del first[_URL]

    second = fresh_manager.repos.get_all()
    assert _URL in second
    assert second[_URL]["repo_path"] != "/tampered"


def test_remove_success_drops_store_entry_and_directory(fresh_manager, monkeypatch):
    _add_working_repo(fresh_manager, monkeypatch)
    path = fresh_manager.repos.get_all()[_URL]["repo_path"]
    assert os.path.lexists(path)

    fresh_manager.repos.remove(_URL)

    assert _URL not in fresh_manager.repos.get_all()
    assert not os.path.lexists(path)
