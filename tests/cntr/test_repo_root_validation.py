#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RepoService fails CLOSED for any unusable repository root.

Regression: `_revalidate_after_update()` treated a missing repo_path as
`(True, None)` -- compatible=True -- and `describe()` never validated the
repository root before calling `load_file_config(local_root=repo_path)`,
so a missing `repo_path` (None) silently fell through to
`local_root=None`, which reads the *process's current working directory*
`.linktools.json` instead of failing. Both `update()` and `describe()` now
route every repository root through `_validate_repo_root()` before
touching it at all.
"""
import json
import os

import pytest

from linktools.cntr.container import ContainerError


def _repo_dir(tmp_path, name="repo_src"):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    return repo_dir


def test_missing_repo_path_describe_reports_available_false(fresh_manager):
    meta = {"type": "local", "repo_path": None}
    info = fresh_manager.repos.describe("broken", meta)

    assert info["available"] is False
    assert info["compatible"] is False
    assert "repository_error" in info


def test_nonexistent_repo_path_describe_reports_available_false(fresh_manager, tmp_path):
    meta = {"type": "local", "repo_path": str(tmp_path / "does-not-exist")}
    info = fresh_manager.repos.describe("broken", meta)

    assert info["available"] is False
    assert info["compatible"] is False
    assert "repository_error" in info


def test_dangling_repo_root_symlink_is_explicitly_identified(fresh_manager, tmp_path):
    target = tmp_path / "gone"
    link = tmp_path / "dangling-repo"
    os.symlink(str(target), str(link), target_is_directory=True)
    meta = {"type": "local", "repo_path": str(link)}

    info = fresh_manager.repos.describe("broken", meta)

    assert info["available"] is False
    assert "dangling" in info["repository_error"]


def test_repo_root_as_plain_file_is_rejected(fresh_manager, tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_text("not a directory", encoding="utf-8")
    meta = {"type": "local", "repo_path": str(path)}

    info = fresh_manager.repos.describe("broken", meta)

    assert info["available"] is False
    assert info["compatible"] is False


def test_deleted_local_source_directory_is_not_compatible(fresh_manager, tmp_path):
    """A local repo whose SOURCE (the symlink target, not the symlink
    itself) has been removed -- the symlink in the repo pool becomes
    dangling. Neither `status` nor `update` may report it usable."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "container.py").write_text("# placeholder\n", encoding="utf-8")
    fresh_manager.repos.add(str(source), force=True)
    url, meta = next(iter(fresh_manager.repos.get_all().items()))

    import shutil
    shutil.rmtree(source)

    info = fresh_manager.repos.describe(url, meta)
    assert info["available"] is False
    assert info["compatible"] is False

    results = fresh_manager.repos.update()
    assert len(results) == 1
    assert results[0].updated is False
    assert results[0].compatible is False


def test_deleted_git_checkout_directory_is_not_compatible(fresh_manager, tmp_path):
    fresh_manager.repos._dump({
        "https://example.com/repo.git": dict(
            type="git", repo_path=str(tmp_path / "gone-checkout"), repo_name="repo"),
    })
    url, meta = next(iter(fresh_manager.repos.get_all().items()))

    info = fresh_manager.repos.describe(url, meta)
    assert info["available"] is False
    assert info["compatible"] is False

    results = fresh_manager.repos.update()
    assert results[0].updated is False
    assert results[0].compatible is False


def test_update_does_not_implicitly_reclone_a_missing_git_checkout(fresh_manager, monkeypatch, tmp_path):
    """update() must fail closed on a missing Git checkout root rather than
    silently re-cloning it (that implicit self-heal belongs to `add`, not
    `update`)."""
    fresh_manager.repos._dump({
        "https://example.com/repo.git": dict(
            type="git", repo_path=str(tmp_path / "gone-checkout"), repo_name="repo"),
    })

    def fail(*args, **kwargs):
        raise AssertionError("RepoGit.update must not be called for a missing checkout root")

    monkeypatch.setattr(fresh_manager.repos.git, "update", fail)

    results = fresh_manager.repos.update()
    assert results[0].updated is False


def test_describe_never_falls_back_to_cwd_linktools_json(fresh_manager, tmp_path, monkeypatch):
    """A missing repo_path must never cause describe() to read the
    process's current working directory's `.linktools.json`."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".linktools.json").write_text(
        json.dumps({"requires": {"linktools-cntr": ">=0.0.1"}}), encoding="utf-8")
    monkeypatch.chdir(cwd)

    meta = {"type": "local", "repo_path": None}
    info = fresh_manager.repos.describe("broken", meta)

    assert info["available"] is False
    assert info.get("requires") == {}
    assert "repository_error" in info


def test_remove_still_works_on_a_broken_repo_record(fresh_manager, tmp_path):
    """Even though the root is unusable, `remove()` must still let the user
    clean up the stale INSTALLED_REPOS entry."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "container.py").write_text("# placeholder\n", encoding="utf-8")
    fresh_manager.repos.add(str(source), force=True)
    url = next(iter(fresh_manager.repos.get_all().keys()))

    import shutil
    shutil.rmtree(source)

    fresh_manager.repos.remove(url)
    assert fresh_manager.repos.get_all() == {}
