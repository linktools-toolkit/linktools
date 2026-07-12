#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RepoService.describe() must defer entirely to the real
LinktoolsFileConfigLoader for whether a repository's ``.linktools.json`` is
usable -- never re-implement that judgment with its own filesystem check.

Regression: describe() used ``os.path.exists(local_path)`` to decide
"present" vs "absent". A dangling symlink satisfies ``lexists`` but not
``exists``, so it was reported as "absent" (silently ignored) even though
the real Loader raises ConfigError for exactly this case (a path that
lexists but cannot be opened) -- Status disagreed with what would actually
happen if this repo's containers were loaded.
"""
import json
import os

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


def _describe(fresh_manager, repo_dir):
    fresh_manager.repos.add(str(repo_dir), force=True)
    meta = fresh_manager.repos.get_all()[str(repo_dir)]
    return fresh_manager.repos.describe(str(repo_dir), meta)


def _add_then_corrupt(fresh_manager, repo_dir, corrupt):
    """Add a repo while its `.linktools.json` is still valid/absent (`add()`
    itself validates and would reject a bad file up front), then corrupt it
    afterward -- the realistic way a repo ends up broken post-install (a
    hand-edit, a bad `repo update`), which is exactly what `status` must
    catch."""
    fresh_manager.repos.add(str(repo_dir), force=True)
    corrupt()
    meta = fresh_manager.repos.get_all()[str(repo_dir)]
    return fresh_manager.repos.describe(str(repo_dir), meta)


def test_absent_local_config_is_compatible(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)
    info = _describe(fresh_manager, repo_dir)
    assert info["local_config"] == "absent"
    assert info["compatible"] is True


def test_normal_local_config_resolves(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)
    (repo_dir / ".linktools.json").write_text(json.dumps({"requires": {}}), encoding="utf-8")
    info = _describe(fresh_manager, repo_dir)
    assert info["local_config"] == "present"
    assert info["compatible"] is True


def test_dangling_symlink_is_present_but_incompatible(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)
    target = tmp_path / "nonexistent-target.json"

    def corrupt():
        os.symlink(str(target), str(repo_dir / ".linktools.json"))

    info = _add_then_corrupt(fresh_manager, repo_dir, corrupt)
    assert info["local_config"] == "present"
    assert info["compatible"] is False
    assert "local_config_error" in info


def test_local_config_as_directory_is_present_but_incompatible(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)

    def corrupt():
        (repo_dir / ".linktools.json").mkdir()

    info = _add_then_corrupt(fresh_manager, repo_dir, corrupt)
    assert info["local_config"] == "present"
    assert info["compatible"] is False


def test_invalid_json_is_present_but_incompatible(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)

    def corrupt():
        (repo_dir / ".linktools.json").write_text("{not json", encoding="utf-8")

    info = _add_then_corrupt(fresh_manager, repo_dir, corrupt)
    assert info["local_config"] == "present"
    assert info["compatible"] is False


def test_unsupported_requirement_is_present_but_incompatible(fresh_manager, tmp_path):
    repo_dir = _repo_dir(tmp_path)

    def corrupt():
        (repo_dir / ".linktools.json").write_text(
            json.dumps({"requires": {"linktools-cntr": ">=999.0.0"}}), encoding="utf-8")

    info = _add_then_corrupt(fresh_manager, repo_dir, corrupt)
    assert info["local_config"] == "present"
    assert info["compatible"] is False
    assert "compatibility_issues" in info


def test_add_rejects_dangling_symlink_before_it_ever_reaches_status(fresh_manager, tmp_path):
    """A dangling `.linktools.json` must never even get into INSTALLED_REPOS
    in the first place -- `add()` validates through the same real Loader."""
    repo_dir = _repo_dir(tmp_path)
    target = tmp_path / "nonexistent-target.json"
    os.symlink(str(target), str(repo_dir / ".linktools.json"))

    import pytest
    with pytest.raises(ContainerError):
        fresh_manager.repos.add(str(repo_dir), force=True)
    assert fresh_manager.repos.get_all() == {}
