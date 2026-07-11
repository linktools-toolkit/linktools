#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cntr's lightweight `.linktools.json` requirement gate: `repo add`/`repo
update` only ever check `requires.linktools-cntr` from the repository's own
*local* file, before its `container.py` is ever imported. No manifest
kind/schema_version/components envelope, no Docker/Compose runtime
requirement gating (removed entirely, not migrated)."""
import json

import pytest

from linktools.cntr.container import ContainerError


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _repo_with_container(tmp_path, name="repo_src", requires=None, marker_name="imported.marker"):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    if requires is not None:
        _write(repo_dir / ".linktools.json", {"requires": requires})
    (repo_dir / "container.py").write_text(
        "from linktools.cntr.container import BaseContainer\n"
        f"open(__file__.rsplit('/', 1)[0] + '/{marker_name}', 'w').close()\n\n\n"
        "class Container(BaseContainer):\n    pass\n",
        encoding="utf-8",
    )
    return repo_dir


# -- repo add: local requirement gating --------------------------------------

def test_add_repo_without_linktools_json_succeeds(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires=None)
    fresh_manager.repos.add(str(repo_dir))
    assert str(repo_dir) in fresh_manager.repos.get_all()


def test_add_repo_with_satisfied_requirement_succeeds(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=0.0.1"})
    fresh_manager.repos.add(str(repo_dir))
    assert str(repo_dir) in fresh_manager.repos.get_all()


def test_add_repo_with_unsatisfied_requirement_raises(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=999.0"})
    with pytest.raises(ContainerError):
        fresh_manager.repos.add(str(repo_dir))


def test_add_repo_cleans_up_on_unsatisfied_requirement(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=999.0"})
    with pytest.raises(ContainerError):
        fresh_manager.repos.add(str(repo_dir))
    assert str(repo_dir) not in fresh_manager.repos.get_all()


def test_add_repo_with_invalid_specifier_raises(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": "not a specifier!!"})
    with pytest.raises(ContainerError):
        fresh_manager.repos.add(str(repo_dir))


def test_ai_requirement_is_ignored_by_cntr(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-ai": ">=999.0"})
    fresh_manager.repos.add(str(repo_dir))
    assert str(repo_dir) in fresh_manager.repos.get_all()


# -- global requires must never widen/override a repo's own declaration -----

def test_global_requires_does_not_affect_local_repo_gating(fresh_manager, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    _write(home.joinpath(".linktools").mkdir(parents=True) or (home / ".linktools" / "linktools.json"),
           {"requires": {"linktools-cntr": ">=999.0"}})
    # The repo itself declares no requirement -- the incompatible *global*
    # requires must not leak in and block it.
    repo_dir = _repo_with_container(tmp_path, requires=None)
    fresh_manager.repos.add(str(repo_dir))
    assert str(repo_dir) in fresh_manager.repos.get_all()


# -- requirement check happens before container.py is ever imported ---------

def test_unsatisfied_requirement_blocks_import_before_it_happens(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=999.0"},
                                     marker_name="should_not_exist.marker")
    with pytest.raises(ContainerError):
        fresh_manager.repos.add(str(repo_dir))
    # add() only symlinks the repo path in; the marker file only gets
    # written if container.py is actually imported.
    assert not list(tmp_path.glob("**/should_not_exist.marker"))


# -- loader skip (installed-but-now-incompatible repo is skipped, not crashed) --

def test_loader_skips_repo_that_became_incompatible(fresh_manager, tmp_path, monkeypatch):
    repo_dir = _repo_with_container(tmp_path, requires=None)
    fresh_manager.repos.add(str(repo_dir))

    # Simulate the repo's .linktools.json changing to something unsatisfiable
    # after it was already installed (e.g. an in-place edit).
    _write(repo_dir / ".linktools.json", {"requires": {"linktools-cntr": ">=999.0"}})

    containers = fresh_manager.loader.load_all()
    assert "container" not in [c.name for c in containers if str(getattr(c.repository_context, "root_path", "")) == str(repo_dir)]


# -- repo update: re-checked after sync, never rolled back ------------------

def test_update_reports_incompatible_after_requirement_regresses(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires=None)
    fresh_manager.repos.add(str(repo_dir))

    _write(repo_dir / ".linktools.json", {"requires": {"linktools-cntr": ">=999.0"}})

    results = fresh_manager.repos.update()
    assert len(results) == 1
    result = results[0]
    assert result.updated is True
    assert result.compatible is False
    assert "linktools-cntr" in (result.error or "")


def test_update_reports_compatible_when_requirement_holds(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=0.0.1"})
    fresh_manager.repos.add(str(repo_dir))

    results = fresh_manager.repos.update()
    assert results[0].compatible is True


# -- repo status --------------------------------------------------------------

def test_describe_repository_reports_requires_and_compatible(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires={"linktools-cntr": ">=0.0.1"})
    fresh_manager.repos.add(str(repo_dir))
    meta = fresh_manager.repos.get_all()[str(repo_dir)]

    info = fresh_manager.repos.describe(str(repo_dir), meta)
    assert info["local_config"] == "present"
    assert info["requires"] == {"linktools-cntr": ">=0.0.1"}
    assert info["compatible"] is True


def test_describe_repository_reports_absent_local_config(fresh_manager, tmp_path):
    repo_dir = _repo_with_container(tmp_path, requires=None)
    fresh_manager.repos.add(str(repo_dir))
    meta = fresh_manager.repos.get_all()[str(repo_dir)]

    info = fresh_manager.repos.describe(str(repo_dir), meta)
    assert info["local_config"] == "absent"
    assert info["compatible"] is True
