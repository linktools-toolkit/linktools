#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-repository config isolation through the real ContainerManager /
ContainerLoader path (spec §71's exact scenario) -- not just the
linktools.core primitive it's built on (see
tests/core/test_shared_config_sources.py for that lower-level coverage).

Two third-party repos, each declaring the SAME field name with a different
local `.linktools.json` value: each repo's container must read its own
repo's value regardless of load order, while a persisted or runtime
override applies to both simultaneously. Source string must never cross
between repos.
"""
import json
import os

import _harness

from linktools.core import ConfigField


def _repo_with_shared_field(tmp_path, name, storage_value):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / ".linktools.json").write_text(
        json.dumps({"environment": {"SHARED_FIELD": storage_value}}), encoding="utf-8",
    )
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'SHARED_FIELD': ConfigField(default='builtin-default')}\n",
        encoding="utf-8",
    )
    return repo_dir


def _fresh_standalone_manager(tmp_path):
    # Not the `fresh_manager` fixture -- it already memoizes `.containers`
    # over just the builtins before a test gets a chance to add repos.
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()
    data_path = tmp_path / "data"
    temp_path = tmp_path / "temp"
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(data_path)
    os.environ["LINKTOOLS_TEMP_PATH"] = str(temp_path)

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    return ContainerManager(Environ(), name="aio")


def _install_two_repos(tmp_path, add_order):
    repo_a = _repo_with_shared_field(tmp_path, "repo_a", "value-a")
    repo_b = _repo_with_shared_field(tmp_path, "repo_b", "value-b")
    manager = _fresh_standalone_manager(tmp_path)
    repos = {"repo_a": repo_a, "repo_b": repo_b}
    for name in add_order:
        manager.repo_store.add(str(repos[name]))
    manager.installed_state.add("repo_a", "repo_b")
    manager.prepare_installed_containers()
    return manager


def test_each_repo_container_reads_its_own_local_value(tmp_path):
    manager = _install_two_repos(tmp_path, add_order=["repo_a", "repo_b"])
    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]

    assert container_a.get_config("SHARED_FIELD") == "value-a"
    assert container_b.get_config("SHARED_FIELD") == "value-b"
    # Different repos must never share a local FileSource.
    assert container_a.env_config is not container_b.env_config


def test_load_order_does_not_affect_which_value_each_container_sees(tmp_path):
    manager = _install_two_repos(tmp_path, add_order=["repo_b", "repo_a"])
    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]

    assert container_a.get_config("SHARED_FIELD") == "value-a"
    assert container_b.get_config("SHARED_FIELD") == "value-b"


def test_persisted_value_overrides_both_repos_simultaneously(tmp_path):
    manager = _install_two_repos(tmp_path, add_order=["repo_a", "repo_b"])
    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]

    container_a.env_config.persist("SHARED_FIELD", "persisted-everywhere")

    assert container_a.get_config("SHARED_FIELD") == "persisted-everywhere"
    assert container_b.get_config("SHARED_FIELD") == "persisted-everywhere"


def test_runtime_override_applies_to_both_repos_simultaneously(tmp_path):
    manager = _install_two_repos(tmp_path, add_order=["repo_a", "repo_b"])
    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]

    container_b.env_config.set("SHARED_FIELD", "runtime-everywhere")

    assert container_a.get_config("SHARED_FIELD") == "runtime-everywhere"
    assert container_b.get_config("SHARED_FIELD") == "runtime-everywhere"


def test_global_file_value_inherited_by_both_repos(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    (home / ".linktools").mkdir()
    (home / ".linktools" / "linktools.json").write_text(
        json.dumps({"environment": {"GLOBAL_ONLY_FIELD": "global-value"}}), encoding="utf-8",
    )

    manager = _install_two_repos(tmp_path, add_order=["repo_a", "repo_b"])
    container_a = manager.containers["repo_a"]
    container_b = manager.containers["repo_b"]

    assert container_a.get_config("GLOBAL_ONLY_FIELD") == "global-value"
    assert container_b.get_config("GLOBAL_ONLY_FIELD") == "global-value"
