#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LifecycleDispatcher.notify_remove must register a removed-but-still-
running container's `configs` defaults onto ITS OWN env_config (the
manager's shared Config for a builtin container, or its own repository's
Config for a third-party one) -- not unconditionally onto the manager's
Config, which is a different object than a repo container's env_config
since the per-repository Config Context split."""
import os

import _harness

from linktools.cntr.context import EventContext


def _repo_with_field_only_container(tmp_path, name="repo_src"):
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    (repo_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr.container import BaseContainer\n\n\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def configs(self):\n"
        "        return {'REPO_ONLY_FIELD': ConfigField(default='repo-default-value')}\n",
        encoding="utf-8",
    )
    return repo_dir


def _fresh_standalone_manager(tmp_path):
    # A manager built from scratch (not the `fresh_manager` fixture, which
    # already memoizes `.containers` over just the builtins before a test
    # gets a chance to add its own repo -- see test_config_field_as_key.py).
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


def test_notify_remove_registers_configs_on_repo_container_own_config(tmp_path):
    repo_dir = _repo_with_field_only_container(tmp_path, name="repo_src")
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo_dir))
    # The loader derives the container's name from its containing repo
    # directory name (no explicit `name` given to BaseContainer, and no
    # nested numeric-prefixed subdirectory) -- "repo_src" here.
    manager.installed_state.add("repo_src")
    manager.prepare_installed_containers()

    container = manager.containers["repo_src"]
    assert container.repository_context is not None
    assert not container.repository_context.builtin
    # Sanity: the repo container's Config is NOT the manager's shared one.
    assert container.env_config is not manager.env_config

    manager.running_state._set(["repo_src"])

    ctx = EventContext()
    ctx.commands = ["up"]
    # A full-project context that no longer includes this container --
    # simulates it having been uninstalled while still running.
    ctx.containers = [c for c in manager.containers.values() if c.name != "repo_src"]
    ctx.target_containers = ctx.containers
    ctx.is_full_containers = True

    with manager.lifecycle.notify_remove(ctx):
        pass

    # The field must resolve through the container's OWN (repo-scoped)
    # Config -- this raised ConfigNotFoundError before the fix, since the
    # default used to be registered on the manager's Config instead.
    assert container.env_config.get("REPO_ONLY_FIELD") == "repo-default-value"
    # And it must NOT have leaked onto the manager's shared Config.
    assert "REPO_ONLY_FIELD" not in manager.env_config.schema
