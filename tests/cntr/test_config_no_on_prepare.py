#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration commands must not run preparation or native validation callbacks."""
import os

import pytest

import _harness
from linktools.cntr.commands.config import ConfigCommand
import linktools.cntr.commands._shared as cntr_shared


def _fresh_standalone_manager(tmp_path):
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


def test_config_set_works_on_fresh_install_with_nothing_installed(tmp_path, monkeypatch):
    manager = _fresh_standalone_manager(tmp_path)
    assert manager.installed_state.get(resolve=False) == []
    monkeypatch.setattr(cntr_shared, "manager", manager)

    # Must not raise "No container installed".
    ConfigCommand().on_command_set(configs={"HOST": "example.com"})
    assert manager.env_config.get("HOST") == "example.com"


def test_config_list_works_on_fresh_install_with_nothing_installed(tmp_path, monkeypatch):
    manager = _fresh_standalone_manager(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    # Must not raise.
    ConfigCommand().on_command_list(names=[])


@pytest.mark.parametrize("subcommand,kwargs", [
    ("on_command_set", dict(configs={"DOCKER_HOST": "/var/run/docker.sock"})),
    ("on_command_get", dict(keys=["DOCKER_HOST"])),
    ("on_command_list", dict(names=[])),
    ("on_command_explain", dict(key="DOCKER_HOST")),
    ("on_command_validate", dict()),
    ("on_command_reload", dict()),
])
def test_config_subcommands_never_run_runtime_callbacks(monkeypatch, fresh_manager, subcommand, kwargs):
    def unexpected(*args, **kwargs):
        raise AssertionError("Configuration commands must remain read-only")

    for container in fresh_manager.containers.values():
        monkeypatch.setattr(container, "on_starting", unexpected)
        monkeypatch.setattr(container, "on_check", unexpected)

    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    getattr(ConfigCommand(), subcommand)(**kwargs)

