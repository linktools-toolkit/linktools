#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config commands must never run a container's on_prepare() (review P1-04).

Before this, every config subcommand (set/get/list/explain/validate/reload)
routed through `prepare_installed_containers()`, which runs every installed
container's `on_prepare()` -- arbitrary third-party file writes, network
access, hook registration -- just to answer a config question. It also
raised "No container installed" on a fresh install (nothing installed yet),
breaking `config set HOST=...` as literally the first command run.
`load_installed_config_metadata()` only registers config fields.
"""
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
def test_config_subcommands_never_call_on_prepare(monkeypatch, fresh_manager, subcommand, kwargs):
    calls = []
    for container in fresh_manager.containers.values():
        original = container.on_prepare

        def make_spy(name, original=original):
            def spy(*a, **kw):
                calls.append(name)
                return original(*a, **kw)
            return spy
        monkeypatch.setattr(container, "on_prepare", make_spy(container.name))

    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    getattr(ConfigCommand(), subcommand)(**kwargs)

    assert calls == [], f"{subcommand} must not call on_prepare(), but it called: {calls}"
