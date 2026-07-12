#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr list``: the very first command a user runs against a project,
so it must not depend on some other command having already registered every
installed container's own config defaults into env_config."""
import os

import pytest

import _harness

import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared


def _make_unprepared_manager(tmp_path):
    """Like ``_harness.make_manager``, but stops short of calling
    ``prepare_installed_containers()`` -- reproducing the state of a manager
    that has never rendered anything yet, exactly like a fresh `ct-cntr list`
    as the first command run against a project."""
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()

    data_path = str(tmp_path / "data")
    temp_path = str(tmp_path / "temp")
    os.environ["LINKTOOLS_PATH"] = os.path.dirname(data_path) or data_path
    os.environ["LINKTOOLS_DATA_PATH"] = data_path
    os.environ["LINKTOOLS_TEMP_PATH"] = temp_path

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    environ = Environ()
    manager = ContainerManager(environ, name="aio")
    manager.installed_state.add(*manager.containers.keys())
    return manager


def test_list_does_not_crash_on_a_never_prepared_manager(tmp_path, monkeypatch):
    """Regression: on_command_list queried running state (which renders each
    container's own compose/Dockerfile template through DockerInspector)
    without first registering any container's config defaults, so a builtin
    like lldap -- whose own compose.yml references its own {{ LLDAP_PORT }}
    -- failed with "'LLDAP_PORT' is undefined" on a cold project."""
    manager = _make_unprepared_manager(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    cntr_main.command.on_command_list()  # must not raise


def test_list_registers_container_configs_before_checking_status(tmp_path, monkeypatch):
    manager = _make_unprepared_manager(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    cntr_main.command.on_command_list()

    # LLDAP_PORT is declared only by lldap's own `configs`; it only becomes a
    # resolvable env_config key once that container's configs are registered.
    assert manager.env_config.get("LLDAP_PORT", type=int) == 0


def test_list_never_queries_the_docker_runtime(tmp_path, monkeypatch):
    """`list` must stay a fast, local-only read: it shows persisted running
    state, never live state -- so it must never shell out to `docker`/
    `docker compose` at all. Use `ct-cntr status` for a live query."""
    manager = _make_unprepared_manager(tmp_path)
    monkeypatch.setattr(cntr_shared, "manager", manager)

    def _fail(*args, **kwargs):
        raise AssertionError("list must not query the Docker runtime")

    monkeypatch.setattr(manager.runtime, "create_docker_process", _fail)
    monkeypatch.setattr(manager.runtime, "create_docker_compose_process", _fail)

    cntr_main.command.on_command_list()  # must not raise
