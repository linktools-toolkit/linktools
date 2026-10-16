#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""get_config accepts a ConfigField directly, not just a str.

Lets a container reference a one-off field (e.g. a nginx domain with its own
fallback) directly at the call site instead of also declaring a separate
`fields` property purely to give the field a home.
"""
import os
from typing import TYPE_CHECKING

import pytest

import _harness
from linktools.cntr import BaseContainer
from linktools.core import ConfigField, ConfigRef, LazyProvider

if TYPE_CHECKING:
    from linktools.cntr import ContainerManager


@pytest.fixture
def field_key_repo(tmp_path):
    repo = tmp_path / "repo"
    c_dir = repo / "100-c"
    c_dir.mkdir(parents=True)
    (c_dir / "container.py").write_text(
        "from linktools.core import ConfigField\n"
        "from linktools.cntr import BaseContainer\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def resolved_default(self):\n"
        "        return self.get_config(ConfigField(name='ONE_OFF', default='fallback'))\n"
        "    @property\n"
        "    def resolved_twice(self):\n"
        "        field = ConfigField(name='SHARED', default='shared-default')\n"
        "        first = self.get_config(field)\n"
        "        second = self.get_config(field)\n"
        "        return first, second\n"
    )
    return repo


def _fresh_standalone_manager(tmp_path):
    # A manager built from scratch (not the `fresh_manager` fixture, which
    # already memoizes `.containers` over just the builtins before a test
    # gets a chance to add its own repo).
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


def _install_single(tmp_path, repo, name):
    manager = _fresh_standalone_manager(tmp_path)
    manager.repos.add(str(repo))
    manager.installed_state.add(name)
    manager.prepare_installed_containers()
    return manager, manager.containers[name]


def test_get_config_accepts_configfield_and_uses_its_default(tmp_path, field_key_repo):
    _, c = _install_single(tmp_path, field_key_repo, "c")
    assert c.resolved_default == "fallback"


def test_get_config_field_is_readable_via_its_name_afterwards(tmp_path, field_key_repo):
    manager, c = _install_single(tmp_path, field_key_repo, "c")
    c.resolved_default  # defines "ONE_OFF" into the schema as a side effect
    # "c" is loaded from a third-party repo, so its config is that
    # repo's own Config, not the manager's shared one -- read it back off
    # the same Config the field was defined into.
    assert c.config.get("ONE_OFF") == "fallback"


def test_repeated_field_definition_is_idempotent(tmp_path, field_key_repo):
    _, c = _install_single(tmp_path, field_key_repo, "c")
    first, second = c.resolved_twice
    assert first == second == "shared-default"


def test_configfield_without_name_raises(fresh_manager):
    field = ConfigField(default="x")
    with pytest.raises(ValueError):
        fresh_manager.containers["portainer"]._resolve_config_key(field)


def test_get_config_still_accepts_plain_string_key(fresh_manager):
    assert fresh_manager.config.get("DOCKER_HOST") == fresh_manager.containers["portainer"].get_config("DOCKER_HOST")


def test_deferred_config_ref_reads_an_explicitly_defined_field(fresh_manager: "ContainerManager") -> None:
    container = fresh_manager.containers["portainer"]
    calls = []
    field = ConfigField(name="DEFERRED_FIELD", provider=LazyProvider(lambda config: calls.append(config) or "42"))
    container.config.define(field)
    reference = ConfigRef(field.name, type=int)
    assert calls == []
    assert container.config.resolve(reference) == 42
    assert len(calls) == 1
    container.config.set(field.name, "43")
    assert container.config.resolve(reference) == 43
    assert len(calls) == 1


def test_container_has_no_get_config_later_alias() -> None:
    assert not hasattr(BaseContainer, "get_config_later")


def test_nginx_domain_provider_via_configfield_key(fresh_manager):
    from linktools.cntr.ext import Nginx

    portainer = fresh_manager.containers["portainer"]
    field = ConfigField(name="SOME_DOMAIN", provider=Nginx.domain(portainer, "x"))
    assert isinstance(field.provider, LazyProvider)
    assert portainer.get_config(field) == "_"  # NGINX_ROOT_DOMAIN falls back to "_" in tests


def test_third_party_nginx_domain_provider_uses_builtin_schema(tmp_path, capsys, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "container.py").write_text(
        "from linktools.cntr import BaseContainer\n"
        "from linktools.cntr.ext import Nginx\n"
        "class Container(BaseContainer):\n"
        "    @property\n"
        "    def fields(self):\n"
        "        return {'AIONUI_DOMAIN': Nginx.domain(self)}\n",
        encoding="utf-8",
    )
    manager, container = _install_single(tmp_path, repo, "repo")
    manager.installed_state.add("nginx")

    manager.prepare_installed_containers()

    assert container.config.get("AIONUI_DOMAIN") == "_"

    import linktools.cntr.commands._shared as shared
    from linktools.cntr.commands.config import ConfigCommand
    monkeypatch.setattr(shared, "manager", manager)
    ConfigCommand().on_command_list(names=["repo"], show_secret=True)
    assert "AIONUI_DOMAIN=_" in capsys.readouterr().out
