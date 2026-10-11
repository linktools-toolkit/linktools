#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loaded containers own lifecycle capabilities without companion registries."""
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from linktools.cntr.registry.loader import ContainerLoader
from linktools.cntr.repo.context import RepositoryConfigContext
from test_loader_structured_errors import _fresh_standalone_manager

if TYPE_CHECKING:
    from linktools.cntr import ContainerManager


def _load(tmp_path, source):
    root = tmp_path / "100-example"
    root.mkdir()
    (root / "container.py").write_text(textwrap.dedent(source), encoding="utf-8")
    manager = _fresh_standalone_manager(tmp_path)
    errors = []
    repository = RepositoryConfigContext(root_path=root, file_config=None, url=None, builtin=True)
    containers = list(ContainerLoader(manager)._load_one(str(root), repository, errors))
    return containers, errors


def test_loading_preserves_native_capabilities_without_invoking_runtime_lifecycle(tmp_path: Path) -> None:
    containers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            def on_init(self):
                self.initialized = True

            def fail(self, *args):
                raise AssertionError("Loading must not invoke runtime lifecycle")

            on_prepare = fail
            on_starting = fail
            on_check = fail
            get_runtime_requirements = fail
    """)
    assert errors == []
    assert len(containers) == 1
    container = containers[0]
    assert container.initialized
    assert not hasattr(container, "on_prepare_config")
    assert not hasattr(container, "on_service_started")
    assert not hasattr(container, "integration_consumer")
    assert not hasattr(container.manager, "integration_consumers")


def test_plain_container_inherits_non_generated_lifecycle_defaults(tmp_path: Path) -> None:
    containers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            pass
    """)
    assert errors == []
    assert len(containers) == 1
    container = containers[0]
    assert container.get_runtime_requirements({"example"}) == {}
    assert not hasattr(container, "render_config")
    assert not hasattr(container, "apply_config")


@pytest.mark.parametrize("method", ["__init__", "on_init"])
def test_native_initialization_failure_does_not_publish_container(tmp_path: Path, method: str) -> None:
    containers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            def %s(self, *args):
                raise ValueError("broken native initialization")
    """ % method)
    assert containers == []
    assert len(errors) == 1
    assert errors[0].expected_name == "example"
    assert "broken native initialization" in errors[0].message


def test_load_all_returns_capable_containers_without_companion_mapping(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.capabilities.cntr import __cap_cntr__

    assets = tmp_path / "assets"
    for name in ("100-example", "101-plain"):
        root = assets / name
        root.mkdir(parents=True)
        (root / "container.py").write_text(textwrap.dedent("""\
            from linktools.cntr import BaseContainer

            class Container(BaseContainer):
                pass
        """), encoding="utf-8")
    manager = _fresh_standalone_manager(tmp_path)
    monkeypatch.setattr(__cap_cntr__, "get_asset_path", lambda *parts: assets)
    result = manager.loader.load_all()
    assert result.errors == []
    assert {container.name for container in result.containers} == {"example", "plain"}
    assert not hasattr(result, "consumers")
    assert not hasattr(manager, "integration_consumers")


def test_builtin_containers_do_not_implement_removed_config_lifecycle(fresh_manager: "ContainerManager") -> None:
    names = ("nginx", "lldap", "authelia", "flare")
    for name in names:
        container = fresh_manager.containers[name]
        assert not hasattr(container, "on_prepare_config")
        assert not hasattr(container, "apply_config")
        assert callable(container.on_starting)
        assert callable(container.on_check)
    assert not hasattr(fresh_manager, "generated_configs")


def test_external_prepare_runs_after_configs_before_selected_services(tmp_path, monkeypatch):
    containers, errors = _load(tmp_path, '''\
        from linktools.cntr import BaseContainer
        from linktools.cntr.lifecycle import HookPhase

        class Container(BaseContainer):
            def on_init(self):
                self.events = []
                self.services = {}
                self.requirements = ()

            @property
            def dependencies(self):
                return self.requirements

            @property
            def configs(self):
                return {"PREPARED_SERVICE": "example-service"}

            def on_prepare(self):
                self.events.append("prepare")
                self.requirements = ("prepared-provider",)
                name = self.get_config("PREPARED_SERVICE")
                self.services[name] = {"image": "example:local"}
                self.add_start_hook(("prepared",), lambda: self.events.append("hook"))
    ''')
    assert errors == []
    container = containers[0]
    manager = container.manager
    monkeypatch.setattr(manager, "containers", {container.name: container})
    monkeypatch.setattr(manager.installed_state, "get", lambda **kwargs: containers)
    assert manager.load_installed_config_metadata() == containers
    assert manager.compose_operations.select([container.name], for_start=True).services == ()
    assert container.events == []
    def resolve(selected):
        assert selected == (container,)
        assert container.dependencies == ("prepared-provider",)
        assert container.events == ["prepare"]
        return selected

    monkeypatch.setattr(manager.resolver, "resolve_dependencies", resolve)
    selected = manager.compose_operations.select([container.name], with_dependencies=True, prepare=True)
    assert selected.services == ("example-service",)
    assert container.events == ["prepare"]
    from linktools.cntr.lifecycle import HookPhase
    container.hooks.call(HookPhase.BEFORE_START)
    assert container.events == ["prepare", "hook"]


def test_external_prepare_failure_propagates_before_model_selection(tmp_path, monkeypatch):
    containers, errors = _load(tmp_path, '''\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            def on_prepare(self):
                raise ValueError("prepare failed")

            @property
            def services(self):
                raise AssertionError("services read before successful preparation")
    ''')
    assert errors == []
    manager = containers[0].manager
    monkeypatch.setattr(manager, "load_installed_config_metadata", lambda: containers)
    with pytest.raises(ValueError, match="prepare failed"):
        manager.prepare_installed_containers()
    with pytest.raises(ValueError, match="prepare failed"):
        manager.compose_operations.select([containers[0].name], prepare=True)


def test_exec_nested_operation_preserves_each_prepare_entry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from linktools.cntr.commands import _shared
    from linktools.cntr.commands.exec_ import ExecCommand

    containers, errors = _load(tmp_path, '''\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            def on_init(self):
                self.prepared = 0
                self.services = {"example": {"image": "example:local"}}

            def on_prepare(self):
                self.prepared += 1
    ''')
    assert errors == []
    container = containers[0]
    manager = container.manager
    monkeypatch.setattr(manager, "load_installed_config_metadata", lambda: containers)
    monkeypatch.setattr(_shared, "manager", manager)
    command = ExecCommand()
    args = SimpleNamespace(exec_name="example", exec_args=[])
    monkeypatch.setattr(ExecCommand, "_subparser", property(lambda self: SimpleNamespace(parse_args=lambda argv: args)))

    def run(parsed):
        assert container.prepared == 1
        manager.compose_operations.select([container.name], prepare=True)
        assert container.prepared == 2
        return 7

    monkeypatch.setattr(command, "parse_subcommand", lambda parsed: SimpleNamespace(run=run))
    assert command.run(args) == 7
