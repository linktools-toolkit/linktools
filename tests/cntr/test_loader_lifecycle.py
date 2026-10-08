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
            generates_config = True
            application_priority = 25
            bootstrap_services = ("example",)

            def on_init(self):
                self.initialized = True

            def fail(self, *args):
                raise AssertionError("Loading must not invoke runtime lifecycle")

            on_prepare_config = fail
            render_config = fail
            validate_config = fail
            apply_config = fail
            render_bootstrap = fail
            on_service_started = fail
            get_runtime_requirements = fail
    """)
    assert errors == []
    assert len(containers) == 1
    container = containers[0]
    assert container.initialized
    assert container.generates_config
    assert container.application_priority == 25
    assert container.bootstrap_services == ("example",)
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
    assert not container.generates_config
    assert container.bootstrap_services == ()
    assert container.get_runtime_requirements({"example"}) == {}
    assert container.generation_label("example", "candidate") is None


@pytest.mark.parametrize("method", ["__init__", "on_init"])
def test_native_initialization_failure_does_not_publish_container(tmp_path: Path, method: str) -> None:
    containers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer

        class Container(BaseContainer):
            generates_config = True

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
    for name, generated in (("100-example", True), ("101-plain", False)):
        root = assets / name
        root.mkdir(parents=True)
        (root / "container.py").write_text(textwrap.dedent("""\
            from linktools.cntr import BaseContainer

            class Container(BaseContainer):
                generates_config = %s
        """ % generated), encoding="utf-8")
    manager = _fresh_standalone_manager(tmp_path)
    monkeypatch.setattr(__cap_cntr__, "get_asset_path", lambda *parts: assets)
    result = manager.loader.load_all()
    assert result.errors == []
    assert len(result.containers) == 2
    assert {container.name for container in result.containers if container.generates_config} == {"example"}
    assert not hasattr(result, "consumers")
    assert not hasattr(manager, "integration_consumers")


def test_builtin_generated_capability_counts_match_actual_loaded_owners(fresh_manager: "ContainerManager") -> None:
    expected = {"nginx", "lldap", "authelia", "flare"}
    assert {name for name, container in fresh_manager.containers.items() if container.generates_config} == expected
    assert set(fresh_manager.generated_configs) == expected
    assert all(container is fresh_manager.containers[name]
               for name, container in fresh_manager.generated_configs.items())
