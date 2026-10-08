#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consumer companions belong to loaded modules, not container instances."""
import textwrap

import pytest

from linktools.cntr.registry.loader import ContainerLoader
from linktools.cntr.repo.context import RepositoryConfigContext
from test_loader_structured_errors import _fresh_standalone_manager


def _load(tmp_path, source):
    root = tmp_path / "100-example"
    root.mkdir()
    (root / "container.py").write_text(textwrap.dedent(source), encoding="utf-8")
    manager = _fresh_standalone_manager(tmp_path)
    errors, consumers = [], {}
    repository = RepositoryConfigContext(root_path=root, file_config=None, url=None, builtin=True)
    containers = list(ContainerLoader(manager)._load_one(str(root), repository, errors, consumers))
    return containers, consumers, errors


def test_explicit_consumer_export_binds_selected_container_without_running_hooks(tmp_path):
    containers, consumers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer
        from linktools.cntr.integration import IntegrationConsumer

        class Container(BaseContainer):
            pass

        class Consumer(IntegrationConsumer):
            def on_prepare(self, context):
                raise AssertionError("Loading must not prepare generated configuration")
    """)
    assert errors == []
    assert list(consumers) == ["example"]
    assert consumers["example"].container is containers[0]
    assert "integration_consumer" not in containers[0].__dict__


def test_unexported_consumer_subclass_is_not_discovered(tmp_path):
    containers, consumers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer
        from linktools.cntr.integration import IntegrationConsumer

        class Container(BaseContainer):
            pass

        class Helper(IntegrationConsumer):
            pass
    """)
    assert len(containers) == 1
    assert consumers == {}
    assert errors == []


@pytest.mark.parametrize("export", ["None", "object()", "object"])
def test_invalid_consumer_export_is_a_structured_load_error(tmp_path, export):
    containers, consumers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer
        class Container(BaseContainer):
            pass
        Consumer = %s
    """ % export)
    assert containers == []
    assert consumers == {}
    assert len(errors) == 1
    assert errors[0].expected_name == "example"
    assert "Consumer must subclass IntegrationConsumer" in errors[0].message


def test_consumer_constructor_failure_does_not_publish_container(tmp_path):
    containers, consumers, errors = _load(tmp_path, """\
        from linktools.cntr import BaseContainer
        from linktools.cntr.integration import IntegrationConsumer

        class Container(BaseContainer):
            pass

        class Consumer(IntegrationConsumer):
            def __init__(self, container):
                raise ValueError("broken companion")
    """)
    assert containers == []
    assert consumers == {}
    assert len(errors) == 1
    assert "broken companion" in errors[0].message


def test_load_all_carries_consumers_in_its_result(tmp_path, monkeypatch):
    from linktools.capabilities.cntr import __cap_cntr__

    assets = tmp_path / "assets"
    root = assets / "100-example"
    root.mkdir(parents=True)
    (root / "container.py").write_text(textwrap.dedent("""\
        from linktools.cntr import BaseContainer
        from linktools.cntr.integration import IntegrationConsumer

        class Container(BaseContainer):
            pass

        class Consumer(IntegrationConsumer):
            pass
    """), encoding="utf-8")
    manager = _fresh_standalone_manager(tmp_path)
    monkeypatch.setattr(__cap_cntr__, "get_asset_path", lambda *parts: assets)
    result = manager.loader.load_all()
    assert result.errors == []
    assert len(result.containers) == 1
    assert result.consumers["example"].container is result.containers[0]
