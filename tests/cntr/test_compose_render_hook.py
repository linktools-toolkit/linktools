#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-render hooks change the same Compose model used by plans and writes."""
from typing import TYPE_CHECKING

import yaml

from linktools.cntr.artifacts import collect_candidates
from linktools.cntr.container import BaseContainer
from linktools.cntr.lifecycle import HookPhase

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any
    from linktools.cntr.manager import ContainerManager


def test_compose_render_hooks_run_after_defaults_and_before_consumers(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    (tmp_path / "compose.yml").write_text("""\
services:
  app:
    image: busybox
networks:
  private:
""", encoding="utf-8")

    calls = []
    container = BaseContainer(fresh_manager, tmp_path, name="999-custom")

    def add_labels(compose: "dict[str, Any]") -> None:
        calls.append("labels")
        service = compose["services"]["app"]
        assert service["container_name"] == container.get_service_name("app")
        assert service["hostname"] == "app"
        assert "restart" in service
        assert "logging" in service
        assert compose["networks"]["private"]["name"] == container.get_service_name("private")
        service["labels"] = {"example.enabled": "true"}

    def add_environment(compose: "dict[str, Any]") -> None:
        calls.append("environment")
        service = compose["services"]["app"]
        assert service["labels"] == {"example.enabled": "true"}
        service["environment"] = {"HOOKED": "yes"}

    container.hooks.register(HookPhase.AFTER_COMPOSE_RENDER, add_environment,
                             key="environment", order=10, after=("labels",))
    container.hooks.register(HookPhase.AFTER_COMPOSE_RENDER, add_labels,
                             key="labels", order=100)
    destination = fresh_manager.data_path / "compose" / "custom.yml"

    rendered = container.docker_compose
    assert rendered["services"]["app"]["labels"] == {"example.enabled": "true"}
    assert rendered["services"]["app"]["environment"] == {"HOOKED": "yes"}
    assert not destination.exists()

    candidates = collect_candidates(fresh_manager, [container])
    assert yaml.safe_load(candidates[str(destination)][2]) == rendered
    assert not destination.exists()

    assert container.get_docker_compose_file() == destination
    assert yaml.safe_load(destination.read_text(encoding="utf-8")) == rendered
    assert container.docker_compose is rendered
    assert calls == ["labels", "environment"]


def test_compose_render_hook_skipped_without_compose_template(
        fresh_manager: "ContainerManager", tmp_path: "Path") -> None:
    container = BaseContainer(fresh_manager, tmp_path, name="999-custom")

    def unexpected(compose: "dict[str, Any]") -> None:
        raise AssertionError("No Compose template was rendered")

    container.hooks.register(HookPhase.AFTER_COMPOSE_RENDER, unexpected, key="unexpected")
    assert container.docker_compose is None


def test_cached_compose_defers_operation_values_and_retries_failed_render(fresh_manager, tmp_path):
    import pytest

    from linktools.cntr import OperationContext
    from linktools.cntr.lifecycle import LifecycleDispatcher
    from linktools.core import Config
    from linktools.errors import ConfigNotFoundError

    (tmp_path / "compose.yml").write_text(
        "services:\n  app:\n    image: busybox\n", encoding="utf-8",
    )
    container = BaseContainer(fresh_manager, tmp_path, name="999-deferred")
    values = {"public_ready": False, "credential": None}
    calls = []

    def prepare():
        assert not Config.is_read_only_resolution()
        if values["credential"] is None:
            raise ConfigNotFoundError("fixture credential unavailable")
        calls.append(values["credential"])

    def configure(compose):
        if not values["public_ready"]:
            raise ConfigNotFoundError("fixture public configuration unavailable")
        compose["services"]["app"]["environment"] = {"ENABLED": "true"}
        container.add_start_hook(("deferred",), prepare)

    container.hooks.register(HookPhase.AFTER_COMPOSE_RENDER, configure, key="configure")
    with Config.read_only_resolution():
        with pytest.raises(ConfigNotFoundError, match="public configuration"):
            container.docker_compose
        assert "docker_compose" not in container.__dict__
        values["public_ready"] = True
        model = container.docker_compose
    assert calls == []
    assert model["services"]["app"]["environment"] == {"ENABLED": "true"}
    context = OperationContext(target_containers=(container,))
    dispatcher = LifecycleDispatcher(fresh_manager)
    with pytest.raises(ConfigNotFoundError, match="credential unavailable"):
        with dispatcher.notify_start(context):
            pytest.fail("application entered after failed preparation")
    for value in ("fixture-first", "fixture-second"):
        values["credential"] = value
        with dispatcher.notify_start(context):
            assert container.docker_compose is model
    assert calls == ["fixture-first", "fixture-second"]
