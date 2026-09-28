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
