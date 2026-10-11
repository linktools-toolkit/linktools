#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scope and rollback regressions for the shared Compose operation path."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from test_lifecycle_rebuild import setup_case


def _applied(manager):
    return [entry[1] for entry in manager.events if entry[0] == "apply"]


def test_partial_up_does_not_prepare_unrelated_images(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("other", {"running": {"image": "running:new"}, "stopped": {"image": "stopped:new"}}),
    ], running=("running",))
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["app"]
    assert ("prepare", "other") not in manager.events
    assert ("check-callback", "other") not in manager.events
    assert ("images", ("app",)) in manager.events


def test_running_declared_consumer_does_not_activate_stopped_sibling(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("consumer", {"running": {"image": "running:new"}, "stopped": {"image": "stopped:new"}}),
    ], running=("running",))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="consumer"),)
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["app", "running"]
    assert not any("stopped" in entry[1] for entry in manager.events if entry[0] == "images")


def test_unrelated_running_sidecar_does_not_apply_pending_change(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("other", {"running": {"image": "running:new"}, "stopped": {"image": "stopped:new"}}),
    ], running=("running",))
    manager.model["services"]["running"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["app"]


def test_unrelated_pending_failure_does_not_interrupt_explicit_start(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("other", {"running": {"image": "running:new"}}),
    ], running=("running",))
    manager.compose_runner.fail = "running"
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["app"]


def test_selected_service_starts_stopped_dependency_first(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:new"}}),
    ])
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["db", "app"]


def test_service_dag_interleaves_owners_by_compose_edges(tmp_path):
    manager = setup_case(tmp_path, [
        ("one", {"a": {"image": "a:new"}, "c": {"image": "c:new", "depends_on": ["b"]}}),
        ("two", {"b": {"image": "b:new", "depends_on": ["a"]}}),
    ])
    manager.compose_operations.up(["one"])
    assert _applied(manager) == ["a", "b", "c"]


def test_preparation_is_observed_by_final_model(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})])
    def prepare(context):
        manager.model["services"]["app"]["environment"] = {"GENERATED": "true"}
    manager.containers["app"].on_starting = prepare
    manager.compose_operations.up(["app"])
    model = yaml.safe_load(AppliedServiceModels(manager, manager.model).previous["app"])
    assert model["services"]["app"]["environment"] == {"GENERATED": "true"}


def test_failed_application_restores_previous_config_and_image(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    old = deepcopy(manager.model)
    old["services"]["app"]["environment"] = {"VALUE": "old"}
    AppliedServiceModels(manager, old).record(("app",))
    manager.model["services"]["app"]["environment"] = {"VALUE": "new"}
    manager.compose_runner.fail = "app"
    with pytest.raises(ContainerError, match="apply failed"):
        manager.compose_operations.up(["app"])
    assert any(entry[0] == "restore" for entry in manager.events)
    restored = yaml.safe_load(AppliedServiceModels(manager, manager.model).previous["app"])
    assert restored["services"]["app"]["environment"] == {"VALUE": "old"}


def test_unchanged_running_target_does_not_recreate(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    AppliedServiceModels(manager, manager.model).record(("app",))
    manager.compose_operations.up(["app"])
    assert ("apply", "app", False) in manager.events


def test_changed_runtime_environment_recreates_only_selected_service(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("other", {"other": {"image": "other:new"}}),
    ], running=("app", "other"))
    AppliedServiceModels(manager, manager.model).record(("app", "other"))
    manager.model["services"]["app"]["environment"] = {"UPDATED": "yes"}
    manager.compose_operations.up(["app"])
    assert _applied(manager) == ["app"]
    assert ("apply", "app", True) in manager.events
