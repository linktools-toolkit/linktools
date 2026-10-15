#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named deployments diagnose cross-scope changes before mutating services."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from test_lifecycle_rebuild import setup_case


def capture_warnings(manager):
    manager.logger.warning = lambda message, *args: manager.events.append(("warning", message % args))


def warnings(manager):
    return [event[1] for event in manager.events if event[0] == "warning"]


def assert_no_runtime_changes(manager: SimpleNamespace) -> None:
    assert not any(event[0] in ("images", "apply", "stop", "restore", "restart", "down")
                   for event in manager.events)


@pytest.mark.parametrize("action", ["up", "restart"])
def test_partial_start_diagnoses_missing_provider_before_mutation(tmp_path, action):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:new"}}),
    ])
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="Required dependencies are not ready: db.*containers: db"):
        getattr(manager.compose_operations, action)(["app"])
    assert_no_runtime_changes(manager)
    assert not warnings(manager)


def test_unchanged_required_provider_does_not_warn(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:local", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:local"}}),
    ], running=("app", "db"))
    AppliedServiceModels(manager, manager.model).record(("app", "db"))
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    assert not warnings(manager)
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]


@pytest.mark.parametrize("pull", [False, True])
def test_partial_up_diagnoses_missing_provider_before_planning_external_build(tmp_path, pull):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["database"]}}),
        ("storage", {"database": {"image": "database:new", "build": "."}}),
    ])

    def unexpected_image_plan(*args, **kwargs):
        pytest.fail("External prerequisites must be diagnosed before image preparation")

    manager.image_preparer.plan = unexpected_image_plan
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="Required dependencies are not ready: database.*containers: storage"):
        manager.compose_operations.up(["app"], pull=pull)
    assert_no_runtime_changes(manager)
    assert not warnings(manager)


def test_ready_required_service_does_not_warn_about_building(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["database"]}}),
        ("storage", {"database": {"image": "database:new", "build": "."}}),
    ], running=("database",))
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    assert not warnings(manager)
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("binding", [
    {"network_mode": "service:db"},
    {"depends_on": {"db": {"restart": True}}},
])
def test_partial_start_diagnoses_unselected_dependent_before_mutation(tmp_path, binding, action):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": dict(binding, image="net:new")}),
    ], running=("db", "net"))
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="Running dependents require a joint operation.*net"):
        getattr(manager.compose_operations, action)(["db"])
    assert_no_runtime_changes(manager)
    assert not warnings(manager)


@pytest.mark.parametrize("bound", [False, True])
def test_unselected_generated_consumer_is_untouched_without_exposing_configuration(tmp_path, bound):
    proxy = {"image": "proxy:new", "environment": {"TOKEN": "private-value"}}
    if bound:
        proxy["network_mode"] = "service:app"
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("proxy", {"proxy": proxy}),
    ], running=("proxy",))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="proxy"),)
    previous = AppliedServiceModels(manager, manager.model).previous["proxy"]
    capture_warnings(manager)
    if bound:
        with pytest.raises(ContainerError, match="Running dependents require a joint operation.*proxy") as caught:
            manager.compose_operations.up(["app"])
        assert "TOKEN" not in str(caught.value) and "private-value" not in str(caught.value)
        assert_no_runtime_changes(manager)
    else:
        manager.compose_operations.up(["app"])
        assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
        assert not any(event[0] in ("restore", "restart", "stop") for event in manager.events)
    assert not warnings(manager)
    assert AppliedServiceModels(manager, manager.model).previous["proxy"] == previous


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("names", [None, ["db", "net"]])
def test_full_and_joint_start_do_not_emit_partial_up_warning(tmp_path, action, names):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
    ], running=("db", "net"))
    capture_warnings(manager)
    getattr(manager.compose_operations, action)(names)
    assert not warnings(manager)
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db", "net"]


def test_partial_up_diagnoses_collateral_recovery_requirement_before_application(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
        ("fail", {"fail": {"image": "fail:new"}}),
    ], running=("db", "net", "fail"))
    manager.compose_runner.fail = "fail"
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="Running dependents require a joint operation.*net"):
        manager.compose_operations.up(["db", "fail"])
    assert_no_runtime_changes(manager)
    assert not warnings(manager)


@pytest.mark.parametrize("names", [None, ["db", "net", "fail"]])
def test_repeated_joint_rebind_during_recovery_does_not_emit_collateral_warning(tmp_path, names):
    def mount(path):
        return [{"type": "bind", "source": str(path), "target": "/config"}]

    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new", "volumes": mount(tmp_path / "new")}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
        ("fail", {"fail": {"image": "fail:new", "volumes": mount(tmp_path / "new")}}),
    ], running=("db", "net", "fail"))
    old = deepcopy(manager.model)
    old["services"]["db"]["volumes"] = mount(tmp_path / "old")
    old["services"]["fail"]["volumes"] = mount(tmp_path / "old")
    AppliedServiceModels(manager, old).record(("db", "net", "fail"))
    manager.integration_snapshot["db"] = (SimpleNamespace(consumer="net"),)
    manager.compose_runner.fail = "fail"
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="apply failed fail"):
        manager.compose_operations.up(names)
    assert sum(event[:2] == ("apply", "net") for event in manager.events) == 2
    assert not warnings(manager)


@pytest.mark.parametrize("names", [None, ["app", "db"]])
def test_recovery_stops_explicitly_selected_new_provider_without_collateral_warning(tmp_path, names):
    shared = [{"type": "bind", "source": str(tmp_path / "shared"), "target": "/config"}]
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new", "volumes": shared}}),
        ("app", {"app": {"image": "app:new", "depends_on": ["db"], "volumes": shared}}),
    ])
    manager.compose_runner.fail = "app"
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="apply failed app"):
        manager.compose_operations.up(names)
    assert ("stop", ("app", "db")) in manager.events
    assert not warnings(manager)


@pytest.mark.parametrize("names", [None, ["app"]])
def test_zero_scale_absent_provider_requires_explicit_selection(tmp_path, names):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:new", "scale": 0}}),
    ])
    capture_warnings(manager)
    manager.compose_runner.wait_service_ready = lambda context, service, model=None: service != "db"
    if names:
        with pytest.raises(ContainerError, match="Required dependencies are not ready: db"):
            manager.compose_operations.up(names)
        assert_no_runtime_changes(manager)
    else:
        manager.compose_operations.up()
    assert not warnings(manager)
