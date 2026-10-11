#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoped deployments announce additional runtime changes before executing them."""
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


def test_partial_up_warns_before_starting_required_provider(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:new"}}),
    ])
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    messages = warnings(manager)
    assert len(messages) == 1
    assert "will start service db" in messages[0]
    assert "required service is not running" in messages[0]
    assert manager.events.index(("warning", messages[0])) < manager.events.index(("apply", "db", True))


def test_unchanged_required_provider_does_not_warn(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:local", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:local"}}),
    ], running=("app", "db"))
    AppliedServiceModels(manager, manager.model).record(("app", "db"))
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    assert not warnings(manager)


@pytest.mark.parametrize("pull,reason", [(False, "build inputs changed"), (True, "image refresh")])
def test_partial_up_warns_before_building_required_service_in_another_container(tmp_path, pull, reason):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["database"]}}),
        ("storage", {"database": {"image": "database:new", "build": "."}}),
    ])
    manager.image_preparer.plan = lambda *args, **kwargs: SimpleNamespace(pull=(), build=("database",))
    capture_warnings(manager)
    manager.compose_operations.up(["app"], pull=pull)
    messages = warnings(manager)
    assert len(messages) == 2
    assert "will build service database (container storage)" in messages[0]
    assert reason in messages[0]
    assert manager.events.index(("warning", messages[0])) < manager.events.index(("images", ("database",)))


def test_required_service_without_planned_build_does_not_warn_about_building(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["database"]}}),
        ("storage", {"database": {"image": "database:new", "build": "."}}),
    ])
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    assert not any("will build" in message for message in warnings(manager))


@pytest.mark.parametrize("binding,action", [
    ({"network_mode": "service:db"}, "recreate"),
    ({"depends_on": {"db": {"restart": True}}}, "restart"),
])
def test_partial_up_warns_before_changing_unselected_dependent(tmp_path, binding, action):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": dict(binding, image="net:new")}),
    ], running=("db", "net"))
    capture_warnings(manager)
    manager.compose_operations.up(["db"])
    messages = warnings(manager)
    assert len(messages) == 1
    assert "will {} service net".format(action) in messages[0]
    event = ("restore", ("net",)) if action == "recreate" else ("restart", "net")
    assert manager.events.index(("warning", messages[0])) < manager.events.index(event)


def test_generated_consumer_warning_does_not_include_configuration_values(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("proxy", {"proxy": {"image": "proxy:new", "environment": {"TOKEN": "private-value"}}}),
    ], running=("proxy",))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="proxy"),)
    capture_warnings(manager)
    manager.compose_operations.up(["app"])
    messages = warnings(manager)
    assert len(messages) == 1 and "will recreate service proxy" in messages[0]
    assert "configuration changed" in messages[0]
    assert "TOKEN" not in messages[0] and "private-value" not in messages[0]


@pytest.mark.parametrize("action,names", [("up", None), ("restart", ["db"])])
def test_full_up_and_partial_restart_do_not_emit_partial_up_warning(tmp_path, action, names):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
    ], running=("db", "net"))
    capture_warnings(manager)
    getattr(manager.compose_operations, action)(names)
    assert not warnings(manager)


def test_recovery_warns_before_newly_discovered_collateral_action(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
        ("fail", {"fail": {"image": "fail:new"}}),
    ], running=("db", "net", "fail"))
    manager.compose_runner.fail = "fail"
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="apply failed fail"):
        manager.compose_operations.up(["db", "fail"])
    messages = warnings(manager)
    assert len(messages) == 1 and "will restore service net" in messages[0]
    assert "deployment recovery" in messages[0]
    assert manager.events.index(("warning", messages[0])) < manager.events.index(("restore", ("net",)))


def test_repeated_rebind_during_recovery_does_not_duplicate_warning(tmp_path):
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
        manager.compose_operations.up(["db", "fail"])
    assert sum(event[:2] == ("apply", "net") for event in manager.events) == 2
    messages = warnings(manager)
    assert len(messages) == 1 and "will recreate service net" in messages[0]


def test_recovery_warns_before_stopping_new_collateral_provider(tmp_path):
    shared = [{"type": "bind", "source": str(tmp_path / "shared"), "target": "/config"}]
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new", "volumes": shared}}),
        ("app", {"app": {"image": "app:new", "depends_on": ["db"], "volumes": shared}}),
    ])
    manager.compose_runner.fail = "app"
    capture_warnings(manager)
    with pytest.raises(ContainerError, match="apply failed app"):
        manager.compose_operations.up(["app"])
    messages = warnings(manager)
    assert len(messages) == 2
    assert "will start service db" in messages[0]
    assert "will stop service db" in messages[1]
    assert manager.events.index(("warning", messages[1])) < manager.events.index(("stop", ("app", "db")))


def test_zero_scale_absent_provider_is_not_announced_as_a_start(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["db"]}}),
        ("db", {"db": {"image": "db:new", "scale": 0}}),
    ])
    capture_warnings(manager)
    manager.compose_runner.wait_service_ready = lambda context, service, model=None: service != "db"
    manager.compose_operations.up(["app"])
    assert not warnings(manager)
