#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native build and recovery relationships cannot bypass explicit operation scope."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr import OperationContext, ContainerError
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.runtime.compose import ComposeRunner
from test_explicit_operation_scope import instrument_images, assert_no_image_or_service_mutation
from test_lifecycle_rebuild import setup_case


@pytest.mark.parametrize("contexts", [{"base": "service:base"}, ["base=service:base"]])
def test_native_build_context_requires_explicit_external_service(tmp_path, contexts):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "build": {"context": "/tmp/app", "additional_contexts": contexts}}}),
        ("base", {"base": {"image": "base:new", "build": {"context": "/tmp/base"}}}),
    ])
    instrument_images(manager)
    with pytest.raises(ContainerError, match="Build contexts.*base"):
        manager.compose_operations.up(["app"], pull=True)
    assert not any(event[0] in ("pull", "build", "apply", "stop") for event in manager.events)
    assert all("base:new" not in event[1] for event in manager.events if event[0] == "inspect-images")


def test_unreferenced_provider_does_not_block_or_expand_application(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}}),
                                    ("provider", {"provider": {"image": "provider:new"}})])
    instrument_images(manager)
    manager.compose_operations.up(["app"], pull=True)
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert [event[1] for event in manager.events if event[0] == "pull"] == [("app",)]
    assert all("provider:new" not in event[1] for event in manager.events if event[0] == "inspect-images")


def test_pending_external_zero_scale_does_not_bypass_namespace_readiness(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "network_mode": "service:db"}}),
        ("db", {"db": {"image": "db:new", "scale": 0}}),
    ])
    instrument_images(manager)
    with pytest.raises(ContainerError, match="dependencies.*db"):
        manager.compose_operations.restart(["app"], pull=True)
    assert_no_image_or_service_mutation(manager)


def test_selected_previous_namespace_provider_must_be_ready_for_rollback(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}}),
                                    ("db", {"db": {"image": "db:new"}})], running=("app",))
    previous = deepcopy(manager.model)
    previous["services"]["app"]["network_mode"] = "service:db"
    AppliedServiceModels(manager, previous).record(("app",))
    instrument_images(manager)
    with pytest.raises(ContainerError, match="dependencies.*db"):
        manager.compose_operations.up(["app"], pull=True)
    assert_no_image_or_service_mutation(manager)


@pytest.mark.parametrize("action", ["pull", "build", "stop", "apply_service", "apply_services",
                                     "restart_service", "apply_saved_services"])
def test_runtime_and_recovery_commands_reject_outside_targets_before_process_creation(action):
    runner = ComposeRunner(SimpleNamespace())
    context = OperationContext(is_full_project=False, target_services=("app",))
    arguments = {"pull": (("outside",),), "build": (SimpleNamespace(services=("outside",)),),
                 "stop": (("outside",),), "apply_service": ("outside",),
                 "apply_services": (("outside",),), "restart_service": ("outside",),
                 "apply_saved_services": (("outside",), {})}
    with pytest.raises(ContainerError, match="unselected services: outside"):
        getattr(runner, action)(context, *arguments[action])


def test_shared_native_completed_dependency_overrides_incomplete_last_fragment(tmp_path):
    manager = setup_case(tmp_path, [
        ("base", {"app": {"image": "app:new", "depends_on": {
            "init": {"condition": "service_completed_successfully"}}}}),
        ("overlay", {"app": {"image": "app:new", "depends_on": {"init": {"required": True}}}}),
        ("initializer", {"init": {"image": "init:new"}}),
    ], running=("init",))
    instance = manager.docker_inspector.get_project_state(None).service_instances("init")[0]
    instance.state, instance.exit_code = "exited", 0
    manager.model["services"]["app"]["depends_on"]["init"]["condition"] = "service_completed_successfully"
    instrument_images(manager)
    manager.compose_operations.up(["base", "overlay"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]


def test_native_build_context_transitive_dependency_stays_in_explicit_scope(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "build": {
            "context": "/tmp/app", "additional_contexts": {"base": "service:base"}}}}),
        ("base", {"base": {"image": "base:new", "build": {
            "context": "/tmp/base", "additional_contexts": {"third": "service:third"}}}}),
        ("third", {"third": {"image": "third:new"}}),
    ])
    instrument_images(manager)
    with pytest.raises(ContainerError, match="Build contexts.*third"):
        manager.compose_operations.up(["app", "base"], pull=True)
    assert not any(event[0] in ("pull", "build", "apply", "stop") for event in manager.events)
