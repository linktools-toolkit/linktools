#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named operations keep preparation and mutation inside explicit container groups."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.images import ImagePreparer, LocalImage
from test_lifecycle_rebuild import setup_case

if TYPE_CHECKING:
    from collections.abc import Sequence
    from linktools.cntr import OperationContext


def instrument_images(manager: SimpleNamespace) -> None:
    available = {}
    preparer = ImagePreparer(manager)
    manager.operation_models = []

    def inspect(images: "Sequence[str]") -> "dict[str, LocalImage | None]":
        manager.events.append(("inspect-images", tuple(images)))
        return {image: available.get(image) for image in images}

    def pull(context: "OperationContext", services: "Sequence[str]") -> None:
        manager.events.append(("pull", tuple(services)))
        for service in services:
            image = context.compose_model["services"][service]["image"]
            available[image] = LocalImage("sha256:old-" + service, None)

    def build(context: "OperationContext", options: SimpleNamespace) -> None:
        manager.events.append(("build", tuple(options.services)))
        for service in options.services:
            image = context.compose_model["services"][service]["image"]
            available[image] = LocalImage("sha256:old-" + service, None)

    preparer.inspect_images = inspect
    manager.image_preparer = preparer
    manager.compose_runner.pull = pull
    manager.compose_runner.build = build
    native = manager.compose_runner.final_model
    manager.compose_runner.final_model = lambda context, **kwargs: native(context)
    apply_services = manager.compose_runner.apply_services

    def apply(context: "OperationContext", services: "Sequence[str]") -> None:
        manager.operation_models.append(deepcopy(context.compose_model))
        apply_services(context, services)

    manager.compose_runner.apply_services = apply
    for owner in manager.containers.values():
        owner.on_prepare = lambda owner=owner: manager.events.append(("resolve", owner.name))


def application_case(tmp_path: Path, running: "tuple[str, ...]") -> SimpleNamespace:
    manager = setup_case(tmp_path, [
        ("safeline", {"safeline": {"image": "safeline:new", "depends_on": ["nginx"]}}),
        ("sublink", {"sublink": {"image": "sublink:new", "depends_on": ["nginx"]}}),
        ("omniroute", {"omniroute": {"image": "omniroute:new", "depends_on": ["auth"]}}),
        ("nginx", {"nginx": {"image": "nginx:new", "build": {"context": "/tmp/nginx"},
                              "depends_on": ["auth"]}}),
        ("auth", {"auth": {"image": "auth:new", "depends_on": ["lldap", "redis"]}}),
        ("lldap", {"lldap": {"image": "lldap:new"}}),
        ("redis", {"redis": {"image": "redis:new"}}),
    ], running=running)
    manager.containers["safeline"].dependencies = ("nginx",)
    instrument_images(manager)
    return manager


def assert_no_image_or_service_mutation(manager: SimpleNamespace) -> None:
    assert not any(event[0] in ("inspect-images", "pull", "build", "images", "apply", "stop",
                               "restart", "restore") for event in manager.events)


@pytest.mark.parametrize("names", [
    ("safeline",), ("sublink",), ("omniroute",), ("safeline", "sublink", "omniroute"),
])
@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("force_pull", [False, True])
def test_named_operations_never_prepare_build_pull_or_apply_external_services(
        tmp_path: Path, names: "tuple[str, ...]", action: str, force_pull: bool) -> None:
    external = ("nginx", "auth", "lldap", "redis")
    running = external + names if action == "restart" else external
    manager = application_case(tmp_path, running=running)
    original = deepcopy(manager.model)
    previous = AppliedServiceModels(manager, manager.model).previous
    snapshots = {name: previous[name] for name in external}
    external_entries = {path: deepcopy(entry) for path, entry in manager.artifact_index.load().items()
                        if entry.get("container") in external}

    getattr(manager.compose_operations, action)(names, pull=force_pull)

    assert [event[1] for event in manager.events if event[0] == "apply"] == list(names)
    assert [event[1] for event in manager.events if event[0] == "pull"] == [names]
    assert not any(event[0] == "build" for event in manager.events)
    for phase in ("resolve", "prepare", "check-callback", "after-callback"):
        assert {event[1] for event in manager.events if event[0] == phase} == set(names)
    selected_images = {name + ":new" for name in names}
    assert all(set(event[1]) <= selected_images for event in manager.events if event[0] == "inspect-images")
    if action == "restart":
        assert [event[1] for event in manager.events if event[0] == "stop"] == [names]
    else:
        assert not any(event[0] == "stop" for event in manager.events)
    current = AppliedServiceModels(manager, manager.model).previous
    assert {name: current[name] for name in external} == snapshots
    assert all(manager.artifact_index.load()[path] == entry for path, entry in external_entries.items())
    assert len(manager.operation_models) == 1
    assert {name: manager.operation_models[0]["services"][name] for name in external} == {
        name: original["services"][name] for name in external}
    assert manager.model == original


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("condition,state,health", [
    ("service_started", None, None),
    ("service_started", "exited", None),
    ("service_healthy", "running", "unhealthy"),
    ("service_completed_successfully", "exited", None),
])
def test_unavailable_external_dependency_fails_before_image_work_or_stops(
        tmp_path: Path, action: str, condition: str, state: "str | None", health: "str | None") -> None:
    groups = [
        ("app", {"app": {"image": "app:new", "depends_on": {"db": {"condition": condition}}}}),
        ("database", {"db": {"image": "db:new"}}),
    ]
    manager = setup_case(tmp_path, groups, running=("app", "db") if state else ("app",))
    if state:
        instance = manager.docker_inspector.get_project_state(None).service_instances("db")[0]
        instance.state, instance.health, instance.exit_code = state, health, 1 if state == "exited" else None
    instrument_images(manager)

    with pytest.raises(ContainerError, match="Required dependencies are not ready.*db.*database"):
        getattr(manager.compose_operations, action)(["app"], pull=True)

    assert_no_image_or_service_mutation(manager)


@pytest.mark.parametrize("action", ["up", "restart"])
def test_named_container_keeps_all_its_services_in_dependency_order(tmp_path: Path, action: str) -> None:
    selected = ("safeline-mgt", "safeline-db")
    manager = setup_case(tmp_path, [
        ("safeline", {
            "safeline-mgt": {"image": "safeline-mgt:new", "depends_on": ["safeline-db", "nginx"]},
            "safeline-db": {"image": "safeline-db:new"},
        }),
        ("nginx", {"nginx": {"image": "nginx:new", "build": {"context": "/tmp/nginx"}}}),
    ], running=("nginx",) + selected if action == "restart" else ("nginx",))
    instrument_images(manager)

    getattr(manager.compose_operations, action)(["safeline"], pull=True)

    assert [event[1] for event in manager.events if event[0] == "apply"] == ["safeline-db", "safeline-mgt"]
    assert [event[1] for event in manager.events if event[0] == "pull"] == [("safeline-db", "safeline-mgt")]
    assert [event[1] for event in manager.events if event[0] == "prepare"] == ["safeline"]
    assert not any(event[0] == "build" for event in manager.events)


def test_pending_external_configuration_does_not_require_its_new_transitive_cache(tmp_path: Path) -> None:
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["provider"]}}),
        ("provider", {"provider": {"image": "provider:new"}}),
        ("cache", {"cache": {"image": "cache:new"}}),
    ], running=("provider",))
    old = AppliedServiceModels(manager, manager.model).previous["provider"]
    manager.model["services"]["provider"]["depends_on"] = ["cache"]
    manager.containers["provider"].dependencies = ("cache",)
    manager.containers["app"].dependencies = ("provider",)
    instrument_images(manager)

    manager.compose_operations.up(["app"], pull=True)

    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert [event[1] for event in manager.events if event[0] == "pull"] == [("app",)]
    assert AppliedServiceModels(manager, manager.model).previous["provider"] == old
    assert not any(event[0] == "prepare" and event[1] in ("provider", "cache") for event in manager.events)


def test_successfully_completed_external_job_satisfies_dependency_without_reapplication(tmp_path: Path) -> None:
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": {
            "migration": {"condition": "service_completed_successfully"}}}}),
        ("database-migration", {"migration": {"image": "migration:new"}}),
    ], running=("migration",))
    job = manager.docker_inspector.get_project_state(None).service_instances("migration")[0]
    job.state, job.exit_code = "exited", 0
    old = AppliedServiceModels(manager, manager.model).previous["migration"]
    instrument_images(manager)

    manager.compose_operations.up(["app"], pull=True)

    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert [event[1] for event in manager.events if event[0] == "pull"] == [("app",)]
    assert AppliedServiceModels(manager, manager.model).previous["migration"] == old


def test_no_names_still_prepares_and_applies_the_full_dependency_order(tmp_path: Path) -> None:
    manager = application_case(tmp_path, running=())

    manager.compose_operations.up(pull=True)

    applied = [event[1] for event in manager.events if event[0] == "apply"]
    assert set(applied) == set(manager.model["services"])
    assert applied.index("lldap") < applied.index("auth") < applied.index("nginx") < applied.index("safeline")
    assert applied.index("redis") < applied.index("auth") < applied.index("omniroute")
    assert [event[1] for event in manager.events if event[0] == "build"] == [("nginx",)]
    assert {event[1] for event in manager.events if event[0] == "prepare"} == set(manager.containers)


@pytest.mark.parametrize("previous_provenance", [False, True])
def test_running_integration_consumer_remains_outside_the_named_operation(
        tmp_path: Path, previous_provenance: bool) -> None:
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new", "build": {"context": "/tmp/nginx"}}}),
    ], running=("nginx",))
    if previous_provenance:
        manager.artifact_index.record({"generated/site.conf": {
            "kind": "generated-config", "container": "nginx", "producers": ["app"]}})
    else:
        manager.integration_snapshot["app"] = (SimpleNamespace(consumer="nginx"),)
    instrument_images(manager)

    manager.compose_operations.up(["app"], pull=True)

    assert [event[1] for event in manager.events if event[0] == "prepare"] == ["app"]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert not any(event[0] == "build" for event in manager.events)


@pytest.mark.parametrize("include_coauthor", [False, True])
def test_shared_service_requires_explicit_selection_of_its_coauthor(
        tmp_path: Path, include_coauthor: bool) -> None:
    manager = setup_case(tmp_path, [
        ("base", {"app": {"image": "app:new"}}),
        ("overlay", {"app": {"image": "app:new", "environment": {"EXTRA": "yes"}},
                     "sibling": {"image": "sibling:new"}}),
    ])
    instrument_images(manager)

    if include_coauthor:
        manager.compose_operations.up(["base", "overlay"])
        assert {event[1] for event in manager.events if event[0] == "prepare"} == {"base", "overlay"}
        assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "sibling"}
    else:
        with pytest.raises(ContainerError, match="overlay"):
            manager.compose_operations.up(["base"])
        assert_no_image_or_service_mutation(manager)
        assert not any(event[0] == "prepare" and event[1] == "overlay" for event in manager.events)


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("evidence", ["saved_namespace", "saved_restart", "native_namespace", "native_restart"])
def test_live_external_dependents_are_diagnosed_before_images_or_service_mutation(
        tmp_path: Path, action: str, evidence: str) -> None:
    worker = {"image": "worker:new"}
    if evidence == "saved_namespace":
        worker["network_mode"] = "service:app"
    elif evidence == "saved_restart":
        worker["depends_on"] = {"app": {"condition": "service_started", "restart": True}}
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}), ("dependent", {"worker": worker}),
    ], running=("app", "worker"))
    instance = manager.docker_inspector.get_project_state(None).service_instances("worker")[0]
    if evidence == "native_namespace":
        instance.namespace_bindings = {"network_mode": "service:app"}
    elif evidence == "native_restart":
        instance.labels["com.docker.compose.depends_on"] = "app:service_started:true"
    instrument_images(manager)

    with pytest.raises(ContainerError, match="joint operation.*worker.*dependent"):
        getattr(manager.compose_operations, action)(["app"], pull=True)

    assert_no_image_or_service_mutation(manager)
