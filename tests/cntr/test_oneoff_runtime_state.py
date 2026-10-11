#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maintenance containers never become deployment or recovery service evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import OperationContext
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
from test_lifecycle_rebuild import setup_case


def instance(oneoff=False, state="running", health="healthy", exit_code=None):
    return ServiceRuntimeState(
        logical_containers=("app",), service="app", runtime_name="maintenance" if oneoff else "app",
        state=state, health=health, image="app:old", exit_code=exit_code,
        labels={"com.docker.compose.oneoff": "True" if oneoff else "False"},
        image_id="sha256:maintenance" if oneoff else "sha256:original")


def test_oneoff_is_retained_for_diagnostics_but_excluded_from_service_views():
    regular, maintenance = instance(), instance(oneoff=True)
    state = ProjectRuntimeState("test", (regular, maintenance), "docker")
    assert state.services == (regular, maintenance)
    assert state.existing_services == state.running_services == {"app"}
    assert state.image_ids == state.running_images == {"app": "sha256:original"}
    assert state.running_container_names == ["app"]
    alone = replace(state, services=(maintenance,))
    assert alone.existing_services == alone.running_services == set()
    assert alone.image_ids == alone.running_images == {}
    assert alone.running_container_names == []


@pytest.mark.parametrize("method", ["wait_service_healthy", "wait_service_ready", "wait_service_completed"])
def test_single_service_wait_ignores_failed_maintenance_container(method):
    completed = method == "wait_service_completed"
    regular = instance(state="exited" if completed else "running", exit_code=0 if completed else None)
    maintenance = instance(oneoff=True, state="exited", health=None, exit_code=1)
    state = ProjectRuntimeState("test", (regular, maintenance), "docker")
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda *args, **kwargs: state)))
    context = OperationContext(project_containers=(), target_services=("app",),
                               compose_model={"services": {"app": {}}})
    getattr(runner, method)(context, "app", timeout=None)


@pytest.mark.parametrize("condition", ["service_started", "service_healthy", "service_completed_successfully"])
def test_recovery_dependency_wait_ignores_oneoff_replicas(condition):
    completed = condition == "service_completed_successfully"
    regular = instance(state="exited" if completed else "running", exit_code=0 if completed else None)
    maintenance = instance(oneoff=True, state="exited", health=None, exit_code=1)
    state = ProjectRuntimeState("test", (regular, maintenance), "docker")
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda *args, **kwargs: state)))
    model = {"services": {"web": {"depends_on": {"app": {"condition": condition}}}, "app": {}}}
    context = OperationContext(project_containers=(), target_services=("web", "app"), compose_model=model)
    runner.wait_service_dependencies(context, "web", model=model)


def test_recovery_pins_regular_service_image_instead_of_newer_oneoff(tmp_path):
    documents = []

    def process(*args, **kwargs):
        documents.append([yaml.safe_load(Path(args[index + 1]).read_text())
                          for index, value in enumerate(args[:-1]) if value == "--file"])
        return SimpleNamespace(check_call=lambda: 0)

    manager = SimpleNamespace(data_path=tmp_path, project_name="test",
                              runtime=SimpleNamespace(create_docker_process=process))
    runner = ComposeRunner(manager)
    runner._resolved_model = lambda process: {"services": {"app": {"image": "sha256:original"}}}
    context = OperationContext(project_containers=(), target_services=("app",),
                               initial_runtime_state=ProjectRuntimeState(
                                   "test", (instance(), instance(oneoff=True)), "docker"))
    runner.apply_saved_services(context, ("app",), {"saved.yml": "services:\n  app:\n    image: app:old\n"})
    assert documents[0][-1]["services"]["app"]["image"] == "sha256:original"


def test_legacy_namespace_capture_ignores_oneoff_bindings(tmp_path):
    old = {"services": {"app": {"image": "app:old", "network_mode": "service:db"},
                        "db": {"image": "db:old"}}}
    regular = replace(instance(), namespace_bindings={"network_mode": "service:db"})
    maintenance = replace(instance(oneoff=True), namespace_bindings={"network_mode": "host"})
    context = OperationContext(
        project_containers=(),
        initial_runtime_state=ProjectRuntimeState("test", (regular, maintenance), "docker"))
    context.service_models = SimpleNamespace(previous={})
    context.previous_compose_contents = {"old.yml": yaml.safe_dump(old)}
    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test", runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: SimpleNamespace(args=args))))
    runner._resolved_model = lambda process: yaml.safe_load(
        Path(process.args[process.args.index("--file") + 1]).read_text())
    saved = yaml.safe_load(runner.saved_service_models(context, ("app",))["app"])
    assert saved["services"]["app"]["network_mode"] == "service:db"


def test_partial_stop_recovery_does_not_count_newly_running_oneoff(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    before = ProjectRuntimeState("test", (instance(), instance(oneoff=True, state="exited", exit_code=0)), "docker")
    after = replace(before, services=(instance(state="exited", exit_code=0), instance(oneoff=True)))
    observations = [before, after]
    manager.docker_inspector.get_project_state = lambda *args, **kwargs: observations.pop(0)

    def partial_stop(context, services):
        raise ContainerError("stop interrupted")

    manager.compose_runner.stop = partial_stop
    with pytest.raises(ContainerError, match="stop interrupted"):
        manager.compose_operations.restart(["app"])
    assert ("restore", ("app",)) in manager.events
