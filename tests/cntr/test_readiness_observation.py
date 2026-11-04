#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Startup does not wait for health or job completion."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from linktools.cntr.errors import ContainerError
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.compose import ComposeRunner, _startup_model
from linktools.cntr.runtime.inspect import ProjectRuntimeState


def test_startup_model_preserves_dependency_order_but_removes_readiness_conditions():
    model = {"services": {
        "db": {},
        "seed": {},
        "web": {"depends_on": {
            "db": {"condition": "service_healthy", "required": False, "restart": True},
            "seed": {"condition": "service_completed_successfully"},
        }},
    }}
    actual = _startup_model(model)
    assert actual["services"]["web"]["depends_on"] == {
        "db": {"condition": "service_started", "required": False, "restart": True},
        "seed": {"condition": "service_started"},
    }
    assert model["services"]["web"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_apply_batch_does_not_inspect_or_wait_and_keeps_compose_start_order():
    commands, models = [], []
    runner = ComposeRunner(SimpleNamespace(runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: SimpleNamespace(
            check_call=lambda: commands.append(args) or 0))))
    runner._model_args = lambda context, model=None: (
        models.append(model) or nullcontext(["compose"]))
    context = SimpleNamespace(
        is_full_project=False, target_services=("db", "web"),
        compose_model={"services": {
            "db": {},
            "web": {"depends_on": {"db": {"condition": "service_healthy"}}},
        }},
    )
    runner.apply_services(context, ("db", "web"))
    assert commands == [("compose", "up", "--detach", "--no-build", "--pull", "never",
                         "--no-deps", "db", "web")]
    assert models[0]["services"]["web"]["depends_on"]["db"]["condition"] == "service_started"


@pytest.mark.parametrize(("state", "health"), [
    ("running", "starting"), ("running", "healthy"), ("running", "unhealthy"),
    ("running", None), ("restarting", "starting"),
])
def test_started_service_does_not_wait_for_health(state, health):
    state = ProjectRuntimeState("test", (SimpleNamespace(
        service="app", state=state, health=health, exit_code=None,
        image_id="sha256:app", labels={}),), "docker")
    calls = []
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: calls.append(services) or state)))
    context = SimpleNamespace(project_containers=(), compose_model={"services": {"app": {}}})
    result = runner.verify_services_started(context, ("app",), {"app": "sha256:app"})
    assert result is state
    assert calls == [("app",)]


def test_unhealthy_service_does_not_block_start():
    state = ProjectRuntimeState("test", (SimpleNamespace(
        service="app", state="running", health="unhealthy", exit_code=None,
        image_id="sha256:app", labels={}),), "docker")
    calls = []
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: calls.append(services) or state)))
    context = SimpleNamespace(project_containers=(), compose_model={"services": {"app": {}}})
    assert runner.verify_services_started(context, ("app",), {"app": "sha256:app"}) is state
    assert calls == [("app",)]


@pytest.mark.parametrize("health", [None, "healthy", "unhealthy"])
def test_successfully_exited_service_does_not_check_health(health):
    state = ProjectRuntimeState("test", (SimpleNamespace(
        service="job", state="exited", health=health, exit_code=0,
        image_id="sha256:job", labels={}),), "docker")
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: state)))
    context = SimpleNamespace(project_containers=(), compose_model={"services": {"job": {}}})
    assert runner.verify_services_started(context, ("job",), {"job": "sha256:job"}) is state


@pytest.mark.parametrize(("condition", "state", "health", "exit_code"), [
    ("service_healthy", "running", "starting", None),
    ("service_healthy", "running", "unhealthy", None),
    ("service_completed_successfully", "running", None, None),
    ("service_completed_successfully", "exited", None, 0),
])
def test_external_dependency_conditions_do_not_wait(condition, state, health, exit_code):
    app = SimpleNamespace(name="app", services={"app": {}}, dependencies=())
    provider = SimpleNamespace(name="db", services={"db": {}}, dependencies=())
    selection = ComposeSelection((app, provider), (app,), ("app",), False)
    model = {"services": {
        "app": {"depends_on": {"db": {"condition": condition}}},
        "db": {},
    }}
    actual = ProjectRuntimeState("test", (SimpleNamespace(
        service="db", state=state, health=health, exit_code=exit_code, labels={}),), "docker")
    ComposeOperations(SimpleNamespace())._check_external_requirements(selection, actual, model)


def test_failed_external_job_dependency_fails_immediately_without_waiting():
    app = SimpleNamespace(name="app", services={"app": {}}, dependencies=())
    provider = SimpleNamespace(name="seed", services={"seed": {}}, dependencies=())
    selection = ComposeSelection((app, provider), (app,), ("app",), False)
    model = {"services": {
        "app": {"depends_on": {"seed": {"condition": "service_completed_successfully"}}},
        "seed": {},
    }}
    actual = ProjectRuntimeState("test", (SimpleNamespace(
        service="seed", state="exited", health=None, exit_code=1, labels={}),), "docker")
    with pytest.raises(ContainerError, match="Required dependencies are unavailable"):
        ComposeOperations(SimpleNamespace())._check_external_requirements(selection, actual, model)
