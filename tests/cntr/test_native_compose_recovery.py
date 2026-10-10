#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression contracts for reconciled native providers and optional Compose edges."""
from types import SimpleNamespace

import pytest

from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner, order_services
from linktools.cntr.runtime.images import ImagePreparer


class Owner:
    dependencies = ()
    application_priority = 0
    bootstrap_services = ()

    def __init__(self, name, services, generation_services=()):
        self.name = name
        self.services = {service: {} for service in services}
        self.generation_services = generation_services or tuple(services)

    def get_runtime_requirements(self, required):
        return {}


def test_optional_compose_provider_not_added_to_start_or_build_scope():
    app = Owner("app", ("app",))
    metrics = Owner("metrics", ("metrics",))
    app.services["app"]["depends_on"] = {
        "metrics": {"condition": "service_healthy", "required": False}}
    project = (app, metrics)
    model = {"services": {"app": app.services["app"], "metrics": {"image": "metrics:latest"}}}
    selected = ComposeOperations(SimpleNamespace()).start_selection(
        ComposeSelection(project, (app,), ("app",), False), model)
    assert selected.services == ("app",)
    assert order_services(project, ("app",), model) == ("app",)
    manager = SimpleNamespace()
    planner = ImagePreparer(manager)
    planner.image_exists = lambda image: True
    full_model = {"services": {"app": dict(app.services["app"], image="app:latest"),
                               "metrics": {"image": "metrics:latest"}}}
    assert planner.plan(full_model, ("app",)).targets == ("app",)


def test_optional_unavailable_dependency_does_not_block_old_model_recovery():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    inspected = []
    manager = SimpleNamespace(
        docker_inspector=SimpleNamespace(
            get_project_state=lambda selected: inspected.append(True) or SimpleNamespace(services=())),
        logger=SimpleNamespace(warning=lambda *args: None))
    runner = ComposeRunner(manager)
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)
    assert inspected == [True]


def test_optional_running_healthy_dependency_still_waits():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    inspector = SimpleNamespace(get_project_state=lambda selected: SimpleNamespace(services=(
        SimpleNamespace(service="metrics", state="running", health="starting"),)))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=inspector))
    calls = []
    runner.wait_service_healthy = lambda ctx, dep, timeout=None: calls.append((dep, timeout))
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)
    assert calls == [("metrics", None)]


def test_optional_selected_but_unavailable_dependency_retains_readiness_requirement():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    runner = ComposeRunner(SimpleNamespace())
    calls = []
    runner.wait_service_healthy = lambda ctx, dep, timeout=None: calls.append((dep, timeout))
    context = SimpleNamespace(containers=(), target_services=("app", "metrics"))
    runner.wait_service_dependencies(context, "app", model=model)
    assert calls == [("metrics", None)]


def test_optional_completed_successfully_is_still_checked():
    model = {"services": {"app": {"depends_on": {
        "seed": {"condition": "service_completed_successfully", "required": False}}}}}
    inspector = SimpleNamespace(get_project_state=lambda selected: SimpleNamespace(services=(
        SimpleNamespace(service="seed", state="exited", exit_code=0),)))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=inspector))
    calls = []
    runner.wait_service_completed = lambda ctx, dep, timeout=None: calls.append((dep, timeout))
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)
    assert calls == [("seed", None)]


def test_optional_stopped_dependency_is_not_polled_for_health():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    inspector = SimpleNamespace(get_project_state=lambda selected: SimpleNamespace(services=(
        SimpleNamespace(service="metrics", state="exited", health="unhealthy", exit_code=1),)))
    runner = ComposeRunner(SimpleNamespace(
        docker_inspector=inspector, logger=SimpleNamespace(warning=lambda *args: None)))
    runner.wait_service_healthy = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Stopped optional dependency must not block application"))
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)


def test_selected_optional_dependency_still_orders_before_consumer():
    app = Owner("app", ("app",))
    metrics = Owner("metrics", ("metrics",))
    app.services["app"]["depends_on"] = {"metrics": {"required": False}}
    assert order_services((app, metrics), ("app", "metrics")) == ("metrics", "app")


def test_native_provider_co_selection_does_not_create_runtime_edges():
    consumer = Owner("consumer", ("consumer",))
    provider = Owner("provider", ("provider",))
    consumer.get_runtime_requirements = lambda required: (
        {"provider": ("provider",)} if "consumer" in required else {})
    selection = ComposeOperations(SimpleNamespace()).start_selection(
        ComposeSelection((consumer, provider), (consumer,), ("consumer",), False))
    assert set(selection.services) == {"provider", "consumer"}
    assert set(owner.name for owner in selection.target_containers) == {"provider", "consumer"}


@pytest.mark.parametrize("selected,condition,state,health,exit_code", [
    (False, "service_healthy", "running", "unhealthy", None),
    (True, "service_healthy", "running", "unhealthy", None),
    (True, "service_completed_successfully", "exited", None, 7),
    (False, "service_started", "exited", None, 7),
])
def test_optional_terminal_failure_warning_without_blocking(
        selected, condition, state, health, exit_code):
    logs = []
    runtime = SimpleNamespace(
        services=(SimpleNamespace(service="metrics", state=state,
                                  health=health, exit_code=exit_code),))
    manager = SimpleNamespace(
        docker_inspector=SimpleNamespace(get_project_state=lambda containers: runtime),
        logger=SimpleNamespace(warning=lambda *args: logs.append(args)))
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": condition, "required": False}}}}}
    context = SimpleNamespace(containers=(),
        target_services=("app", "metrics") if selected else ("app",))
    ComposeRunner(manager).wait_service_dependencies(context, "app", model=model)
    assert logs and "metrics" in logs[0][1]


def test_required_unhealthy_dependency_still_fails():
    runtime = SimpleNamespace(
        services=(SimpleNamespace(service="metrics", state="running", health="unhealthy"),))
    runner = ComposeRunner(SimpleNamespace(
        docker_inspector=SimpleNamespace(get_project_state=lambda owners: runtime)))
    with pytest.raises(ContainerError, match="unhealthy"):
        runner.wait_service_dependencies(
            SimpleNamespace(containers=()), "app",
            model={"services": {"app": {"depends_on": {
                "metrics": {"condition": "service_healthy"}}}}})


def test_optional_dependency_does_not_swallow_runtime_inspection_failures():
    def fail(_):
        raise RuntimeError("inspection unavailable")
    runner = ComposeRunner(SimpleNamespace(
        docker_inspector=SimpleNamespace(get_project_state=fail)))
    with pytest.raises(RuntimeError, match="inspection unavailable"):
        runner.wait_service_dependencies(
            SimpleNamespace(containers=()), "app",
            model={"services": {"app": {"depends_on": {
                "metrics": {"condition": "service_healthy", "required": False}}}}})
