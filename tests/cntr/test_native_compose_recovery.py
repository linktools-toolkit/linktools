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
    runner.wait_service_dependencies(SimpleNamespace(project_containers=()), "app", model=model)
    assert inspected == [True]


def test_optional_running_healthy_dependency_still_waits():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    inspector = SimpleNamespace(get_project_state=lambda selected: SimpleNamespace(services=(
        SimpleNamespace(service="metrics", state="running", health="starting"),)))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=inspector))
    calls = []
    runner.wait_service_healthy = lambda ctx, dep, timeout=None: calls.append((dep, timeout))
    runner.wait_service_dependencies(SimpleNamespace(project_containers=()), "app", model=model)
    assert calls == [("metrics", None)]


def test_optional_selected_but_unavailable_dependency_retains_readiness_requirement():
    model = {"services": {"app": {"depends_on": {
        "metrics": {"condition": "service_healthy", "required": False}}}}}
    runner = ComposeRunner(SimpleNamespace())
    calls = []
    runner.wait_service_healthy = lambda ctx, dep, timeout=None: calls.append((dep, timeout))
    context = SimpleNamespace(project_containers=(), target_services=("app", "metrics"))
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
    runner.wait_service_dependencies(SimpleNamespace(project_containers=()), "app", model=model)
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
    runner.wait_service_dependencies(SimpleNamespace(project_containers=()), "app", model=model)


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
    context = SimpleNamespace(project_containers=(),
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
            SimpleNamespace(project_containers=()), "app",
            model={"services": {"app": {"depends_on": {
                "metrics": {"condition": "service_healthy"}}}}})


def test_optional_dependency_does_not_swallow_runtime_inspection_failures():
    def fail(_):
        raise RuntimeError("inspection unavailable")
    runner = ComposeRunner(SimpleNamespace(
        docker_inspector=SimpleNamespace(get_project_state=fail)))
    with pytest.raises(RuntimeError, match="inspection unavailable"):
        runner.wait_service_dependencies(
            SimpleNamespace(project_containers=()), "app",
            model={"services": {"app": {"depends_on": {
                "metrics": {"condition": "service_healthy", "required": False}}}}})


def test_restart_saved_orphan_uses_saved_compose_model(tmp_path):
    from pathlib import Path
    import yaml

    calls = []
    saved = {"services": {"retired": {"image": "old:image"}}}

    def process(*args):
        path = args[args.index("--file") + 1]
        assert yaml.safe_load(Path(path).read_text()) == saved
        calls.append(args)
        return SimpleNamespace(check_call=lambda: 0)

    manager = SimpleNamespace(data_path=tmp_path, project_name="test",
                              runtime=SimpleNamespace(create_docker_process=process))
    runner = ComposeRunner(manager)
    context = SimpleNamespace(compose_model={"services": {"current": {"image": "new:image"}}})
    runner.restart_service(context, "retired", model=saved)
    assert calls[0][-3:] == ("restart", "--no-deps", "retired")


@pytest.mark.parametrize("separate_owner", [False, True])
@pytest.mark.parametrize("services", [("cache", "web"), ("web", "cache")])
def test_legacy_dependency_removed_from_current_owner_uses_saved_declaration(tmp_path, separate_owner, services):
    from pathlib import Path
    import yaml

    old = {"services": {"db": {"image": "db:old"}, "cache": {"image": "cache:old"},
                        "web": {"image": "web:old", "network_mode": "service:db"}}}
    manager = SimpleNamespace(data_path=tmp_path, project_name="test",
                              runtime=SimpleNamespace(create_docker_process=lambda *args, **kw: SimpleNamespace(args=args)))
    runner = ComposeRunner(manager)

    def resolve(process):
        result = {"services": {}}
        for index, arg in enumerate(process.args[:-1]):
            if arg == "--file":
                result["services"].update(yaml.safe_load(Path(process.args[index + 1]).read_text())["services"])
        return result

    runner._resolved_model = resolve
    saved = {"app.yml": yaml.safe_dump(old)}
    if separate_owner:
        saved = {"app.yml": yaml.safe_dump({"services": {"web": old["services"]["web"]}}),
                 "provider.yml": yaml.safe_dump({"services": {name: old["services"][name] for name in ("db", "cache")}})}
    saved["unrelated.yml"] = "invalid: ["
    context = SimpleNamespace(project_containers=(Owner("app", ("web", "cache") if not separate_owner else ("web",)),
                                          Owner("provider", ("cache",)) if separate_owner else Owner("empty", ())),
                              previous_compose_contents=saved,
                              service_models=SimpleNamespace(previous={}))
    assert yaml.safe_load(runner.saved_service_models(context, services)["web"]) == old
