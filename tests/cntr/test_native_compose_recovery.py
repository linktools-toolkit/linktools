#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression contracts for reconciled native providers and optional Compose edges."""
from types import SimpleNamespace

from linktools.cntr._operations import ComposeOperations, ComposeSelection
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


def test_implicit_native_consumer_expands_providers_when_generation_changed():
    lldap = Owner("lldap", ("lldap",))
    nginx = Owner("nginx", ("nginx",))
    nginx.application_priority = 100
    nginx.get_runtime_requirements = lambda required: (
        {"authelia": ("authelia",)} if "nginx" in required else {})
    authelia = Owner("authelia", ("authelia", "authelia-redis"), ("authelia",))
    authelia.dependencies = ("lldap",)
    project = (nginx, lldap, authelia)
    explicit = ComposeSelection(project, (lldap,), ("lldap",), False)
    model = {"services": {service: spec
                          for owner in project for service, spec in owner.services.items()}}
    context = SimpleNamespace(initial_running_services={"nginx"},
                              changed_compose_services={"nginx"}, compose_model=model)
    ops = ComposeOperations(SimpleNamespace())
    preliminary = ops._reconcile_selection(explicit, context)
    assert "nginx" in preliminary.services and "authelia" not in preliminary.services

    final = ops._reconcile_selection(explicit, context, {"nginx"})
    assert {"lldap", "authelia", "nginx"} <= set(final.services)
    assert "authelia-redis" not in final.services
    assert {"nginx", "authelia"} <= set(final.native_roots)
    applied = order_services(project, final.services, model, {"nginx"},
                             dependency_roots=final.native_roots)
    assert applied.index("lldap") < applied.index("authelia") < applied.index("nginx")


def test_redis_sidecar_remains_independent_of_owner_native_dependencies():
    target = Owner("lldap", ("lldap",))
    nginx = Owner("nginx", ("nginx",))
    authelia = Owner("authelia", ("authelia", "authelia-redis"), ("authelia",))
    authelia.dependencies = ("nginx",)
    project = (nginx, target, authelia)
    explicit = ComposeSelection(project, (target,), ("lldap",), False)
    context = SimpleNamespace(initial_running_services={"authelia-redis"},
                              changed_compose_services={"authelia-redis"})
    selection = ComposeOperations(SimpleNamespace())._reconcile_selection(
        explicit, context, {"authelia"})
    assert set(selection.services) == {"lldap", "authelia-redis"}
    assert "authelia" not in selection.native_roots


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
    assert order_services(project, ("app",), model, dependency_roots={"app"}) == ("app",)
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
    manager = SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda selected: inspected.append(True) or SimpleNamespace(services=())))
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
    runner = ComposeRunner(SimpleNamespace(docker_inspector=inspector))
    runner.wait_service_healthy = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Stopped optional dependency must not block application"))
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)


def test_selected_optional_dependency_still_orders_before_consumer():
    app = Owner("app", ("app",))
    metrics = Owner("metrics", ("metrics",))
    app.services["app"]["depends_on"] = {"metrics": {"required": False}}
    assert order_services((app, metrics), ("app", "metrics")) == ("metrics", "app")


def test_native_provider_callbacks_follow_provider_service_order():
    consumer = Owner("consumer", ("consumer",))
    provider = Owner("provider", ("provider",))
    consumer.get_runtime_requirements = lambda required: (
        {"provider": ("provider",)} if "consumer" in required else {})
    selection = ComposeOperations(SimpleNamespace()).start_selection(
        ComposeSelection((consumer, provider), (consumer,), ("consumer",), False))
    assert selection.services == ("provider", "consumer")
    assert tuple(owner.name for owner in selection.target_containers) == ("provider", "consumer")


def test_sidecar_callback_order_does_not_expand_owning_container_dependencies():
    explicit = Owner("explicit", ("explicit",))
    native = Owner("owner", ("native", "sidecar"), ("native",))
    native.dependencies = ("unrelated",)
    unrelated = Owner("unrelated", ("unrelated",))
    project = (native, explicit, unrelated)
    selection = ComposeOperations(SimpleNamespace()).start_selection(
        ComposeSelection(project, (native, explicit), ("sidecar", "explicit"), False),
        dependency_roots=(explicit,))
    assert set(selection.services) == {"sidecar", "explicit"}
    assert "unrelated" not in {owner.name for owner in selection.target_containers}
