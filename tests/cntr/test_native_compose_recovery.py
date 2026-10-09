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
    manager = SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda selected: (_ for _ in ()).throw(
            AssertionError("Optional dependency must not be polled"))))
    runner = ComposeRunner(manager)
    runner.wait_service_dependencies(SimpleNamespace(containers=()), "app", model=model)


def test_selected_optional_dependency_still_orders_before_consumer():
    app = Owner("app", ("app",))
    metrics = Owner("metrics", ("metrics",))
    app.services["app"]["depends_on"] = {"metrics": {"required": False}}
    assert order_services((app, metrics), ("app", "metrics")) == ("metrics", "app")
