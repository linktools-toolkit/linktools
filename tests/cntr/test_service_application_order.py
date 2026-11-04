#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service priority never overrides dependency start order."""
from types import SimpleNamespace
from contextlib import nullcontext


from linktools.cntr import BaseContainer
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.compose import ComposeRunner, order_services, _startup_model


class Container(BaseContainer):
    name = ""
    dependencies = ()

    def __init__(self, name, services, priority=0, dependencies=()):
        self.name = name
        self.services = services
        self.application_priority = priority
        self.dependencies = dependencies


def test_high_priority_provider_precedes_low_priority_dependent():
    provider = Container("provider", {"provider": {}}, priority=100)
    client = Container("client", {"client": {"depends_on": ["provider"]}}, priority=-100)
    assert order_services((client, provider), ("client", "provider")) == ("provider", "client")


def test_unrelated_services_keep_declared_order():
    later = Container("later", {"later": {}}, priority=100)
    first = Container("first", {"first": {}}, priority=-100)
    assert order_services((later, first), ("later", "first")) == ("later", "first")


def test_implicit_sidecar_has_no_extra_owner_start_edge():
    provider = Container("nginx", {"nginx": {}}, priority=100)
    sidecar = Container("authelia", {"authelia-redis": {}}, priority=-100,
                        dependencies=("nginx",))
    selected = ("authelia-redis", "nginx")

    assert order_services((sidecar, provider), selected) == selected


def test_implicit_service_preserves_compose_edges_without_owner_dependencies():
    target = Container("target", {"target": {"depends_on": ["redis"]}})
    owner = Container("owner", {"redis": {"depends_on": ["database"]}},
                      dependencies=("nginx",))
    database = Container("database", {"database": {}})
    nginx = Container("nginx", {"nginx": {}})
    project = (target, database, nginx, owner)
    selection = ComposeSelection(project, (target,), ("target",), False)
    # Selection through a Compose service edge does not promote its owner's
    # unrelated logical container dependencies.
    result = ComposeOperations(SimpleNamespace()).required_selection(selection)
    assert "database" in result.services and "redis" in result.services
    assert "nginx" not in result.services
    assert result.services.index("database") < result.services.index("redis")


def test_explicit_co_selection_orders_only_native_dependency_edges():
    nginx = Container("nginx", {"nginx": {}})
    provider = Container("provider", {"provider": {"depends_on": ["nginx"]}})
    assert order_services((nginx, provider), ("nginx", "provider")) == ("nginx", "provider")


def test_selected_optional_dependency_still_orders_before_consumer():
    app = Container("app", {"app": {"depends_on": {"metrics": {"required": False}}}})
    metrics = Container("metrics", {"metrics": {}})
    assert order_services((app, metrics), ("app", "metrics")) == ("metrics", "app")





def test_startup_model_keeps_order_and_restart_flags_without_health_or_job_waits():
    model = {"services": {"db": {}, "seed": {}, "app": {"depends_on": {
        "db": {"condition": "service_healthy", "restart": True},
        "seed": {"condition": "service_completed_successfully"},
    }}}}
    result = _startup_model(model)
    assert result["services"]["app"]["depends_on"] == {
        "db": {"condition": "service_started", "restart": True},
        "seed": {"condition": "service_started"},
    }
    assert model["services"]["app"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_apply_service_does_not_query_dependencies_before_start():
    calls, models = [], []
    runner = ComposeRunner(SimpleNamespace(runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs:
        SimpleNamespace(check_call=lambda: calls.append(args) or 0))))
    runner._model_args = lambda context, model=None: (models.append(model) or nullcontext(["compose"]))
    context = SimpleNamespace(project_containers=(), is_full_project=False, target_services=("app",),
        compose_model={"services": {"app": {"depends_on": {
            "db": {"condition": "service_healthy"}}}}})
    runner.apply_service(context, "app")
    assert calls == [("compose", "up", "--detach", "--no-build", "--pull", "never", "--no-deps", "app")]
    assert models[0]["services"]["app"]["depends_on"]["db"]["condition"] == "service_started"
