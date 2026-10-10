#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service priority never overrides dependencies or their readiness conditions."""
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from linktools.cntr import BaseContainer, ContainerError
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.compose import ComposeRunner, order_services


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
    result = ComposeOperations(SimpleNamespace()).start_selection(selection)
    assert "database" in result.services and "redis" in result.services
    assert "nginx" not in result.services
    assert result.services.index("database") < result.services.index("redis")


def test_group_co_selection_does_not_inject_runtime_dependency_cycle():
    nginx = Container("nginx", {"nginx": {}})
    provider = Container("provider", {"provider": {"depends_on": ["nginx"]}})
    nginx.get_runtime_requirements = lambda required: {"provider": ("provider",)}
    assert order_services((nginx, provider), ("nginx", "provider")) == ("nginx", "provider")


@pytest.mark.parametrize("condition,method", [
    ("service_healthy", "wait_service_healthy"),
    ("service_completed_successfully", "wait_service_completed"),
])
def test_apply_service_always_waits_for_dependency_before_start(condition, method, monkeypatch):
    calls = []
    process = SimpleNamespace(check_call=lambda: calls.append("apply"))
    manager = SimpleNamespace(runtime=SimpleNamespace(create_docker_process=lambda *a, **kw: process))
    runner = ComposeRunner(manager)
    runner._model_args = lambda context: nullcontext(["compose"])
    context = SimpleNamespace(project_containers=(), is_full_project=False, compose_model={"services": {
        "app": {"depends_on": {"provider": {"condition": condition}}}}})
    monkeypatch.setattr(runner, method, lambda ctx, service, timeout=None: calls.append((condition, service)))
    runner.apply_service(context, "app")
    assert calls == [(condition, "provider"), "apply"]


@pytest.mark.parametrize("code", [0, 1, None])
def test_completed_dependency_requires_observed_successful_exit(code):
    state = SimpleNamespace(services=(SimpleNamespace(service="job", state="exited", exit_code=code),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda cs: state)))
    context = SimpleNamespace(project_containers=())
    if code == 0:
        runner.wait_service_completed(context, "job", timeout=0)
    else:
        with pytest.raises(ContainerError, match="failed"):
            runner.wait_service_completed(context, "job", timeout=0)


def test_running_dependency_is_not_completed():
    state = SimpleNamespace(services=(SimpleNamespace(service="job", state="running", exit_code=0),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda cs: state)))
    with pytest.raises(ContainerError, match="did not complete"):
        runner.wait_service_completed(SimpleNamespace(project_containers=()), "job", timeout=0)



def test_started_dependency_does_not_require_one_shot_service_to_keep_running():
    calls = []
    process = SimpleNamespace(check_call=lambda: calls.append("apply"))
    runner = ComposeRunner(SimpleNamespace(runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: process)))
    runner._model_args = lambda context: nullcontext(["compose"])
    context = SimpleNamespace(project_containers=(), is_full_project=False, compose_model={"services": {
        "app": {"depends_on": ["job"]}}})
    runner.apply_service(context, "app")
    assert calls == ["apply"]
