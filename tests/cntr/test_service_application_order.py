#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service priority never overrides dependencies or their readiness conditions."""
from types import SimpleNamespace

import pytest

from linktools.cntr import BaseContainer, ContainerError
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


def test_priority_only_breaks_ties_between_ready_services():
    later = Container("later", {"later": {}}, priority=100)
    first = Container("first", {"first": {}}, priority=-100)
    assert order_services((later, first), ("later", "first")) == ("first", "later")


def test_implicit_sidecar_does_not_inherit_owner_strong_dependency_order():
    provider = Container("nginx", {"nginx": {}}, priority=100)
    sidecar = Container("authelia", {"authelia-redis": {}}, priority=-100,
                        dependencies=("nginx",))
    selected = ("authelia-redis", "nginx")

    assert order_services((sidecar, provider), selected) == ("nginx", "authelia-redis")
    assert order_services(
        (sidecar, provider), selected, dependency_roots={"nginx"}) == selected


def test_bootstrap_availability_breaks_only_started_or_healthy_edges():
    nginx = Container("nginx", {"nginx": {}}, priority=100)
    provider = Container("provider", {"provider": {"depends_on": {
        "nginx": {"condition": "service_healthy"}}}})
    nginx.get_runtime_requirements = lambda required: {"provider": ("provider",)}
    assert order_services((nginx, provider), ("nginx", "provider"), available_services=("nginx",)) == (
        "provider", "nginx")
    provider.services["provider"]["depends_on"]["nginx"]["condition"] = "service_completed_successfully"
    with pytest.raises(ContainerError, match="cycle"):
        order_services((nginx, provider), ("nginx", "provider"), available_services=("nginx",))


@pytest.mark.parametrize("condition,method", [
    ("service_healthy", "wait_service_healthy"),
    ("service_completed_successfully", "wait_service_completed"),
])
def test_apply_service_always_waits_for_dependency_before_start(condition, method, monkeypatch):
    calls = []
    process = SimpleNamespace(check_call=lambda: calls.append("apply"))
    manager = SimpleNamespace(runtime=SimpleNamespace(create_docker_compose_process=lambda *a: process))
    runner = ComposeRunner(manager)
    context = SimpleNamespace(containers=(), is_full_containers=False, compose_model={"services": {
        "app": {"depends_on": {"provider": {"condition": condition}}}}})
    monkeypatch.setattr(runner, method, lambda ctx, service: calls.append((condition, service)))
    runner.apply_service(context, "app")
    assert calls == [(condition, "provider"), "apply"]


@pytest.mark.parametrize("code", [0, 1, None])
def test_completed_dependency_requires_observed_successful_exit(code):
    state = SimpleNamespace(services=(SimpleNamespace(service="job", state="exited", exit_code=code),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda cs: state)))
    context = SimpleNamespace(containers=())
    if code == 0:
        runner.wait_service_completed(context, "job", timeout=0)
    else:
        with pytest.raises(ContainerError, match="failed"):
            runner.wait_service_completed(context, "job", timeout=0)


def test_running_dependency_is_not_completed():
    state = SimpleNamespace(services=(SimpleNamespace(service="job", state="running", exit_code=0),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda cs: state)))
    with pytest.raises(ContainerError, match="did not complete"):
        runner.wait_service_completed(SimpleNamespace(containers=()), "job", timeout=0)



def test_started_dependency_does_not_require_one_shot_service_to_keep_running():
    calls = []
    process = SimpleNamespace(check_call=lambda: calls.append("apply"))
    runner = ComposeRunner(SimpleNamespace(runtime=SimpleNamespace(
        create_docker_compose_process=lambda *args: process)))
    context = SimpleNamespace(containers=(), is_full_containers=False, compose_model={"services": {
        "app": {"depends_on": ["job"]}}})
    runner.apply_service(context, "app")
    assert calls == ["apply"]
