#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr status``: read-only, defaults to non-interactive sudo."""
import json

import pytest

from linktools.cntr.commands.status import collect_status
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState


def _state(services):
    return ProjectRuntimeState(project="aio", services=tuple(services), backend="docker")


@pytest.fixture
def with_nginx_services(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    nginx.__dict__["services"] = {"nginx": {}}
    return nginx


def test_status_is_read_only(fresh_manager, monkeypatch, with_nginx_services):
    monkeypatch.setattr(
        fresh_manager.compose_operations, "status",
        lambda sudo_prompt=False: ((with_nginx_services,), _state([
            ServiceRuntimeState(logical_container="nginx", service="nginx", runtime_name="aio-nginx",
                                state="running", health="healthy", image=None, exit_code=0, labels={}),
        ])),
    )
    persisted_calls = []
    monkeypatch.setattr(fresh_manager.running_state, "_set", lambda *a, **k: persisted_calls.append(a))

    payload = collect_status(fresh_manager)

    assert persisted_calls == []
    assert payload["containers"][0]["container"] == "nginx"
    assert payload["containers"][0]["status"] == "running"


def test_status_defaults_to_non_interactive_sudo(fresh_manager, monkeypatch, with_nginx_services):
    recorded = {}

    def fake_get_project_state(containers, allow_sudo_prompt=False):
        recorded["allow_sudo_prompt"] = allow_sudo_prompt
        return _state([])

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", fake_get_project_state)
    collect_status(fresh_manager)
    assert recorded["allow_sudo_prompt"] is False


def test_status_sudo_prompt_flag_is_forwarded(fresh_manager, monkeypatch, with_nginx_services):
    recorded = {}

    def fake_get_project_state(containers, allow_sudo_prompt=False):
        recorded["allow_sudo_prompt"] = allow_sudo_prompt
        return _state([])

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", fake_get_project_state)
    collect_status(fresh_manager, sudo_prompt=True)
    assert recorded["allow_sudo_prompt"] is True


def test_status_query_failure_marks_unknown_instead_of_crashing(fresh_manager, monkeypatch, with_nginx_services):
    from linktools.cntr.runtime.structured import StructuredCommandError

    def raise_error(containers, allow_sudo_prompt=False):
        raise StructuredCommandError("docker unreachable")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", raise_error)
    payload = collect_status(fresh_manager)
    assert payload["queryable"] is False
    assert all(entry["status"] == "unknown" for entry in payload["containers"])


def test_status_rejects_unknown_container_name(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    with pytest.raises(ContainerError):
        collect_status(fresh_manager, names=["does-not-exist"])


def test_status_aggregation_running_degraded_exited(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    authelia = fresh_manager.containers["authelia"]
    nginx.__dict__["services"] = {"nginx": {}}
    authelia.__dict__["services"] = {"authelia": {}, "redis": {}}

    services = [
        ServiceRuntimeState(logical_container="nginx", service="nginx", runtime_name="aio-nginx",
                            state="running", health="healthy", image=None, exit_code=0, labels={}),
        ServiceRuntimeState(logical_container="authelia", service="authelia", runtime_name="aio-authelia",
                            state="running", health="unhealthy", image=None, exit_code=0, labels={}),
        ServiceRuntimeState(logical_container="authelia", service="redis", runtime_name="aio-redis",
                            state="exited", health=None, image=None, exit_code=1, labels={}),
    ]
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state(services))

    payload = collect_status(fresh_manager, names=["nginx", "authelia"])
    by_name = {entry["container"]: entry for entry in payload["containers"]}
    assert by_name["nginx"]["status"] == "running"
    assert by_name["authelia"]["status"] == "degraded"


def test_status_missing_when_declared_but_not_found(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    nginx.__dict__["services"] = {"nginx": {}}
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    payload = collect_status(fresh_manager, names=["nginx"])
    assert payload["containers"][0]["status"] == "missing"


def test_orphan_service_excluded_by_default_included_with_all_services(fresh_manager, monkeypatch):
    nginx = fresh_manager.containers["nginx"]
    nginx.__dict__["services"] = {"nginx": {}}
    services = [
        ServiceRuntimeState(logical_container="nginx", service="nginx", runtime_name="aio-nginx",
                            state="running", health=None, image=None, exit_code=0, labels={}),
        ServiceRuntimeState(logical_container=None, service="orphan", runtime_name="orphan-1",
                            state="running", health=None, image=None, exit_code=0, labels={}),
    ]
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state(services))

    default_payload = collect_status(fresh_manager)
    assert default_payload["orphan_services"] == []

    full_payload = collect_status(fresh_manager, all_services=True)
    assert len(full_payload["orphan_services"]) == 1
    assert full_payload["orphan_services"][0]["service"] == "orphan"


def test_json_schema_version_is_stable(fresh_manager, monkeypatch, with_nginx_services):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    payload = collect_status(fresh_manager)
    assert payload["schema_version"] == 1
    json.dumps(payload)  # must be JSON-serializable as-is
