#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr status``: read-only; blocks on a sudo password prompt like any
other docker call if the configured docker type needs one."""
import json

import pytest

from linktools.cntr.commands.status import ServiceStatus, collect_status, select_status_containers
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
        fresh_manager.docker_inspector, "get_project_state",
        lambda *a, **k: _state([
            ServiceRuntimeState(logical_container="nginx", service="nginx", runtime_name="aio-nginx",
                                state="running", health="healthy", image=None, exit_code=0, labels={}),
        ]),
    )
    persisted_calls = []
    monkeypatch.setattr(fresh_manager.running_state, "_set", lambda *a, **k: persisted_calls.append(a))

    payload = collect_status(fresh_manager)

    assert persisted_calls == []
    assert payload["containers"][0]["container"] == "nginx"
    assert payload["containers"][0]["status"] == "running"


def test_status_query_failure_marks_unknown_instead_of_crashing(fresh_manager, monkeypatch, with_nginx_services):
    """A denied sudo policy (or any other unqueryable-runtime failure) must
    degrade to "unknown" status, not crash `ct-cntr status`."""
    from linktools.cntr.runtime.inspect import RuntimeInspectionUnavailable

    def raise_error(containers):
        raise RuntimeInspectionUnavailable("sudo: a password is required")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", raise_error)
    payload = collect_status(fresh_manager)
    assert payload["queryable"] is False
    assert payload["error"] == "sudo: a password is required"
    assert all(entry["status"] == "unknown" for entry in payload["containers"])


def test_status_output_error_propagates_instead_of_being_swallowed(fresh_manager, monkeypatch, with_nginx_services):
    """A structurally invalid docker inspect response must not be treated
    the same as an unqueryable runtime -- it must surface as an error
    (non-zero exit), not silently render every container as unknown."""
    from linktools.cntr.runtime.inspect import RuntimeInspectionOutputError

    def raise_error(containers):
        raise RuntimeInspectionOutputError("docker inspect output root is not a list")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", raise_error)
    with pytest.raises(RuntimeInspectionOutputError):
        collect_status(fresh_manager)


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


# -- name pre-validation must happen before any Docker query (spec §40) -----

def test_unknown_container_name_never_queries_docker(fresh_manager, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("DockerInspector must not be called for an unknown container name")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", fail)
    with pytest.raises(ContainerError):
        collect_status(fresh_manager, names=["does-not-exist"])


def test_multiple_unknown_names_reported_together(fresh_manager, monkeypatch):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    with pytest.raises(ContainerError) as exc_info:
        collect_status(fresh_manager, names=["nope-a", "nope-b"])
    assert "nope-a" in str(exc_info.value)
    assert "nope-b" in str(exc_info.value)


def test_select_status_containers_unit(fresh_manager):
    nginx = fresh_manager.containers["nginx"]
    authelia = fresh_manager.containers["authelia"]
    containers = [nginx, authelia]

    assert select_status_containers(containers, None) == (nginx, authelia)
    assert select_status_containers(containers, []) == (nginx, authelia)
    assert select_status_containers(containers, ["nginx"]) == (nginx,)
    # Duplicates collapse to one entry, in first-seen order.
    assert select_status_containers(containers, ["nginx", "nginx"]) == (nginx,)

    with pytest.raises(ContainerError):
        select_status_containers(containers, ["nginx", "unknown-one", "unknown-two"])


# -- richer aggregation rules (spec §43) -------------------------------------

def _svc(state, health=None, service="nginx", logical_container="nginx"):
    return ServiceRuntimeState(logical_container=logical_container, service=service, runtime_name="rt",
                               state=state, health=health, image=None, exit_code=0, labels={})


@pytest.mark.parametrize("state", ["paused", "created", "removing", "some-unrecognized-state"])
def test_single_service_odd_state_is_degraded(fresh_manager, monkeypatch, with_nginx_services, state):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([_svc(state)]))
    payload = collect_status(fresh_manager)
    assert payload["containers"][0]["status"] == "degraded"


def test_all_exited_is_exited(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    authelia.__dict__["services"] = {"authelia": {}, "redis": {}}
    services = [
        _svc("exited", service="authelia", logical_container="authelia"),
        _svc("dead", service="redis", logical_container="authelia"),
    ]
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state(services))
    payload = collect_status(fresh_manager, names=["authelia"])
    assert payload["containers"][0]["status"] == "exited"


def test_unhealthy_alone_is_degraded_not_running(fresh_manager, monkeypatch, with_nginx_services):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state",
                        lambda *a, **k: _state([_svc("running", health="unhealthy")]))
    payload = collect_status(fresh_manager)
    assert payload["containers"][0]["status"] == "degraded"


def test_no_healthcheck_does_not_degrade(fresh_manager, monkeypatch, with_nginx_services):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state",
                        lambda *a, **k: _state([_svc("running", health=None)]))
    payload = collect_status(fresh_manager)
    assert payload["containers"][0]["status"] == "running"


def test_partial_missing_service_is_degraded(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    authelia.__dict__["services"] = {"authelia": {}, "redis": {}}
    # Only "authelia" observed; "redis" is declared but never appears.
    services = [_svc("running", service="authelia", logical_container="authelia")]
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state(services))
    payload = collect_status(fresh_manager, names=["authelia"])
    entry = payload["containers"][0]
    assert entry["status"] == "degraded"
    by_service = {s["service"]: s for s in entry["services"]}
    assert by_service["authelia"]["observed"] is True
    assert by_service["redis"]["observed"] is False
    assert by_service["redis"]["state"] == "missing"


def test_all_missing_reports_missing_not_degraded(fresh_manager, monkeypatch):
    authelia = fresh_manager.containers["authelia"]
    authelia.__dict__["services"] = {"authelia": {}, "redis": {}}
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    payload = collect_status(fresh_manager, names=["authelia"])
    entry = payload["containers"][0]
    assert entry["status"] == "missing"
    assert all(s["observed"] is False for s in entry["services"])


def test_json_payload_shows_observed_field(fresh_manager, monkeypatch, with_nginx_services):
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda *a, **k: _state([]))
    payload = collect_status(fresh_manager)
    svc = payload["containers"][0]["services"][0]
    assert svc == {"service": "nginx", "runtime_name": None, "state": "missing", "health": None, "observed": False}


def test_unknown_runtime_query_forces_unknown_status_even_if_all_expected(fresh_manager, monkeypatch,
                                                                          with_nginx_services):
    """queryable=False must report "unknown", never "missing" -- an
    unqueryable runtime is not evidence that nothing is running."""
    from linktools.cntr.runtime.inspect import RuntimeInspectionUnavailable

    def raise_error(*a, **k):
        raise RuntimeInspectionUnavailable("denied")

    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", raise_error)
    payload = collect_status(fresh_manager)
    assert payload["containers"][0]["status"] == "unknown"


# -- ServiceStatus is a display-only projection, never persisted ------------

def test_service_status_is_a_plain_display_object():
    status = ServiceStatus(logical_container="nginx", service="nginx", state="missing", observed=False)
    assert status.observed is False
    assert status.runtime_name is None
    assert status.health is None
