#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DockerInspector: docker compose ps JSON parsing tolerance, service ->
logical-container mapping, and version probes (Spec Part II). Uses fake
CommandResult stand-ins for `docker`/`docker compose` output -- no real
Docker needed."""
import pytest

from linktools.cntr.runtime.inspect import DockerInspector, ProjectRuntimeState
from linktools.cntr.runtime.structured import CommandResult, StructuredCommandError


def _stub_execute_text(monkeypatch, manager, stdout, returncode=0, succeeded=None):
    result = CommandResult(
        args=("docker", "compose", "ps"), returncode=returncode,
        stdout=stdout, stderr="", duration=0.01,
    )
    monkeypatch.setattr(manager.structured_runner, "execute_text", lambda *a, **k: result)
    return result


def _service_container(fresh_manager, name, services):
    container = fresh_manager.containers[name]
    container.__dict__["services"] = {s: {} for s in services}
    return container


@pytest.fixture
def inspector(fresh_manager):
    return DockerInspector(fresh_manager)


def test_empty_containers_returns_empty_state_without_touching_process(inspector, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("must not attempt to create a process for an empty container list")

    monkeypatch.setattr(inspector.manager.runtime, "create_docker_compose_process", fail)
    state = inspector.get_project_state([])
    assert state.services == ()
    assert state.running_container_names == []


def test_parses_json_array(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager, (
        '[{"Name": "aio-nginx", "Service": "nginx", "State": "running", '
        '"Health": "healthy", "Image": "nginx:1", "ExitCode": 0}]'
    ))
    state = inspector.get_project_state([nginx])
    assert len(state.services) == 1
    svc = state.services[0]
    assert svc.logical_container == "nginx"
    assert svc.runtime_name == "aio-nginx"
    assert svc.state == "running"
    assert svc.health == "healthy"


def test_parses_line_delimited_json(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx", "extra"])
    stdout = (
        '{"Name": "aio-nginx", "Service": "nginx", "State": "running"}\n'
        '{"Name": "aio-extra", "Service": "extra", "State": "exited"}\n'
    )
    _stub_execute_text(monkeypatch, fresh_manager, stdout)
    state = inspector.get_project_state([nginx])
    assert {s.service for s in state.services} == {"nginx", "extra"}


def test_parses_single_json_object(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager, '{"Name": "aio-nginx", "Service": "nginx", "State": "running"}')
    state = inspector.get_project_state([nginx])
    assert len(state.services) == 1


def test_handles_empty_output(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager, "")
    state = inspector.get_project_state([nginx])
    assert state.services == ()


def test_unknown_field_is_ignored(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager,
                       '[{"Name": "aio-nginx", "Service": "nginx", "State": "running", "SomeNewField": 42}]')
    state = inspector.get_project_state([nginx])
    assert state.services[0].service == "nginx"


def test_unknown_service_has_no_logical_container(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager,
                       '[{"Name": "orphan-1", "Service": "orphan", "State": "running"}]')
    state = inspector.get_project_state([nginx])
    assert state.services[0].logical_container is None
    # An orphan/unknown service must not count toward any logical container's
    # running set.
    assert state.running_container_names == []


def test_labels_string_form_is_parsed_to_dict(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager,
                       '[{"Name": "aio-nginx", "Service": "nginx", "State": "running", "Labels": "a=1,b=2"}]')
    state = inspector.get_project_state([nginx])
    assert state.services[0].labels == {"a": "1", "b": "2"}


def test_labels_missing_defaults_to_empty_dict(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager,
                       '[{"Name": "aio-nginx", "Service": "nginx", "State": "running"}]')
    state = inspector.get_project_state([nginx])
    assert state.services[0].labels == {}


def test_running_container_names_requires_running_or_restarting(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["a", "b"])
    _stub_execute_text(monkeypatch, fresh_manager, (
        '[{"Name": "x1", "Service": "a", "State": "restarting"},'
        ' {"Name": "x2", "Service": "b", "State": "exited"}]'
    ))
    state = inspector.get_project_state([nginx])
    # >=1 service running/restarting is enough for the binary compat contract.
    assert state.running_container_names == ["nginx"]


def test_running_container_names_excludes_fully_stopped_container(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["a"])
    _stub_execute_text(monkeypatch, fresh_manager, '[{"Name": "x1", "Service": "a", "State": "exited"}]')
    state = inspector.get_project_state([nginx])
    assert state.running_container_names == []


def test_duplicate_service_across_containers_is_first_wins(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["shared"])
    authelia = _service_container(fresh_manager, "authelia", ["shared"])
    _stub_execute_text(monkeypatch, fresh_manager, '[{"Name": "x1", "Service": "shared", "State": "running"}]')
    state = inspector.get_project_state([nginx, authelia])
    assert state.services[0].logical_container == "nginx"


def test_project_and_backend_are_from_manager(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_execute_text(monkeypatch, fresh_manager, "")
    state = inspector.get_project_state([nginx])
    assert state.project == fresh_manager.project_name
    assert state.backend == fresh_manager.container_type


def test_nonzero_exit_raises_structured_command_error(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def raise_error(*a, **k):
        raise StructuredCommandError("docker compose ps failed")

    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text", raise_error)
    with pytest.raises(StructuredCommandError):
        inspector.get_project_state([nginx])


# -- version probes -----------------------------------------------------------

def test_get_engine_version_parses_client_and_server(inspector, fresh_manager, monkeypatch):
    data = {"Client": {"Version": "24.0.5", "ApiVersion": "1.43"}, "Server": {"Version": "24.0.6"}}
    monkeypatch.setattr(fresh_manager.structured_runner, "execute_json", lambda *a, **k: data)
    version = inspector.get_engine_version()
    assert version.client == "24.0.5"
    assert version.server == "24.0.6"
    assert version.api == "1.43"


def test_get_engine_version_on_failure_returns_all_none(inspector, fresh_manager, monkeypatch):
    def raise_error(*a, **k):
        raise StructuredCommandError("docker version failed")

    monkeypatch.setattr(fresh_manager.structured_runner, "execute_json", raise_error)
    version = inspector.get_engine_version()
    assert version.client is None
    assert version.server is None
    assert version.api is None


def test_get_compose_version_strips_output(inspector, fresh_manager, monkeypatch):
    result = CommandResult(args=(), returncode=0, stdout="2.24.5\n", stderr="", duration=0.0)
    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text", lambda *a, **k: result)
    assert inspector.get_compose_version() == "2.24.5"


def test_get_compose_version_on_failure_returns_none(inspector, fresh_manager, monkeypatch):
    def raise_error(*a, **k):
        raise StructuredCommandError("docker compose not found")

    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text", raise_error)
    assert inspector.get_compose_version() is None


def test_validate_compose_uses_config_args_quiet(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    recorded = []

    def fake_create(containers, *args, privilege=None, **kwargs):
        recorded.append(args)

        class _Proc:
            pass

        return _Proc()

    monkeypatch.setattr(fresh_manager.runtime, "create_docker_compose_process", fake_create)
    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text",
                        lambda *a, **k: CommandResult(args=(), returncode=0, stdout="", stderr="", duration=0.0))
    inspector.validate_compose([nginx])
    assert recorded[0] == ("config", "--quiet")
