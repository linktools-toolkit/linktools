#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DockerInspector: Compose project container-id listing (``compose ps
--quiet``) followed by a batch ``docker inspect``, service -> logical-
container mapping, disappearance-race recovery, and version probes. Uses
fake CommandResult/process stand-ins for `docker`/`docker compose` output
-- no real Docker needed."""
import pytest

from linktools.cntr.runtime.inspect import (
    DockerInspector, RuntimeInspectionOutputError, RuntimeInspectionUnavailable,
)
from linktools.cntr.runtime.structured import CommandResult, StructuredCommandError


def _service_container(fresh_manager, name, services):
    container = fresh_manager.containers[name]
    container.__dict__["services"] = {s: {} for s in services}
    return container


@pytest.fixture
def inspector(fresh_manager):
    return DockerInspector(fresh_manager)


def _stub_ids(monkeypatch, manager, stdout, returncode=0):
    result = CommandResult(args=("docker", "compose", "ps"), returncode=returncode,
                           stdout=stdout, stderr="", duration=0.01)
    monkeypatch.setattr(manager.structured_runner, "execute_text", lambda *a, **k: result)


def _stub_inspect(monkeypatch, manager, data):
    monkeypatch.setattr(manager.structured_runner, "execute_json", lambda *a, **k: data)


def _item(name="/aio-nginx", project="aio", service="nginx", status="running", **state_overrides):
    state = dict(Status=status)
    state.update(state_overrides)
    return {
        "Name": name,
        "Config": {
            "Image": "nginx:latest",
            "Labels": {
                "com.docker.compose.project": project,
                "com.docker.compose.service": service,
            } if service is not None else {"com.docker.compose.project": project},
        },
        "State": state,
    }


class _FakeProcess:
    def __init__(self, args):
        self.args = args


def _stub_dispatched(monkeypatch, manager, ps_stdout, dispatch):
    """`dispatch(container_ids)` decides the docker inspect response/error
    for whatever ids `create_docker_process("inspect", ...)` was called
    with -- used to simulate the batch-then-per-id recovery flow."""
    _stub_ids(monkeypatch, manager, ps_stdout)

    def fake_create_docker_process(*args, **kwargs):
        return _FakeProcess(args)

    def fake_execute_json(process, timeout=None, check=True):
        ids = [a for a in process.args if a not in ("inspect", "--type", "container")]
        return dispatch(ids)

    monkeypatch.setattr(manager.runtime, "create_docker_process", fake_create_docker_process)
    monkeypatch.setattr(manager.structured_runner, "execute_json", fake_execute_json)


# -- container id parsing (`compose ps --quiet`) ------------------------------

def test_empty_containers_returns_empty_state_without_touching_process(inspector, monkeypatch):
    def fail(*a, **k):
        raise AssertionError("must not attempt to create a process for an empty container list")

    monkeypatch.setattr(inspector.manager.runtime, "create_docker_compose_process", fail)
    state = inspector.get_project_state([])
    assert state.services == ()
    assert state.running_container_names == []


def test_empty_id_output_is_empty_project(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "")
    state = inspector.get_project_state([nginx])
    assert state.services == ()


def test_one_id_is_inspected(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item()])
    state = inspector.get_project_state([nginx])
    assert len(state.services) == 1
    assert state.services[0].runtime_name == "aio-nginx"


def test_multiple_ids_are_inspected(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx", "extra"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\ndef456def456\n")
    _stub_inspect(monkeypatch, fresh_manager, [
        _item(name="/aio-nginx", service="nginx"),
        _item(name="/aio-extra", service="extra"),
    ])
    state = inspector.get_project_state([nginx])
    assert {s.service for s in state.services} == {"nginx", "extra"}


def test_duplicate_ids_are_deduped_stably(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    seen_ids = []
    real_create_docker_process = fresh_manager.runtime.create_docker_process

    def fake_create_docker_process(*args, **kwargs):
        # create_docker_compose_process() itself routes through
        # create_docker_process("compose", ...) to build the ps --quiet
        # process -- only the "inspect" invocation is under test here.
        if args and args[0] == "inspect":
            seen_ids.append([a for a in args if a not in ("inspect", "--type", "container")])
            return _FakeProcess(args)
        return real_create_docker_process(*args, **kwargs)

    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\nabc123abc123\ndef456def456\nabc123abc123\n")
    monkeypatch.setattr(fresh_manager.runtime, "create_docker_process", fake_create_docker_process)
    monkeypatch.setattr(fresh_manager.structured_runner, "execute_json", lambda *a, **k: [_item()])

    inspector.get_project_state([nginx])
    assert seen_ids == [["abc123abc123", "def456def456"]]


def test_invalid_id_line_raises_output_error(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "not-a-hex-id\n")
    with pytest.raises(RuntimeInspectionOutputError):
        inspector.get_project_state([nginx])


def test_ps_quiet_failure_raises_unavailable(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def raise_error(*a, **k):
        raise StructuredCommandError("compose ps failed")

    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text", raise_error)
    with pytest.raises(RuntimeInspectionUnavailable):
        inspector.get_project_state([nginx])


def test_docker_binary_missing_raises_unavailable(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def raise_os_error(*a, **k):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(fresh_manager.structured_runner, "execute_text", raise_os_error)
    with pytest.raises(RuntimeInspectionUnavailable):
        inspector.get_project_state([nginx])


# -- docker inspect output validation -----------------------------------------

def test_inspect_root_not_array_raises_output_error(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, {"not": "a list"})
    with pytest.raises(RuntimeInspectionOutputError):
        inspector.get_project_state([nginx])


def test_inspect_item_not_object_raises_output_error(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, ["not-an-object"])
    with pytest.raises(RuntimeInspectionOutputError):
        inspector.get_project_state([nginx])


def test_empty_inspect_array_for_known_ids_raises_output_error(inspector, fresh_manager, monkeypatch):
    """A returncode-0 `docker inspect` that reports zero results for ids we
    just listed is corruption, not "everything vanished" -- that legitimate
    case only applies to the per-id recovery path."""
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [])
    with pytest.raises(RuntimeInspectionOutputError):
        inspector.get_project_state([nginx])


# -- state normalization -------------------------------------------------------

@pytest.mark.parametrize("status", ["running", "restarting", "paused", "exited", "dead", "created", "removing"])
def test_state_status_is_normalized_lowercase(inspector, fresh_manager, monkeypatch, status):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item(status=status)])
    state = inspector.get_project_state([nginx])
    assert state.services[0].state == status


def test_missing_status_falls_back_to_running_bool(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    item = _item(status="")
    item["State"] = {"Running": True}
    _stub_inspect(monkeypatch, fresh_manager, [item])
    state = inspector.get_project_state([nginx])
    assert state.services[0].state == "running"


def test_missing_status_and_all_bools_false_is_unknown(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    item = _item(status="")
    item["State"] = {"Running": False, "Restarting": False, "Paused": False, "Dead": False}
    _stub_inspect(monkeypatch, fresh_manager, [item])
    state = inspector.get_project_state([nginx])
    assert state.services[0].state == "unknown"
    # An empty/unrecognized state must never be silently treated as exited.
    assert state.services[0].state != "exited"


def test_health_status_is_read_from_state_health(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    item = _item(Health={"Status": "unhealthy"})
    _stub_inspect(monkeypatch, fresh_manager, [item])
    state = inspector.get_project_state([nginx])
    assert state.services[0].health == "unhealthy"


def test_missing_health_is_none(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item()])
    state = inspector.get_project_state([nginx])
    assert state.services[0].health is None


def test_string_exit_code_is_converted(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    item = _item(status="exited", ExitCode="137")
    _stub_inspect(monkeypatch, fresh_manager, [item])
    state = inspector.get_project_state([nginx])
    assert state.services[0].exit_code == 137


def test_non_numeric_exit_code_is_none(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    item = _item(status="exited", ExitCode="not-a-number")
    _stub_inspect(monkeypatch, fresh_manager, [item])
    state = inspector.get_project_state([nginx])
    assert state.services[0].exit_code is None


# -- project/service filtering -------------------------------------------------

def test_other_project_container_is_filtered_out(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item(project="some-other-project")])
    state = inspector.get_project_state([nginx])
    assert state.services == ()


def test_unknown_service_has_no_logical_container(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item(name="/orphan-1", service="orphan")])
    state = inspector.get_project_state([nginx])
    assert state.services[0].logical_container is None
    # An orphan/unknown service must not count toward any logical container's
    # running set.
    assert state.running_container_names == []


def test_duplicate_service_across_containers_is_first_wins(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["shared"])
    authelia = _service_container(fresh_manager, "authelia", ["shared"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item(service="shared")])
    state = inspector.get_project_state([nginx, authelia])
    assert state.services[0].logical_container == "nginx"


def test_running_container_names_requires_running_or_restarting(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["a", "b"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\ndef456def456\n")
    _stub_inspect(monkeypatch, fresh_manager, [
        _item(name="/x1", service="a", status="restarting"),
        _item(name="/x2", service="b", status="exited"),
    ])
    state = inspector.get_project_state([nginx])
    # >=1 service running/restarting is enough for the binary compat contract.
    assert state.running_container_names == ["nginx"]


def test_running_container_names_excludes_fully_stopped_container(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["a"])
    _stub_ids(monkeypatch, fresh_manager, "abc123abc123\n")
    _stub_inspect(monkeypatch, fresh_manager, [_item(name="/x1", service="a", status="exited")])
    state = inspector.get_project_state([nginx])
    assert state.running_container_names == []


def test_project_and_backend_are_from_manager(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])
    _stub_ids(monkeypatch, fresh_manager, "")
    state = inspector.get_project_state([nginx])
    assert state.project == fresh_manager.project_name
    assert state.backend == fresh_manager.container_type
    assert state.source == "docker-inspect"


# -- disappearance race recovery (Spec section 13) ----------------------------

def test_bulk_inspect_failure_recovers_via_individual_inspect(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def dispatch(ids):
        if len(ids) > 1:
            raise StructuredCommandError("bulk inspect failed")
        [container_id] = ids
        if container_id == "dead0000dead":
            raise StructuredCommandError("Error: No such container: dead0000dead")
        return [_item()]

    _stub_dispatched(monkeypatch, fresh_manager, "dead0000dead\nabc123abc123\n", dispatch)
    state = inspector.get_project_state([nginx])
    assert len(state.services) == 1
    assert state.services[0].service == "nginx"


def test_all_containers_vanished_returns_empty_state_not_error(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def dispatch(ids):
        if len(ids) > 1:
            raise StructuredCommandError("bulk inspect failed")
        raise StructuredCommandError("Error: No such container")

    _stub_dispatched(monkeypatch, fresh_manager, "dead0000dead\ndead1111dead\n", dispatch)
    state = inspector.get_project_state([nginx])
    assert state.services == ()


def test_recovery_non_not_found_error_propagates(inspector, fresh_manager, monkeypatch):
    nginx = _service_container(fresh_manager, "nginx", ["nginx"])

    def dispatch(ids):
        if len(ids) > 1:
            raise StructuredCommandError("bulk inspect failed")
        raise StructuredCommandError("permission denied")

    _stub_dispatched(monkeypatch, fresh_manager, "abc123abc123\n", dispatch)
    with pytest.raises(RuntimeInspectionUnavailable):
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
