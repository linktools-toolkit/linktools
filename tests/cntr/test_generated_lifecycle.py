#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated candidates preserve the active tree until validation succeeds."""
import os
from types import SimpleNamespace

import pytest

from linktools.cntr.artifacts import GeneratedCandidate
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner


class _Generated:
    name = "test"
    services = {"test": {}}

    def __init__(self, path):
        self.generated_config_path = path
        self.manager = SimpleNamespace(data_path=path.parent,
                                       artifact_index=SimpleNamespace(record=lambda entries: None))
        self.content = "first"

    def render_generated_config(self, generation_id):
        return {"config": self.content, "health": generation_id}


def test_candidate_does_not_publish_until_requested(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner)
    assert not os.path.lexists(tmp_path / "generated/current")
    first.publish()
    old_inode = (tmp_path / "generated").stat().st_ino
    owner.content = "second"
    second = GeneratedCandidate(owner)
    assert (tmp_path / "generated/current/config").read_text() == "first"
    second.publish()
    assert (tmp_path / "generated/current/config").read_text() == "second"
    assert (tmp_path / "generated").stat().st_ino == old_inode
    second.restore()
    assert (tmp_path / "generated/current/config").read_text() == "first"


def test_unchanged_candidate_reuses_generation_and_inode(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner)
    first.publish()
    before = os.stat(first.path).st_ino
    second = GeneratedCandidate(owner)
    assert not second.changed
    assert second.generation_id == first.generation_id
    assert os.stat(second.path).st_ino == before
    assert second.changed_files == ()


def test_render_failure_preserves_current(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner)
    first.publish()
    def fail(generation):
        raise ValueError("bad template")
    with pytest.raises(ValueError, match="bad template"):
        GeneratedCandidate(owner, render=fail)
    assert GeneratedCandidate.current_id(str(owner.generated_config_path)) == first.generation_id


def test_apply_failure_restores_and_confirms_old_generation(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner)
    first.publish()
    owner.content = "second"
    second = GeneratedCandidate(owner)
    calls = []
    def apply(candidate, context):
        calls.append(candidate.generation_id)
        if candidate.generation_id == second.generation_id:
            raise RuntimeError("new failed")
    owner.apply_generated_config = apply
    context = SimpleNamespace(generated_candidates={})
    with pytest.raises(RuntimeError, match="new failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, second, context, True)
    assert calls == [second.generation_id, first.generation_id]
    assert GeneratedCandidate.current_id(str(owner.generated_config_path)) == first.generation_id


def test_rollback_failure_reports_both_failures(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner)
    first.publish()
    owner.content = "second"
    second = GeneratedCandidate(owner)
    def apply(candidate, context):
        raise RuntimeError("new failed" if candidate.generation_id == second.generation_id else "old failed")
    owner.apply_generated_config = apply
    with pytest.raises(ContainerError, match="new failed.*rollback failed: old failed"):
        ComposeOperations(owner.manager)._publish_candidate(
            owner, second, SimpleNamespace(generated_candidates={}), True)


def test_isolated_validation_preserves_image_env_and_mounts_without_network_identity():
    runner = ComposeRunner(SimpleNamespace(project_name="project"))
    model = {"services": {"nginx": {"image": "nginx:target", "ports": ["80:80"],
        "depends_on": {"app": {}}, "networks": {"private": {"ipv4_address": "10.0.0.2"}},
        "environment": {"PASSWORD_FILE": "/generated/current/password"},
        "volumes": [{"type": "bind", "source": "/host/generated", "target": "/generated", "read_only": True}]}}}
    args = runner.isolated_service_args(model, "nginx", ["nginx", "-t"],
                                       {"PASSWORD_FILE": "/generated/candidate/password"})
    assert args[:5] == ["run", "--rm", "--network", "none", "--mount"]
    assert "PASSWORD_FILE=/generated/candidate/password" in args
    assert "nginx:target" in args
    assert not any(value in " ".join(args) for value in ("10.0.0.2", "80:80", "depends_on"))


def test_config_sources_expand_sync_without_starting_sources():
    app = SimpleNamespace(name="app")
    stopped = SimpleNamespace(name="oidc-app")
    consumer = SimpleNamespace(name="authelia")
    manager = SimpleNamespace(config_source_snapshot={"oidc-app": ("authelia",)},
                              resolver=SimpleNamespace(resolve_dependencies=tuple))
    selected = ComposeSelection((app, consumer, stopped), (app, consumer), ("app", "authelia"), False)
    synced = ComposeOperations(manager).sync_selection(selected)
    assert tuple(item.name for item in synced) == ("app", "authelia", "oidc-app")
    assert tuple(item.name for item in selected.target_containers) == ("app", "authelia")


def test_restart_validates_every_candidate_before_stopping(fresh_manager, monkeypatch):
    from test_exec_routing import _record
    recorded = _record(fresh_manager, monkeypatch)
    validated = []
    def fail(candidate, context):
        validated.append(candidate.container.name)
        raise ContainerError("native syntax failed")
    monkeypatch.setattr(fresh_manager.containers["nginx"], "validate_generated_config", fail)
    with pytest.raises(ContainerError, match="native syntax failed"):
        fresh_manager.compose_operations.restart(["portainer"])
    assert validated == ["nginx"]
    assert not any(command[0] in ("stop", "up") for command in recorded)


def test_provider_failure_does_not_apply_full_nginx(fresh_manager, monkeypatch):
    from test_exec_routing import _record
    recorded = _record(fresh_manager, monkeypatch)
    full_nginx = []
    monkeypatch.setattr(fresh_manager.containers["nginx"], "apply_generated_config",
                        lambda candidate, context: full_nginx.append(candidate.generation_id))
    def fail(candidate, context):
        raise ContainerError("authentication unavailable")
    monkeypatch.setattr(fresh_manager.containers["authelia"], "apply_generated_config", fail)
    with pytest.raises(ContainerError, match="authentication unavailable"):
        fresh_manager.compose_operations.up(["portainer"])
    assert not full_nginx
    assert any(command[0] == "up" and command[-1] == "nginx" for command in recorded)


def test_confirmed_generation_requires_actual_target_image():
    from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
    service = ServiceRuntimeState(("authelia",), "authelia", "runtime", "running", "healthy",
                                  "authelia:latest", None,
                                  {"io.linktools.cntr.generation": "id"}, image_id="sha256:old")
    manager = SimpleNamespace(project_name="p",
        docker_inspector=SimpleNamespace(get_project_state=lambda containers:
            ProjectRuntimeState("p", (service,), "docker")),
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(stdout="sha256:new")),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None))
    runner = ComposeRunner(manager)
    runner.final_model = lambda context: {"services": {"authelia": {"image": "authelia:latest"}}}
    context = SimpleNamespace(containers=())
    candidate = SimpleNamespace(generation_id="id")
    assert not runner.is_generation_current(context, "authelia", candidate)
    manager.structured_runner.execute = lambda *args, **kwargs: SimpleNamespace(stdout="sha256:old")
    assert runner.is_generation_current(context, "authelia", candidate)


def test_running_sync_nginx_expands_auth_provider_before_publish(fresh_manager, monkeypatch):
    from test_exec_routing import _record
    from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
    recorded = _record(fresh_manager, monkeypatch)
    declarations = dict(fresh_manager.integration_snapshot)
    declarations["portainer"] = {}
    monkeypatch.setattr(fresh_manager, "integration_snapshot", declarations)
    state = ServiceRuntimeState(("nginx",), "nginx", "nginx", "running", "healthy", "nginx:test", None, {})
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda containers:
                        ProjectRuntimeState(fresh_manager.project_name, (state,), "docker"))
    fresh_manager.compose_operations.up(["portainer"])
    starts = [command[-1] for command in recorded if command[0] == "up"]
    assert starts.index("authelia") < starts.index("nginx")
    assert {"authelia", "lldap", "safeline-mgt"} <= set(starts)


def test_compose_only_apply_failure_restores_previous_service_model(tmp_path):
    config = tmp_path / "app.yml"
    config.write_text("new")
    applied = []
    def fail(*args):
        raise RuntimeError("new process failed")
    runner = SimpleNamespace(apply_services=fail,
                             apply_saved_services=lambda context, services, files: applied.append(files))
    context = SimpleNamespace(saved_compose={str(config): "old"}, compose_files={str(config): "new"},
                              compose_owners={str(config): "app"})
    owner = SimpleNamespace(name="app", services={"app": {}})
    with pytest.raises(RuntimeError, match="new process failed"):
        ComposeOperations(SimpleNamespace(compose_runner=runner))._apply_services_with_rollback(owner, context, True)
    assert config.read_text() == "old"
    assert applied == [{str(config): "old"}]


def test_changed_compose_environment_prevents_generation_reuse():
    runner = ComposeRunner(SimpleNamespace())
    context = SimpleNamespace(changed_compose_services={"authelia"})
    assert not runner.is_generation_current(context, "authelia", SimpleNamespace(generation_id="same"))


def test_native_diagnostic_keeps_site_source_but_omits_secret():
    error = 'nginx: [emerg] invalid secret-token in /etc/nginx/generated/id/sites/s_617070_776562/business.conf:12'
    manager = SimpleNamespace(nginx_sites={("app", "web"): SimpleNamespace(template="/templates/app.j2")},
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs:
            SimpleNamespace(succeeded=False, stdout="", stderr=error, returncode=1)),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None))
    runner = ComposeRunner(manager)
    runner.final_model = lambda context: {}
    runner.isolated_service_args = lambda *args: []
    with pytest.raises(ContainerError) as raised:
        runner.validate_service(SimpleNamespace(), "nginx", ["nginx", "-t"])
    message = str(raised.value)
    assert "'app'/'web'" in message
    assert "/templates/app.j2" in message
    assert "business.conf:12" in message
    assert "secret-token" not in message


def test_saved_generated_model_retains_previous_generation_label():
    import yaml
    captured = []

    def process(*args):
        files = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        captured.extend(yaml.safe_load(open(path).read()) for path in files)
        return SimpleNamespace(check_call=lambda: None)

    manager = SimpleNamespace(project_name="test", runtime=SimpleNamespace(create_docker_process=process))
    context = SimpleNamespace(generated_candidates={"authelia": SimpleNamespace(
        container=SimpleNamespace(name="authelia"), generation_id="previous")})
    ComposeRunner(manager).apply_saved_services(
        context, ("authelia",), {"old.yml": "services:\n  authelia:\n    image: authelia:old\n"})
    assert captured[-1]["services"]["authelia"]["labels"] == {"io.linktools.cntr.generation": "previous"}
