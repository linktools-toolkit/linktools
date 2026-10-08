#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated candidates preserve the active tree until validation succeeds."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.cntr.artifacts import AppliedServiceModels, GeneratedCandidate
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.container import BaseContainer, ContainerError
from linktools.cntr.runtime.compose import ComposeRunner


class _Generated(BaseContainer):
    generates_config = True
    name = "test"
    services = {"test": {}}

    def __init__(self, path):
        self.path = path
        self.manager = SimpleNamespace(data_path=path.parent,
                                       artifact_index=SimpleNamespace(record=lambda entries: None),
                                       running_state=SimpleNamespace(mark_started=lambda context: None))
        self.content = "first"

    def get_app_path(self, *parts):
        return self.path

    def render_config(self, generation_id):
        return {"config": self.content, "health": generation_id}


def test_candidate_does_not_publish_until_requested(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    assert not os.path.lexists(tmp_path / "generated/current")
    first.publish()
    old_inode = (tmp_path / "generated").stat().st_ino
    owner.content = "second"
    second = GeneratedCandidate(owner, owner.render_config)
    assert (tmp_path / "generated/current/config").read_text() == "first"
    second.publish()
    assert (tmp_path / "generated/current/config").read_text() == "second"
    assert (tmp_path / "generated").stat().st_ino == old_inode
    second.restore()
    assert (tmp_path / "generated/current/config").read_text() == "first"


def test_unchanged_candidate_reuses_generation_and_inode(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    first.publish()
    before = os.stat(first.path).st_ino
    second = GeneratedCandidate(owner, owner.render_config)
    assert not second.changed
    assert second.generation_id == first.generation_id
    assert os.stat(second.path).st_ino == before
    assert second.changed_files == ()


def test_render_failure_preserves_current(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    first.publish()
    def fail(generation):
        raise ValueError("bad template")
    with pytest.raises(ValueError, match="bad template"):
        GeneratedCandidate(owner, render=fail)
    assert GeneratedCandidate.current_id(str(owner.path)) == first.generation_id


def test_apply_failure_restores_and_confirms_old_generation(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    first.publish()
    owner.content = "second"
    second = GeneratedCandidate(owner, owner.render_config)
    calls = []
    def apply(context, candidate, services):
        calls.append(candidate.generation_id)
        if candidate.generation_id == second.generation_id:
            raise RuntimeError("new failed")
    owner.apply_config = apply
    owner.manager.generated_configs = {"test": owner}
    context = SimpleNamespace(generated_candidates={}, initial_running_services={"test"},
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={}, applied_generation_services={},
        service_models=AppliedServiceModels(owner.manager, {"services": owner.services}))
    with pytest.raises(RuntimeError, match="new failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, second, context, ("test",))
    assert calls == [second.generation_id, first.generation_id]
    assert GeneratedCandidate.current_id(str(owner.path)) == first.generation_id


def test_rollback_failure_reports_both_failures(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    first.publish()
    owner.content = "second"
    second = GeneratedCandidate(owner, owner.render_config)
    def apply(context, candidate, services):
        raise RuntimeError("new failed" if candidate.generation_id == second.generation_id else "old failed")
    owner.apply_config = apply
    owner.manager.generated_configs = {"test": owner}
    context = SimpleNamespace(generated_candidates={}, initial_running_services={"test"},
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={}, applied_generation_services={},
        service_models=AppliedServiceModels(owner.manager, {"services": owner.services}))
    with pytest.raises(ContainerError, match="new failed.*rollback failed: old failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, second, context, ("test",))


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


def test_full_configuration_scope_does_not_expand_startup_selection():
    app = SimpleNamespace(name="app")
    stopped = SimpleNamespace(name="oidc-app")
    consumer = SimpleNamespace(name="authelia")
    selected = ComposeSelection((app, consumer, stopped), (app, consumer), ("app", "authelia"), False)
    synced = selected.project_containers
    assert tuple(item.name for item in synced) == ("app", "authelia", "oidc-app")
    assert tuple(item.name for item in selected.target_containers) == ("app", "authelia")


def test_restart_validates_every_candidate_before_stopping(fresh_manager, monkeypatch):
    from test_exec_routing import _record
    recorded = _record(fresh_manager, monkeypatch)
    validated = []
    def fail(context, candidate):
        validated.append(candidate.container.name)
        raise ContainerError("native syntax failed")
    monkeypatch.setattr(fresh_manager.generated_configs["nginx"], "validate_config", fail)
    with pytest.raises(ContainerError, match="native syntax failed"):
        fresh_manager.compose_operations.restart(["portainer"])
    assert validated == ["nginx"]
    assert not any(command[0] in ("stop", "up") for command in recorded)


def test_provider_failure_does_not_apply_full_nginx(fresh_manager, monkeypatch):
    from test_exec_routing import _record
    recorded = _record(fresh_manager, monkeypatch)
    full_nginx = []
    monkeypatch.setattr(fresh_manager.generated_configs["nginx"], "apply_config",
                        lambda context, candidate, services: (
                            fresh_manager.compose_runner.apply_services(context, services)
                            if candidate.generation_id == "bootstrap" else full_nginx.append(candidate.generation_id)))
    def fail(context, candidate, services):
        raise ContainerError("authentication unavailable")
    monkeypatch.setattr(fresh_manager.generated_configs["authelia"], "apply_config", fail)
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
    declarations["portainer"] = ()
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
                              compose_owners={str(config): "app"}, initial_running_services={"app"}, applied_compose={},
                              service_models=AppliedServiceModels(SimpleNamespace(data_path=tmp_path),
                                                                 {"services": {"app": {}}}))
    owner = SimpleNamespace(name="app", services={"app": {}})
    with pytest.raises(RuntimeError, match="new process failed"):
        ComposeOperations(SimpleNamespace(compose_runner=runner))._apply_services_with_rollback(owner, context, ("app",))
    assert config.read_text() == "old"
    assert applied == [{str(config): "old"}]


def test_changed_compose_environment_prevents_generation_reuse():
    runner = ComposeRunner(SimpleNamespace())
    context = SimpleNamespace(changed_compose_services={"authelia"})
    assert not runner.is_generation_current(context, "authelia", SimpleNamespace(generation_id="same"))


def test_native_diagnostic_keeps_site_source_but_omits_secret():
    from _harness import builtin_container_type

    error = 'nginx: [emerg] invalid secret-token in /etc/nginx/generated/id/sites/s_617070_776562.conf:12'
    container = object.__new__(builtin_container_type("100-nginx"))
    container._name = "proxy"
    container.__dict__["services"] = {"nginx": {}}
    manager = SimpleNamespace(nginx_sites={("app", "web"): SimpleNamespace(template="/templates/app.j2")},
        containers={"proxy": container}, generated_configs={"proxy": container},
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs:
            SimpleNamespace(succeeded=False, stdout="", stderr=error, returncode=1)),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None))
    runner = ComposeRunner(manager)
    runner.final_model = lambda context: {}
    runner.isolated_service_args = lambda *args: []
    manager.compose_runner = runner
    container.manager = manager
    with pytest.raises(ContainerError) as raised:
        container.validate_config(SimpleNamespace(), SimpleNamespace(generation_id="id"))
    message = str(raised.value)
    assert "'app'/'web'" in message
    assert "/templates/app.j2" in message
    assert "s_617070_776562.conf:12" in message
    assert "secret-token" not in message


def test_saved_generated_model_retains_previous_generation_label():
    import yaml
    captured = []

    def process(*args):
        files = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        captured.extend(yaml.safe_load(open(path).read()) for path in files)
        return SimpleNamespace(check_call=lambda: None)

    from _harness import builtin_container_type
    container = object.__new__(builtin_container_type("102-authelia"))
    container._name = "authelia"
    manager = SimpleNamespace(project_name="test", runtime=SimpleNamespace(create_docker_process=process),
                              containers={"authelia": container}, generated_configs={"authelia": container})
    context = SimpleNamespace(generated_candidates={"authelia": SimpleNamespace(
        container=container,
        generation_id="previous")})
    ComposeRunner(manager).apply_saved_services(
        context, ("authelia",), {"old.yml": "services:\n  authelia:\n    image: authelia:old\n"})
    assert captured[-1]["services"]["authelia"]["labels"] == {"io.linktools.cntr.generation": "previous"}


def test_navigation_consumer_syncs_without_becoming_start_requirement(fresh_manager):
    operations = fresh_manager.compose_operations
    explicit = operations.select(["portainer"], metadata_only=True, for_start=True)
    selection = operations.start_selection(explicit)
    assert "flare" not in [container.name for container in selection.target_containers]
    assert "flare" in [container.name for container in selection.project_containers]
    selected_flare = operations.select(["flare"], metadata_only=True, for_start=True)
    assert "flare" in [container.name for container in operations.start_selection(selected_flare).target_containers]


def test_removed_final_declarations_still_synchronize_aggregate_consumers(fresh_manager, monkeypatch):
    producer = fresh_manager.containers["portainer"]
    monkeypatch.setattr(producer, "integrations", [])
    explicit = fresh_manager.compose_operations.select(["portainer"], metadata_only=True, for_start=True)
    selection = fresh_manager.compose_operations.start_selection(explicit)
    assert [c.name for c in selection.target_containers] == ["portainer"]
    assert {"nginx", "flare"} <= {c.name for c in selection.project_containers}
    assert "portainer" not in [c.name for c, _, _ in fresh_manager.iter_integrations("flare")]
    assert ("portainer", "web") not in fresh_manager.nginx_sites


@pytest.mark.parametrize("running", [False, True])
def test_partial_update_applies_navigation_only_if_flare_is_running(fresh_manager, monkeypatch, running):
    from test_exec_routing import _record
    from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState

    recorded = _record(fresh_manager, monkeypatch)
    services = (ServiceRuntimeState(("flare",), "flare", "flare-runtime", "running",
                                    None, "flare:test", None, {}),) if running else ()
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda containers:
                        ProjectRuntimeState(fresh_manager.project_name, services, "docker"))
    published = []

    def candidate(container, render):
        return SimpleNamespace(container=container, changed=True, generation_id="candidate", previous_id=None,
                               publish=lambda: published.append(container.name), restore=lambda: None)

    monkeypatch.setattr("linktools.cntr.artifacts.GeneratedCandidate", candidate)
    fresh_manager.compose_operations.up(["portainer"])
    assert "flare" in published
    assert any(command[0] == "up" and command[-1] == "flare" for command in recorded) is running


def test_unchanged_candidate_replaces_bootstrap_before_application(tmp_path):
    owner = _Generated(tmp_path / "generated")
    full = GeneratedCandidate(owner, owner.render_config)
    full.publish()
    unchanged = GeneratedCandidate(owner, owner.render_config)
    assert not unchanged.changed
    bootstrap = GeneratedCandidate(owner, lambda generation: {"config": "health-only", "health": generation})
    bootstrap.publish()
    applied = []

    def apply(context, candidate, services):
        applied.append(GeneratedCandidate.current_id(candidate.root))
        assert (Path(candidate.root) / "current/config").read_text() == "first"

    owner.apply_config = apply
    owner.manager.generated_configs = {"test": owner}
    context = SimpleNamespace(generated_candidates={}, initial_running_services=set(),
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={}, applied_generation_services={},
        service_models=AppliedServiceModels(owner.manager, {"services": owner.services}))
    ComposeOperations(owner.manager)._publish_candidate(owner, unchanged, context, ("test",))
    assert applied == [full.generation_id]


def test_later_generated_sibling_failure_restores_earlier_sibling_snapshots(tmp_path):
    import yaml
    from linktools.cntr.artifacts import AppliedServiceModels

    owner = _Generated(tmp_path / "generated")
    owner.services = {"test": {}, "sidecar": {}}
    old_model = {"services": {name: {"image": "old"} for name in owner.services}}
    new_model = {"services": {name: {"image": "new"} for name in owner.services}}
    AppliedServiceModels(owner.manager, old_model).record(owner.services)
    full = GeneratedCandidate(owner, owner.render_config)
    full.publish()
    owner.content = "new"
    candidate = GeneratedCandidate(owner, owner.render_config)
    calls = []

    def apply(context, value, services):
        calls.append((value.generation_id, tuple(services)))
        if value.generation_id == candidate.generation_id and services == ("sidecar",):
            raise RuntimeError("sidecar failed")

    owner.apply_config = apply
    owner.manager.generated_configs = {"test": owner}
    path = str(tmp_path / "test.yml")
    context = SimpleNamespace(generated_candidates={}, initial_running_services=set(owner.services),
        applied_generation_services={}, applied_compose={},
        saved_compose={path: yaml.safe_dump(old_model)}, compose_files={path: yaml.safe_dump(new_model)},
        compose_owners={path: "test"}, service_models=AppliedServiceModels(owner.manager, new_model))
    operations = ComposeOperations(owner.manager)
    operations._publish_candidate(owner, candidate, context, ("test",))
    with pytest.raises(RuntimeError, match="sidecar failed"):
        operations._publish_candidate(owner, candidate, context, ("sidecar",))
    assert calls[-1] == (full.generation_id, ("test", "sidecar"))
    assert yaml.safe_load((tmp_path / "compose/applied/test.yml").read_text()) == old_model
    assert not AppliedServiceModels(owner.manager, old_model).changed_services


@pytest.mark.parametrize("service,returncode,stderr", [
    ("nginx", 0, "nginx: [warn] conflicting server name secret-token"),
    ("authelia", 1, "invalid secret-token in /generated/id/configuration.yml:12"),
])
def test_asset_validator_interprets_raw_result_without_exposing_secrets(service, returncode, stderr):
    from _harness import builtin_container_type

    result = SimpleNamespace(succeeded=returncode == 0, returncode=returncode, stdout="", stderr=stderr)
    manager = SimpleNamespace(
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs: result),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None))
    runner = ComposeRunner(manager)
    manager.compose_runner = runner
    runner.final_model = lambda context: {}
    runner.isolated_service_args = lambda *args: []
    context = SimpleNamespace()
    assert runner.validate_service(context, service, [service], check=False) is result
    container = object.__new__(builtin_container_type("100-nginx" if service == "nginx" else "102-authelia"))
    container.manager = manager
    with pytest.raises(ContainerError, match="Native validation failed") as raised:
        container.validate_config(context, SimpleNamespace(generation_id="id"))
    assert "secret-token" not in str(raised.value)


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_native_validation_reuses_command_snapshot_or_resolves_fresh(has_snapshot):
    model = {"services": {"app": {"image": "app:prepared"}}}
    resolved = []
    validated = []
    runner = ComposeRunner(SimpleNamespace(
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(succeeded=True)),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None)))
    runner.final_model = lambda context: resolved.append(None) or model
    runner.isolated_service_args = lambda candidate, *args: validated.append(candidate) or []
    context = SimpleNamespace(compose_model=model) if has_snapshot else SimpleNamespace()
    for _ in range(2):
        runner.validate_service(context, "app", ("validate",))
    assert validated == [model, model]
    assert len(resolved) == (0 if has_snapshot else 2)


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_generation_image_check_reuses_command_snapshot_or_resolves_fresh(has_snapshot):
    from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState

    model = {"services": {"app": {"image": "app:prepared"}}}
    state = ServiceRuntimeState(("app",), "app", "runtime", "running", "healthy", "app:prepared", None,
                                {"io.linktools.cntr.generation": "id"}, image_id="sha256:prepared")
    resolved = []
    commands = []
    runner = ComposeRunner(SimpleNamespace(project_name="p",
        docker_inspector=SimpleNamespace(get_project_state=lambda containers: ProjectRuntimeState("p", (state,), "docker")),
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(stdout="sha256:prepared")),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: commands.append(args))))
    runner.final_model = lambda context: resolved.append(None) or model
    context = SimpleNamespace(containers=())
    if has_snapshot:
        context.compose_model = model
    for _ in range(2):
        assert runner.is_generation_current(context, "app", SimpleNamespace(generation_id="id"))
    assert len(resolved) == (0 if has_snapshot else 2)
    assert all(command[-1] == "app:prepared" for command in commands)
