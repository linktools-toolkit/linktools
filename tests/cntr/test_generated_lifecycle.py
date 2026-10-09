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
                                       artifact_index=SimpleNamespace(record=lambda entries, remove=(): None),
                                       running_state=SimpleNamespace(mark_started=lambda context: None))
        self.manager.compose_runner = ComposeRunner(self.manager)
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
    AppliedServiceModels(owner.manager, {"services": owner.services}).record(("test",))
    context = SimpleNamespace(generated_candidates={}, initial_running_services={"test"},
        native_running_images={"test": "sha256:test"}, containers=(owner,), initial_healthy_services=set(),
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
    AppliedServiceModels(owner.manager, {"services": owner.services}).record(("test",))
    context = SimpleNamespace(generated_candidates={}, initial_running_services={"test"},
        native_running_images={"test": "sha256:test"}, containers=(owner,), initial_healthy_services=set(),
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={}, applied_generation_services={},
        service_models=AppliedServiceModels(owner.manager, {"services": owner.services}))
    with pytest.raises(ContainerError, match="new failed.*rollback failed: old failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, second, context, ("test",))


def test_first_upgrade_failure_restores_running_service_without_previous_generation(tmp_path):
    owner = _Generated(tmp_path / "generated")
    candidate = GeneratedCandidate(owner, owner.render_config)
    assert candidate.previous_id is None
    old_compose = tmp_path / "legacy.yml"
    old_compose.write_text("services:\n  test:\n    image: legacy:test\n")
    calls = []
    owner.manager.compose_runner = SimpleNamespace(
        apply_saved_services=lambda context, services, files: calls.append(
            ("restore", tuple(services), tuple(files.values()))),
        wait_service_running=lambda context, service: calls.append(("running", service)),
        wait_service_healthy=lambda context, service: calls.append(("healthy", service)),
        saved_service_models=lambda context, services: {
            service: old_compose.read_text() for service in services},
    )
    owner.manager.running_state = SimpleNamespace(
        mark_started=lambda context: calls.append(("state", context.target_containers[0].name)),
    )
    owner.rollback_config = lambda context: calls.append(("native",))
    owner.apply_config = lambda context, candidate, services: (_ for _ in ()).throw(
        RuntimeError("new failed"))
    context = SimpleNamespace(
        generated_candidates={}, initial_running_services={"test"}, initial_healthy_services={"test"},
        native_running_images={"test": "sha256:test"}, containers=(owner,),
        saved_compose={str(old_compose): old_compose.read_text()},
        compose_files={str(old_compose): "services:\n  test:\n    image: new:test\n"},
        compose_owners={str(old_compose): "test"}, applied_compose={},
        applied_generation_services={}, service_models=AppliedServiceModels(
            owner.manager, {"services": owner.services}),
    )
    with pytest.raises(RuntimeError, match="new failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, candidate, context, ("test",))
    assert GeneratedCandidate.current_id(str(owner.path)) is None
    assert calls == [
        ("native",),
        ("restore", ("test",), ("services:\n  test:\n    image: legacy:test\n",)),
        ("healthy", "test"),
        ("state", "test"),
    ]


def test_first_deployment_failure_does_not_start_unrelated_services(tmp_path):
    owner = _Generated(tmp_path / "generated")
    candidate = GeneratedCandidate(owner, owner.render_config)
    calls = []
    owner.rollback_config = lambda context: calls.append("native")
    owner.apply_config = lambda context, candidate, services: (_ for _ in ()).throw(
        RuntimeError("first failed"))
    owner.manager.compose_runner = SimpleNamespace(
        stop=lambda context, services: calls.append(("stop", tuple(services))),
        apply_saved_services=lambda *args: pytest.fail("unexpected rollback deployment"))
    owner.manager.running_state = SimpleNamespace(
        mark_stopped=lambda context: calls.append(("stopped",)),
        mark_started=lambda context: pytest.fail("unexpected restored running service"))
    context = SimpleNamespace(
        generated_candidates={}, initial_running_services=set(),
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={},
        applied_generation_services={}, service_models=AppliedServiceModels(
            owner.manager, {"services": owner.services}),
    )
    with pytest.raises(RuntimeError, match="first failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, candidate, context, ("test",))
    assert calls == [("stop", ("test",)), "native", ("stopped",)]
    assert GeneratedCandidate.current_id(str(owner.path)) is None


def test_partial_cold_generated_start_failure_stops_all_new_siblings(tmp_path):
    owner = _Generated(tmp_path / "generated")
    owner.services = {"test": {}, "sidecar": {}}
    candidate = GeneratedCandidate(owner, owner.render_config)
    events = []

    def apply(context, value, services):
        events.append(("apply", tuple(services)))
        if services == ("sidecar",):
            raise RuntimeError("sidecar failed")

    owner.apply_config = apply
    owner.manager.compose_runner = SimpleNamespace(
        stop=lambda context, services: events.append(("stop", tuple(services))),
    )
    owner.manager.running_state = SimpleNamespace(
        mark_stopped=lambda context: events.append(("stopped",)),
        mark_started=lambda context: pytest.fail("unexpected restarted old service"))
    compose_path = tmp_path / "test.yml"
    compose_path.write_text(
        "services:\n  test:\n    image: new\n  sidecar:\n    image: new\n")
    context = SimpleNamespace(
        generated_candidates={}, initial_running_services=set(),
        saved_compose={}, compose_files={str(compose_path): compose_path.read_text()},
        compose_owners={str(compose_path): "test"}, applied_compose={},
        applied_generation_services={}, service_models=AppliedServiceModels(
            owner.manager, {"services": owner.services}),
    )
    operations = ComposeOperations(owner.manager)
    operations._publish_candidate(owner, candidate, context, ("test",))
    assert (tmp_path / "compose/applied/test.yml").exists()
    with pytest.raises(RuntimeError, match="sidecar failed"):
        operations._publish_candidate(owner, candidate, context, ("sidecar",))
    assert events == [
        ("apply", ("test",)), ("apply", ("sidecar",)),
        ("stop", ("test", "sidecar")), ("stopped",),
    ]
    assert GeneratedCandidate.current_id(str(owner.path)) is None
    assert not (tmp_path / "compose/applied/test.yml").exists()
    assert not (tmp_path / "compose/applied/services" / "74657374.yml").exists()


def test_first_upgrade_reports_unrecoverable_missing_compose_snapshot(tmp_path):
    owner = _Generated(tmp_path / "generated")
    candidate = GeneratedCandidate(owner, owner.render_config)
    owner.apply_config = lambda context, candidate, services: (_ for _ in ()).throw(
        RuntimeError("new failed"))
    owner.manager.compose_runner = SimpleNamespace()
    context = SimpleNamespace(
        generated_candidates={}, initial_running_services={"test"},
        native_running_images={"test": "sha256:test"}, containers=(owner,),
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={},
        applied_generation_services={}, service_models=AppliedServiceModels(
            owner.manager, {"services": owner.services}),
    )
    with pytest.raises(ContainerError, match="Cannot replace running service test without a previous Compose model"):
        ComposeOperations(owner.manager)._publish_candidate(owner, candidate, context, ("test",))
    assert GeneratedCandidate.current_id(str(owner.path)) is None


def test_cold_bootstrap_ignores_saved_compose_while_other_restores_use_it():
    commands = []
    applied_saved = []
    runner = ComposeRunner(SimpleNamespace(
        runtime=SimpleNamespace(create_docker_compose_process=lambda containers, *args:
            commands.append(args) or SimpleNamespace(check_call=lambda: 0))))
    runner.wait_service_dependencies = lambda *args: None
    runner.apply_saved_services = lambda context, services, files: applied_saved.append(
        (tuple(services), files))
    context = SimpleNamespace(
        containers=(), is_full_containers=False, generated_candidates={},
        rollback_service_models={"nginx": "services:\n  nginx:\n    image: old\n"},
        rollback_compose_files={"old.yml": "services:\n  nginx:\n    image: old\n"},
        bootstrap_fallback_services={"nginx"},
    )
    runner.apply_service(context, "nginx")
    assert applied_saved == []
    assert commands and commands[0][-1] == "nginx"
    runner.apply_service(context, "other")
    assert applied_saved == [(("other",), context.rollback_compose_files)]


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


def test_isolated_mount_override_replaces_only_certificate_volume():
    runner = ComposeRunner(SimpleNamespace(project_name="project"))
    model = {"services": {"nginx": {"image": "nginx:target", "volumes": [
        {"type": "bind", "source": "/host/certs", "target": "/etc/certs"},
        {"type": "bind", "source": "/host/generated", "target": "/etc/nginx/generated", "read_only": True},
    ]}}}
    args = runner.isolated_service_args(model, "nginx", ("nginx", "-t"),
                                        mount_overrides={"/etc/certs": "/host/candidate"})
    assert "type=bind,source=/host/candidate,target=/etc/certs,readonly" in args
    assert "type=bind,source=/host/generated,target=/etc/nginx/generated,readonly" in args
    assert "/host/certs" not in " ".join(args)


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
    state = ServiceRuntimeState(("nginx",), "nginx", "nginx", "running", "healthy", "nginx:test", None, {},
                                image_id="sha256:nginx")
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
                             saved_service_models=lambda context, services: {"app": "old"},
                             wait_service_running=lambda context, service: None,
                             apply_saved_services=lambda context, services, files: applied.append(files))
    context = SimpleNamespace(saved_compose={str(config): "old"}, compose_files={str(config): "new"},
                              compose_owners={str(config): "app"}, initial_running_services={"app"}, applied_compose={},
                              initial_healthy_services=set(),
                              service_models=AppliedServiceModels(SimpleNamespace(data_path=tmp_path,
                                  artifact_index=SimpleNamespace(record=lambda entries, remove=(): None)),
                                                                 {"services": {"app": {}}}))
    owner = SimpleNamespace(name="app", services={"app": {}},
                            on_service_started=lambda context, service: None)
    started = []
    manager = SimpleNamespace(compose_runner=runner, running_state=SimpleNamespace(mark_started=started.append))
    with pytest.raises(RuntimeError, match="new process failed"):
        ComposeOperations(manager)._apply_services_with_rollback(owner, context, ("app",))
    assert config.read_text() == "old"
    assert applied == [{"previous.yml": "old"}]
    assert len(started) == 1
    assert started[0].target_containers == [owner]
    assert started[0].is_full_containers is False


def test_changed_compose_environment_prevents_generation_reuse():
    runner = ComposeRunner(SimpleNamespace())
    context = SimpleNamespace(changed_compose_services={"authelia"})
    assert not runner.is_generation_current(context, "authelia", SimpleNamespace(generation_id="same"))


def test_native_diagnostic_keeps_site_source_but_omits_secret(tmp_path):
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
        container.validate_config(SimpleNamespace(), SimpleNamespace(generation_id="id", path=str(tmp_path)))
    message = str(raised.value)
    assert "'app'/'web'" in message
    assert "/templates/app.j2" in message
    assert "s_617070_776562.conf:12" in message
    assert "secret-token" not in message


def test_saved_generated_model_retains_previous_generation_label(tmp_path):
    import yaml
    captured = []

    def process(*args, **kwargs):
        files = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        captured.extend(yaml.safe_load(Path(path).read_text()) for path in files)
        return SimpleNamespace(check_call=lambda: None)

    from _harness import builtin_container_type
    container = object.__new__(builtin_container_type("102-authelia"))
    container._name = "authelia"
    container.__dict__["services"] = {"authelia": {}}
    manager = SimpleNamespace(project_name="test", runtime=SimpleNamespace(create_docker_process=process),
        structured_runner=SimpleNamespace(execute_json=lambda process, **kwargs: {
            "services": {"authelia": {"image": "sha256:old"}}}),
        containers={"authelia": container}, generated_configs={"authelia": container})
    context = SimpleNamespace(containers=(container,), compose_files={str(tmp_path / "old.yml"): ""},
        native_running_images={"authelia": "sha256:old"},
        generated_candidates={"authelia": SimpleNamespace(container=container, generation_id="previous")})
    ComposeRunner(manager).apply_saved_services(
        context, ("authelia",), {"old.yml": "services:\n  authelia:\n    image: authelia:old\n"})
    assert captured[-1]["services"]["authelia"]["labels"] == {"io.linktools.cntr.generation": "previous"}
    assert captured[-1]["services"]["authelia"]["image"] == "sha256:old"


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
    if running:
        from linktools.cntr.artifacts import compose_candidate
        path, content = compose_candidate(fresh_manager.containers["flare"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    services = (ServiceRuntimeState(("flare",), "flare", "flare-runtime", "running",
                                    None, "flare:test", None, {}, image_id="sha256:flare"),) if running else ()
    monkeypatch.setattr(fresh_manager.docker_inspector, "get_project_state", lambda containers:
                        ProjectRuntimeState(fresh_manager.project_name, services, "docker"))
    published = []

    def candidate(container, render):
        return SimpleNamespace(container=container, changed=True, generation_id="candidate", previous_id=None,
                               publish=lambda: published.append(container.name), restore=lambda: None)

    monkeypatch.setattr("linktools.cntr.artifacts.GeneratedCandidate", candidate)
    fresh_manager.compose_operations.up(["portainer"])
    assert ("flare" in published) is running
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
        native_running_images={name: "sha256:" + name for name in owner.services}, containers=(owner,),
        applied_generation_services={}, applied_compose={},
        saved_compose={path: yaml.safe_dump(old_model)}, compose_files={path: yaml.safe_dump(new_model)},
        compose_owners={path: "test"}, service_models=AppliedServiceModels(owner.manager, new_model))
    operations = ComposeOperations(owner.manager)
    operations._publish_candidate(owner, candidate, context, ("test",))
    with pytest.raises(RuntimeError, match="sidecar failed"):
        operations._publish_candidate(owner, candidate, context, ("sidecar",))
    assert calls[-2:] == [(full.generation_id, ("test",)), (full.generation_id, ("sidecar",))]
    assert yaml.safe_load((tmp_path / "compose/applied/test.yml").read_text()) == old_model
    assert not AppliedServiceModels(owner.manager, old_model).changed_services


@pytest.mark.parametrize("service,returncode,stderr", [
    ("nginx", 0, "nginx: [warn] conflicting server name secret-token"),
    ("authelia", 1, "invalid secret-token in /generated/id/configuration.yml:12"),
])
def test_asset_validator_interprets_raw_result_without_exposing_secrets(service, returncode, stderr, tmp_path):
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
        container.validate_config(context, SimpleNamespace(generation_id="id", path=str(tmp_path)))
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


@pytest.mark.parametrize("image_present,selected", [(False, False), (True, False), (False, True)])
def test_unselected_running_native_validation_uses_available_image_identity(image_present, selected):
    model = {"services": {"nginx": {"image": "nginx:target"}}}
    used = []
    inspected = []
    preparer = SimpleNamespace(image_exists=lambda image: inspected.append(image) or image_present)
    manager = SimpleNamespace(
        image_preparer=preparer,
        structured_runner=SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(succeeded=True)),
        runtime=SimpleNamespace(create_docker_process=lambda *args, **kwargs: None),
    )
    runner = ComposeRunner(manager)
    runner.isolated_service_args = lambda actual, *args: used.append(actual) or []
    context = SimpleNamespace(
        compose_model=model, native_running_images={"nginx": "sha256:running"},
        image_preparation_targets={"nginx"} if selected else set(),
    )
    runner.validate_service(context, "nginx", ("nginx", "-t"))
    expected = "nginx:target" if selected or image_present else "sha256:running"
    assert used[0]["services"]["nginx"]["image"] == expected
    assert model["services"]["nginx"]["image"] == "nginx:target"
    assert inspected == ([] if selected else ["nginx:target"])


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


def test_publication_failure_restores_original_generation_before_runtime_apply(tmp_path):
    owner = _Generated(tmp_path / "generated")
    first = GeneratedCandidate(owner, owner.render_config)
    first.publish()
    owner.content = "second"
    candidate = GeneratedCandidate(owner, owner.render_config)
    models = AppliedServiceModels(owner.manager, {"services": owner.services})
    models.record(("test",))
    models = AppliedServiceModels(owner.manager, {"services": owner.services})
    calls = []
    owner.apply_config = lambda context, value, services: calls.append(value.generation_id)
    owner.manager.compose_runner = SimpleNamespace(
        saved_service_models=lambda context, services: {
            "test": models.previous["test"]})
    context = SimpleNamespace(
        generated_candidates={}, initial_running_services={"test"},
        native_running_images={"test": "sha256:test"}, containers=(owner,),
        saved_compose={}, compose_files={}, compose_owners={}, applied_compose={},
        applied_generation_services={}, service_models=models)

    original_publish = candidate.publish

    def fail_after_publish():
        original_publish()
        raise RuntimeError("publication failed")

    candidate.publish = fail_after_publish
    with pytest.raises(RuntimeError, match="publication failed"):
        ComposeOperations(owner.manager)._publish_candidate(owner, candidate, context, ("test",))
    assert GeneratedCandidate.current_id(str(owner.path)) == first.generation_id
    assert calls == [first.generation_id]
