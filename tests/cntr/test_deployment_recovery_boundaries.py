#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migration inputs and dependent namespaces survive failed deployments."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import replace

import pytest
import yaml

from linktools.cntr._container.compose import write_docker_compose_file
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner
from linktools.cntr.runtime.inspect import ProjectRuntimeState
from linktools.cntr.runtime.process import RuntimeProcessFactory
from test_lifecycle_rebuild import setup_case


@pytest.mark.parametrize("saved_snapshot", [False, True])
def test_inspection_cannot_overwrite_migration_recovery_input(tmp_path, saved_snapshot):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "environment": {"VERSION": "new"}}}),
    ], running=("app",))
    previous = {"services": {"app": {"image": "app:old", "environment": {"VERSION": "old"}}}}
    snapshots = AppliedServiceModels(manager, previous)
    snapshots.record(("app",))
    if not saved_snapshot:
        for path in (tmp_path / "compose" / "applied" / "services").iterdir():
            path.unlink()
    compose_file = tmp_path / "compose" / "app.yml"
    compose_file.write_text(yaml.safe_dump(previous))
    owner = manager.containers["app"]
    owner.repo_context = None
    owner.get_source_path = lambda name: tmp_path / "source" / name
    owner.get_docker_compose_file = lambda: write_docker_compose_file(owner)
    manager.docker_compose_names = ("compose.yml",)
    manager.env_config = SimpleNamespace(get=lambda *args, **kwargs: None)
    manager.container_type = "docker-rootless"
    manager.runtime = RuntimeProcessFactory(manager)
    manager.runtime.create_process = lambda *args, **kwargs: SimpleNamespace(args=args)
    actual = manager.docker_inspector.get_project_state(None)

    def inspect(containers):
        manager.runtime.docker_compose_args(containers, "ps", "--all", "--quiet")
        return actual

    manager.docker_inspector.get_project_state = inspect
    runner = ComposeRunner(manager)

    def resolve(process):
        paths = [process.args[index + 1] for index, arg in enumerate(process.args[:-1])
                 if arg == "--file"]
        assert len(paths) == 1
        return yaml.safe_load(Path(paths[0]).read_text())

    runner._resolved_model = resolve
    manager.compose_runner.saved_service_models = runner.saved_service_models

    def check(context):
        assert yaml.safe_load(compose_file.read_text())["services"]["app"]["environment"] == {"VERSION": "new"}
        old = yaml.safe_load(context.previous_compose_contents[str(compose_file)])
        if not saved_snapshot:
            assert old["services"]["app"]["environment"] == {"VERSION": "old"}
        recovered = yaml.safe_load(runner.saved_service_models(context, ("app",))["app"])
        assert recovered["services"]["app"]["environment"] == {"VERSION": "old"}
        raise ContainerError("stop before application")

    owner.on_check = check
    with pytest.raises(ContainerError, match="stop before application"):
        manager.compose_operations.up(["app"])
    assert not any(event[0] in ("apply", "stop", "restore") for event in manager.events)


@pytest.mark.parametrize("names", [None, ["app"]])
def test_running_orphan_does_not_abort_after_applying_current_services(tmp_path, names):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    actual = manager.docker_inspector.get_project_state(None)
    actual = replace(actual, services=actual.services + (SimpleNamespace(service="retired", state="running", image_id="sha256:retired",
                                       labels={}, health=None, exit_code=None),))
    manager.docker_inspector.get_project_state = lambda containers: actual
    manager.compose_operations.up(names)
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert ("after-callback", "app") in manager.events
    assert not any(event[0] == "restore" for event in manager.events)


@pytest.mark.parametrize("binding", [
    {"network_mode": "service:db"},
    {"ipc": "service:db"},
    {"pid": "service:db"},
    {"volumes_from": ["db:ro"]},
    {"depends_on": {"db": {"restart": True}}},
])
def test_recovery_rebinds_running_dependents_without_starting_stopped_siblings(tmp_path, binding):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": dict(binding, image="net:new")}),
        ("child", {"child": {"image": "child:new", "network_mode": "service:net"}}),
        ("stopped", {"stopped": {"image": "stopped:new", "network_mode": "service:db"}}),
        ("other", {"other": {"image": "other:new"}}),
    ], running=("db", "net", "child", "other"))
    manager.compose_runner.fail = "db"
    with pytest.raises(ContainerError, match="apply failed db"):
        manager.compose_operations.up(["db"])
    restored = [event[1] for event in manager.events if event[0] == "restore"]
    if "depends_on" in binding:
        assert restored == [("db",)]
        assert ("restart", "net") in manager.events
    else:
        assert restored == [("db",), ("net",), ("child",)]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db"]
    assert manager.running_state.get_persisted() == ["child", "db", "net", "other"]


@pytest.mark.parametrize("existing_web", [False, True])
@pytest.mark.parametrize("health_dependency", [False, True])
def test_provider_rollback_rebinds_successful_new_binding_without_reverting_it(tmp_path, existing_web, health_dependency):
    def mounts(path):
        return [{"type": "bind", "source": str(path), "target": "/config"}]

    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new", "volumes": mounts(tmp_path / "new")}}),
        ("web", {"web": {"image": "web:new", "network_mode": "service:db",
                         "environment": {"VERSION": "new"}}}),
        ("fail", {"fail": {"image": "fail:new", "volumes": mounts(tmp_path / "new")}}),
    ], running=("db", "fail", "web") if existing_web else ("db", "fail"))
    old = deepcopy(manager.model)
    for name in ("db", "fail"):
        old["services"][name]["volumes"] = mounts(tmp_path / "old")
    old["services"]["web"].pop("network_mode")
    old["services"]["web"]["environment"] = {"VERSION": "old"}
    old["services"]["web"]["image"] = "web:old"
    if health_dependency:
        old["services"]["fail"]["depends_on"] = {"web": {"condition": "service_healthy"}}
        restore = manager.compose_runner.apply_saved_services

        def restore_after_ready(context, services, files):
            if "fail" in services:
                assert sum(event[:2] == ("apply", "web") for event in manager.events) == 2
            restore(context, services, files)

        manager.compose_runner.apply_saved_services = restore_after_ready
    AppliedServiceModels(manager, old).record(("db", "fail", "web") if existing_web else ("db", "fail"))
    manager.image_preparer.image_id = lambda image: (
        "sha256:new-web" if image == "web:new" else "sha256:old-" + image.split(":")[0])
    manager.compose_runner.fail = "fail"
    with pytest.raises(ContainerError, match="apply failed fail"):
        manager.compose_operations.up()
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("db",), ("fail",)]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db", "web", "fail", "web"]
    applied_web = yaml.safe_load(AppliedServiceModels(manager, manager.model).previous["web"])["services"]["web"]
    assert applied_web["image"] == "web:new"
    assert applied_web["network_mode"] == "service:db"
    assert applied_web["environment"] == {"VERSION": "new"}


def test_provider_rollback_does_not_restore_removed_binding(tmp_path):
    manager = setup_case(tmp_path, [
        ("web", {"web": {"image": "web:new", "environment": {"VERSION": "new"}}}),
        ("db", {"db": {"image": "db:new"}}),
    ], running=("db", "web"))
    old = deepcopy(manager.model)
    old["services"]["web"]["network_mode"] = "service:db"
    old["services"]["web"]["environment"] = {"VERSION": "old"}
    AppliedServiceModels(manager, old).record(("db", "web"))
    manager.compose_runner.fail = "db"
    with pytest.raises(ContainerError, match="apply failed db"):
        manager.compose_operations.up()
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("db",)]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["web", "db"]
    applied_web = yaml.safe_load(AppliedServiceModels(manager, manager.model).previous["web"])["services"]["web"]
    assert "network_mode" not in applied_web
    assert applied_web["environment"] == {"VERSION": "new"}


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("binding", [{"network_mode": "service:db"},
                                      {"depends_on": {"db": {"restart": True}}}])
def test_known_running_orphan_respects_removal_and_saved_model(tmp_path, full, failure, binding):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": dict(binding, image="net:old")}),
    ], running=("db", "net"))
    manager.model["services"].pop("net")
    manager.containers.pop("net")
    manager.integration_snapshot.pop("net")
    manager.load_installed_config_metadata = lambda: [manager.containers["db"]]
    restart = manager.compose_runner.restart_service

    def restart_saved(context, service, model=None):
        assert service in (model or context.compose_model)["services"]
        return restart(context, service)

    manager.compose_runner.restart_service = restart_saved
    if failure:
        manager.compose_runner.fail = "db"
        with pytest.raises(ContainerError, match="apply failed db"):
            manager.compose_operations.up(None if full else ["db"])
    else:
        manager.compose_operations.up(None if full else ["db"])
    restored = [event[1] for event in manager.events if event[0] == "restore"]
    expected = [("db",)] if failure else []
    if not full and "network_mode" in binding:
        expected.append(("net",))
    assert restored == expected
    assert (("restart", "net") in manager.events) == (not full and "depends_on" in binding)


def test_cleanup_stops_new_namespace_dependents_of_discarded_provider(tmp_path):
    shared = [{"type": "bind", "source": str(tmp_path / "shared"), "target": "/config"}]
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new", "volumes": shared}}),
        ("web", {"web": {"image": "web:new", "network_mode": "service:db"}}),
        ("child", {"child": {"image": "child:new", "network_mode": "service:web"}}),
        ("fail", {"fail": {"image": "fail:new", "volumes": shared}}),
    ])
    manager.compose_runner.fail = "fail"
    with pytest.raises(ContainerError, match="apply failed fail"):
        manager.compose_operations.up()
    assert ("stop", ("fail", "child", "web", "db")) in manager.events
    assert manager.running_state.get_persisted() == []
    assert not AppliedServiceModels(manager, manager.model).previous


@pytest.mark.parametrize("old_binding", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("captured_web_owner", [False, True])
def test_legacy_untouched_service_uses_captured_old_binding(tmp_path, old_binding, failure, captured_web_owner):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("web", {"web": dict({} if old_binding else {"network_mode": "service:db"}, image="web:new")}),
    ], running=("db", "web"))
    old = deepcopy(manager.model)
    old["services"]["db"]["image"] = "db:old"
    old["services"]["web"]["image"] = "web:old"
    old["services"]["web"].pop("network_mode", None)
    if old_binding:
        old["services"]["web"]["network_mode"] = "service:db"
    for path in (tmp_path / "compose" / "applied" / "services").iterdir():
        path.unlink()
    for name in (("db", "web") if captured_web_owner else ("db",)):
        (tmp_path / "compose" / (name + ".yml")).write_text(yaml.safe_dump(old))
    manager.runtime = SimpleNamespace(create_docker_process=lambda *args, **kwargs: SimpleNamespace(args=args))
    native = ComposeRunner(manager)

    def resolve(process):
        path = process.args[process.args.index("--file") + 1]
        return yaml.safe_load(Path(path).read_text())

    native._resolved_model = resolve
    manager.compose_runner.saved_service_models = native.saved_service_models
    if failure:
        manager.compose_runner.fail = "db"
        with pytest.raises(ContainerError, match="apply failed db"):
            manager.compose_operations.up(["db"])
    else:
        manager.compose_operations.up(["db"])
    restored = [event[1] for event in manager.events if event[0] == "restore"]
    assert restored == ([("db",)] if failure else []) + ([("web",)] if old_binding else [])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db"]


def test_unknown_untouched_service_does_not_use_pending_dependency(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("web", {"web": {"image": "web:new", "network_mode": "service:db"}}),
    ], running=("db", "web"))
    snapshots = AppliedServiceModels(manager, manager.model)
    Path(snapshots._path("web")).unlink()
    (tmp_path / "compose" / "db.yml").write_text(yaml.safe_dump({"services": {"db": {"image": "db:old"}}}))
    manager.compose_operations.up(["db"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db"]
    assert not any(event[0] in ("restore", "restart") for event in manager.events)


@pytest.mark.parametrize("phase", ["apply", "ready"])
def test_restart_interrupt_restores_pending_services_and_reraises_original(tmp_path, phase):
    manager = setup_case(tmp_path, [
        ("app", {name: {"image": name + ":new"} for name in ("first", "second", "third")}),
    ], running=("first", "second", "third"))
    interrupt = KeyboardInterrupt()
    method = "apply_service" if phase == "apply" else "wait_service_ready"
    original = getattr(manager.compose_runner, method)

    def interrupted(context, service, *args, **kwargs):
        if service == "second" and kwargs.get("model") is None:
            raise interrupt
        return original(context, service, *args, **kwargs)

    setattr(manager.compose_runner, method, interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.compose_operations.restart(["app"])
    assert caught.value is interrupt
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("second",), ("third",)]
    assert manager.running_state.get_persisted() == ["app"]
    snapshots = AppliedServiceModels(manager, manager.model).previous
    for name, version in (("first", "new"), ("second", "old"), ("third", "old")):
        assert yaml.safe_load(snapshots[name])["services"][name]["image"] == name + ":" + version


def test_restart_interrupt_during_partial_stop_restores_only_stopped_services(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {name: {"image": name + ":new"} for name in ("first", "second")}),
    ], running=("first", "second"))
    initial = manager.docker_inspector.get_project_state(None)
    observed = deepcopy(initial)
    observed.services[0].state = "exited"
    interrupt = KeyboardInterrupt()

    def interrupted(context, services):
        manager.docker_inspector.get_project_state = lambda containers: observed
        raise interrupt

    manager.compose_runner.stop = interrupted
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.compose_operations.restart(["app"])
    assert caught.value is interrupt
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("first",)]
    assert not any(event[0] == "apply" for event in manager.events)


def test_deployment_recovery_remains_interruptible(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    first, second = KeyboardInterrupt("deploy"), KeyboardInterrupt("recovery")

    def interrupted(context, service, recreate=False):
        raise first

    def interrupted_recovery(context, services, files):
        raise second

    manager.compose_runner.apply_service = interrupted
    manager.compose_runner.apply_saved_services = interrupted_recovery
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.compose_operations.restart(["app"])
    assert caught.value is second


def test_interrupted_deployment_reports_recovery_failure(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    interrupt = KeyboardInterrupt()

    def interrupted(context, service, recreate=False):
        raise interrupt

    manager.compose_runner.apply_service = interrupted
    manager.compose_runner.restore_fails = True
    with pytest.raises(ContainerError, match="Operation failed: KeyboardInterrupt; recovery failed: restore failed") as caught:
        manager.compose_operations.restart(["app"])
    assert caught.value.__cause__ is interrupt


def stopped_provider_case(tmp_path, binding):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:local"}}),
        ("net", {"net": dict(binding, image="net:local")}),
        ("child", {"child": {"image": "child:local", "network_mode": "service:net"}}),
        ("stopped", {"stopped": {"image": "stopped:local", "network_mode": "service:db"}}),
    ], running=("net", "child"))
    AppliedServiceModels(manager, manager.model).record(("db", "net", "child", "stopped"))
    manager.model["services"]["db"]["environment"] = {"VERSION": "new"}
    actual = manager.docker_inspector.get_project_state(None)
    actual = replace(actual, services=actual.services + (SimpleNamespace(service="db", state="exited", image_id="sha256:stopped-db",
                                       labels={}, health=None, exit_code=0),))
    manager.docker_inspector.get_project_state = lambda containers: actual
    return manager


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("binding", [
    {"network_mode": "service:db"}, {"ipc": "service:db"},
    {"pid": "service:db"}, {"volumes_from": ["db:ro"]},
])
def test_stopped_provider_replacement_rebinds_live_namespace_consumers(tmp_path, full, binding):
    manager = stopped_provider_case(tmp_path, binding)
    manager.compose_operations.up(None if full else ["db"])
    if full:
        assert ("apply", "net", True) in manager.events
        assert ("apply", "child", True) in manager.events
    else:
        assert [event[1] for event in manager.events if event[0] == "restore"] == [("net",), ("child",)]
        assert not any(event[0] == "apply" and event[1] == "stopped" for event in manager.events)


@pytest.mark.parametrize("provider_running", [False, True])
@pytest.mark.parametrize("failed_running", [False, True])
def test_later_failure_rebinds_dependents_of_retained_successful_provider(tmp_path, provider_running, failed_running):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
        ("child", {"child": {"image": "child:new", "network_mode": "service:net"}}),
        ("fail", {"fail": {"image": "fail:new"}}),
    ], running=("net", "child") + (("db",) if provider_running else ()) + (("fail",) if failed_running else ()))
    manager.compose_runner.fail = "fail"
    with pytest.raises(ContainerError, match="apply failed fail"):
        manager.compose_operations.up(["db", "fail"])
    restored = [event[1] for event in manager.events if event[0] == "restore"]
    assert ("net",) in restored and ("child",) in restored
    assert ("db",) not in restored
    snapshots = AppliedServiceModels(manager, manager.model).previous
    assert yaml.safe_load(snapshots["db"])["services"]["db"]["image"] == "db:new"


@pytest.mark.parametrize("binding", [
    {"network_mode": "service:db"}, {"ipc": "service:db"},
    {"pid": "service:db"}, {"volumes_from": ["db:ro"]},
])
def test_failed_stopped_provider_restores_live_bindings_then_stops_old_provider(tmp_path, binding):
    manager = stopped_provider_case(tmp_path, binding)
    manager.compose_runner.fail = "db"
    restore = manager.compose_runner.apply_saved_services
    stop = manager.compose_runner.stop
    active = {"net", "child"}
    original_contexts = []

    def restore_with_dependencies(context, services, files, *, image_ids=None):
        service, = services
        if service == "db":
            assert image_ids[service] == "sha256:stopped-db"
            assert service not in context.initial_runtime_state.running_images
            assert yaml.safe_load(next(iter(files.values())))["services"][service].get("environment") is None
            original_contexts.append(context)
        if service == "net":
            assert "db" in active
        if service == "child":
            assert "net" in active
        restore(context, services, files, image_ids=image_ids)
        active.update(services)

    def stop_after_rebinding(context, services):
        if "db" in services:
            assert ("restore", ("child",)) in manager.events
        stop(context, services)
        active.difference_update(services)

    manager.compose_runner.apply_saved_services = restore_with_dependencies
    manager.compose_runner.stop = stop_after_rebinding
    with pytest.raises(ContainerError, match="apply failed db"):
        manager.compose_operations.up(["db"])
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("db",), ("net",), ("child",)]
    assert active == {"net", "child"}
    assert manager.running_state.get_persisted() == ["child", "net"]
    assert original_contexts and "db" not in original_contexts[0].initial_runtime_state.running_images


@pytest.mark.parametrize("missing", ["image", "model"])
def test_stopped_namespace_provider_requires_recovery_input_before_mutation(tmp_path, missing):
    manager = stopped_provider_case(tmp_path, {"network_mode": "service:db"})
    if missing == "image":
        manager.docker_inspector.get_project_state(None).services[-1].image_id = None
    else:
        snapshots = AppliedServiceModels(manager, manager.model)
        Path(snapshots._path("db")).unlink()
    with pytest.raises(ContainerError, match="original image ID|Missing restore input"):
        manager.compose_operations.up(["db"])
    assert not any(event[0] in ("apply", "stop", "restore") for event in manager.events)


def test_failed_temporary_provider_recovery_stops_it_again(tmp_path):
    manager = stopped_provider_case(tmp_path, {"network_mode": "service:db"})
    manager.compose_runner.fail = "db"
    restore = manager.compose_runner.apply_saved_services

    def fail_dependent_restore(context, services, files, *, image_ids=None):
        if "net" in services:
            raise ContainerError("dependent restore failed")
        restore(context, services, files, image_ids=image_ids)

    manager.compose_runner.apply_saved_services = fail_dependent_restore
    with pytest.raises(ContainerError, match="recovery failed: dependent restore failed"):
        manager.compose_operations.up(["db"])
    assert ("stop", ("db",)) in manager.events
    assert manager.running_state.get_persisted() == ["child", "net"]


def test_explicit_recovery_image_ids_do_not_change_initial_running_images(tmp_path):
    commands = []
    files = []

    def process(*args, **kwargs):
        commands.append(args)
        files.append([Path(args[index + 1]).read_text() for index, value in enumerate(args[:-1]) if value == "--file"])
        return SimpleNamespace(check_call=lambda: 0)

    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test",
                                            runtime=SimpleNamespace(create_docker_process=process)))
    runner._resolved_model = lambda process: {"services": {"db": {"image": "sha256:old-stopped"}}}
    runner.wait_service_dependencies = lambda *args, **kwargs: None
    context = SimpleNamespace(initial_runtime_state=ProjectRuntimeState("test", (), "docker"), project_containers=())
    runner.apply_saved_services(context, ("db",), {"old.yml": "services: {db: {image: mutable:tag}}"},
                                image_ids={"db": "sha256:old-stopped"})
    assert yaml.safe_load(files[-1][-1]) == {"services": {"db": {"image": "sha256:old-stopped"}}}
    assert context.initial_runtime_state.running_images == {}


@pytest.mark.parametrize("legacy", [False, True])
def test_recovery_configuration_explicitly_selects_profiled_service(tmp_path, legacy):
    commands = []
    model = {"services": {"optional": {"image": "optional:old", "profiles": ["debug"]}}}

    def process(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(args=args, check_call=lambda: 0)

    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test",
                                          runtime=SimpleNamespace(create_docker_process=process)))
    runner._resolved_model = lambda process: model if process.args[-1] == "optional" else {"services": {}}
    runner.wait_service_dependencies = lambda *args, **kwargs: None
    context = SimpleNamespace(initial_runtime_state=ProjectRuntimeState("test", (
        SimpleNamespace(service="optional", state="running", image_id="sha256:original"),), "docker"), project_containers=(),
                              service_models=SimpleNamespace(previous={}),
                              previous_compose_contents={"old.yml": yaml.safe_dump(model)})
    if legacy:
        assert "optional" in runner.saved_service_models(context, ("optional",))
    else:
        runner.apply_saved_services(context, ("optional",), context.previous_compose_contents)
    config = next(command for command in commands if "config" in command)
    assert config[-4:] == ("config", "--format", "json", "optional")


@pytest.mark.parametrize("binding", [None, {"network_mode": "service:db"},
                                     {"depends_on": {"db": {"restart": True}}}])
def test_legacy_capture_only_resolves_live_profiled_consumers_of_selected_provider(tmp_path, binding):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("optional", {"optional": {"image": "optional:new", "profiles": ["debug"]}}),
    ], running=("db", "optional"))
    old = deepcopy(manager.model)
    old["services"]["db"]["image"] = "db:old"
    old["services"]["optional"].update(binding or {"env_file": ["/missing.env"]})
    optional_snapshot = AppliedServiceModels(manager, old)._path("optional")
    Path(optional_snapshot).unlink()
    (tmp_path / "compose" / "optional.yml").write_text(yaml.safe_dump(old))
    manager.compose_runner.final_model = lambda context, preserve_disabled=False, privilege=None: deepcopy(
        manager.model if preserve_disabled else {"services": {"db": manager.model["services"]["db"]}})
    capture = manager.compose_runner.saved_service_models

    def capture_needed(context, services):
        if "optional" in services:
            assert binding is not None, "unrelated disabled env_file must not be loaded"
            context.service_models.retain_previous({"optional": yaml.safe_dump(old)})
        return capture(context, services)

    manager.compose_runner.saved_service_models = capture_needed
    manager.compose_operations.up()
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db"]
    if binding and "network_mode" in binding:
        assert ("restore", ("optional",)) in manager.events
    elif binding:
        assert ("restart", "optional") in manager.events
    else:
        assert not any(event[0] in ("restore", "restart") for event in manager.events)


def legacy_labeled_case(tmp_path, binding, label, child_restart=False):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("worker", {"worker": {"image": "worker:new"}}),
        ("child", {"child": {"image": "child:new"}}),
    ], running=("db", "worker", "child"))
    old = deepcopy(manager.model)
    old["services"]["worker"].update(binding)
    old["services"]["child"].update(network_mode="service:worker", depends_on={
        "worker": {"condition": "service_started", "restart": child_restart}})
    store = AppliedServiceModels(manager, old)
    for name in ("worker", "child"):
        Path(store._path(name)).unlink()
        (tmp_path / "compose" / (name + ".yml")).write_text(yaml.safe_dump(old))
    actual = manager.docker_inspector.get_project_state(None)
    actual.services[1].labels["com.docker.compose.depends_on"] = label
    actual.services[2].labels["com.docker.compose.depends_on"] = "worker:service_started:" + str(child_restart).lower()
    actual.services[1].namespace_bindings = {"network_mode": "service:db"} if "network_mode" in binding else {}
    actual.services[2].namespace_bindings = {"network_mode": "service:worker"}
    manager.runtime = SimpleNamespace(create_docker_process=lambda *args, **kwargs: SimpleNamespace(args=args))
    native = ComposeRunner(manager)

    def resolve(process):
        path = process.args[process.args.index("--file") + 1]
        resolved = yaml.safe_load(Path(path).read_text())
        if "network_mode" in binding:
            assert resolved["services"]["worker"]["network_mode"] == "service:db"
        return resolved

    native._resolved_model = resolve

    def capture(context, services):
        manager.events.append(("capture-restore", tuple(services)))
        return native.saved_service_models(context, services)

    manager.compose_runner.saved_service_models = capture
    return manager


@pytest.mark.parametrize("label", ["db", "db:service_started", "db:service_started:true", "db:service_started:false"])
def test_native_labels_capture_interpolated_namespace_and_then_its_child(tmp_path, label):
    manager = legacy_labeled_case(tmp_path, {"network_mode": "${REVIEW_NETWORK_MODE}"}, label)
    manager.compose_operations.up(["db"])
    captures = [event[1] for event in manager.events if event[0] == "capture-restore"]
    assert {"worker", "child"}.issubset(set().union(*captures))
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("worker",), ("child",)]


@pytest.mark.parametrize("label", ["db:service_started:false", ""])
def test_native_false_or_empty_dependency_label_does_not_capture_unrelated_legacy_consumer(tmp_path, label):
    manager = legacy_labeled_case(tmp_path, {"depends_on": {"db": {"restart": False}},
                                           "env_file": ["/missing.env"]}, label)
    manager.compose_operations.up(["db"])
    assert not any(event[0] in ("restore", "restart") for event in manager.events)
    assert all(event[1] == ("db",) for event in manager.events if event[0] == "capture-restore")


def test_native_implicit_restart_edge_restarts_child_without_recreating_namespace(tmp_path):
    manager = legacy_labeled_case(tmp_path, {"depends_on": {"db": {"restart": True}}},
                                 "db:service_started:true", child_restart=True)
    manager.compose_operations.up(["db"])
    assert [event[1] for event in manager.events if event[0] == "restart"] == ["worker", "child"]
    assert not any(event[0] == "restore" for event in manager.events)


def test_nonshared_runtime_namespace_and_false_restart_do_not_capture_dynamic_legacy_consumer(tmp_path):
    manager = legacy_labeled_case(tmp_path, {"network_mode": "${REVIEW_NETWORK_MODE}",
                                           "env_file": ["/missing.env"]}, "db:service_started:false")
    manager.docker_inspector.get_project_state(None).services[1].namespace_bindings = {"network_mode": "host"}
    manager.compose_operations.up(["db"])
    assert all(event[1] == ("db",) for event in manager.events if event[0] == "capture-restore")


@pytest.mark.parametrize("bindings", [
    {}, {"network_mode": "host"}, {"network_mode": "bridge"},
    {"network_mode": "service:db"}, {"network_mode": "container:external-db"},
    {"volumes_from": ["db:ro", "container:external-db:rw"]},
])
def test_legacy_namespace_pinning_precedes_native_interpolation(tmp_path, bindings):
    old = {"services": {"worker": {"image": "worker:old", "network_mode": "${CHANGED_MODE}",
                                    "volumes_from": ["${CHANGED_VOLUME}"]}, "db": {"image": "db:old"}}}
    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test", runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: SimpleNamespace(args=args))))
    context = SimpleNamespace(service_models=SimpleNamespace(previous={}), project_containers=(),
                              previous_compose_contents={"old.yml": yaml.safe_dump(old)},
                              initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(
                                  service="worker", namespace_bindings=bindings),)))

    def resolve(process):
        path = process.args[process.args.index("--file") + 1]
        model = yaml.safe_load(Path(path).read_text())
        assert model["services"]["worker"] == dict(image="worker:old", **bindings)
        return model

    runner._resolved_model = resolve
    recovered = yaml.safe_load(runner.saved_service_models(context, ("worker",))["worker"])
    assert recovered["services"]["worker"] == dict(image="worker:old", **bindings)
    assert context.previous_compose_contents["old.yml"] == yaml.safe_dump(old)


def test_legacy_namespace_pinning_does_not_add_network_mode_to_networks_service(tmp_path):
    old = {"services": {"worker": {"image": "worker:old", "networks": {"default": {}}}}}
    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test", runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: SimpleNamespace(args=args))))
    context = SimpleNamespace(service_models=SimpleNamespace(previous={}), project_containers=(),
                              previous_compose_contents={"old.yml": yaml.safe_dump(old)},
                              initial_runtime_state=SimpleNamespace(services=(SimpleNamespace(service="worker",
                                  namespace_bindings={"network_mode": "test_default", "ipc": "private",
                                                      "pid": "", "volumes_from": []}),)))
    runner._resolved_model = lambda process: yaml.safe_load(Path(process.args[process.args.index("--file") + 1]).read_text())
    assert yaml.safe_load(runner.saved_service_models(context, ("worker",))["worker"]) == old


def test_second_interrupt_does_not_force_temporary_provider_cleanup(tmp_path):
    manager = stopped_provider_case(tmp_path, {"network_mode": "service:db"})
    manager.compose_runner.fail = "db"
    restore = manager.compose_runner.apply_saved_services
    interrupt = KeyboardInterrupt("cancel recovery")

    def interrupted_restore(context, services, files, *, image_ids=None):
        if "net" in services:
            raise interrupt
        restore(context, services, files, image_ids=image_ids)

    manager.compose_runner.apply_saved_services = interrupted_restore
    with pytest.raises(KeyboardInterrupt) as caught:
        manager.compose_operations.up(["db"])
    assert caught.value is interrupt
    assert ("restore", ("db",)) in manager.events
    assert not any(event[0] == "stop" for event in manager.events)
    assert manager.running_state.get_persisted() == ["child", "db", "net"]


def test_temporary_provider_cleanup_failure_preserves_both_recovery_errors(tmp_path):
    manager = stopped_provider_case(tmp_path, {"network_mode": "service:db"})
    manager.compose_runner.fail = "db"
    restore = manager.compose_runner.apply_saved_services

    def failed_restore(context, services, files, *, image_ids=None):
        if "net" in services:
            raise ContainerError("consumer restore failed")
        restore(context, services, files, image_ids=image_ids)

    def failed_stop(context, services):
        raise ContainerError("provider stop failed")

    manager.compose_runner.apply_saved_services = failed_restore
    manager.compose_runner.stop = failed_stop
    with pytest.raises(ContainerError, match="consumer restore failed; stopping temporary providers failed: provider stop failed"):
        manager.compose_operations.up(["db"])
    assert manager.running_state.get_persisted() == ["child", "db", "net"]


def test_stopped_unrelated_service_does_not_add_a_recovery_requirement(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:new"}}),
        ("net", {"net": {"image": "net:new"}}),
    ], running=("net",))
    actual = manager.docker_inspector.get_project_state(None)
    actual = replace(actual, services=actual.services + (SimpleNamespace(service="db", state="exited", image_id=None,
                                       labels={}, health=None, exit_code=0),))
    manager.docker_inspector.get_project_state = lambda containers: actual
    manager.compose_operations.up(["db"])
    assert ("apply", "db", True) in manager.events


def test_legacy_stopped_namespace_preflight_follows_newly_captured_models(tmp_path):
    manager = setup_case(tmp_path, [
        ("vpn", {"vpn": {"image": "vpn:new"}}),
        ("db", {"db": {"image": "db:new", "network_mode": "service:vpn"}}),
        ("net", {"net": {"image": "net:new", "network_mode": "service:db"}}),
    ], running=("net",))
    actual = manager.docker_inspector.get_project_state(None)
    actual = replace(actual, services=actual.services + tuple(SimpleNamespace(service=name, state="exited", image_id="sha256:old-" + name,
                                             labels={}, health=None, exit_code=0) for name in ("db", "vpn")))
    manager.docker_inspector.get_project_state = lambda containers: actual
    original = manager.compose_runner.saved_service_models
    captured = []

    def saved_models(context, services):
        captured.extend(services)
        known = tuple(service for service in services if service in context.service_models.previous)
        results = original(context, known)
        for service in services:
            if service not in results:
                results[service] = yaml.safe_dump(manager.model)
        return results

    manager.compose_runner.saved_service_models = saved_models
    manager.compose_operations.up(["db"])
    first_apply = next(index for index, event in enumerate(manager.events) if event[0] == "apply")
    assert "db" in captured and "vpn" in captured
    assert ("capture-restore", ("net",)) in manager.events[first_apply:]


def test_recovery_metadata_failure_keeps_observed_restoration_state(tmp_path, monkeypatch):
    manager = setup_case(tmp_path, [
        (name, {name: {"image": name + ":new"}}) for name in ("first", "second")
    ], running=("first", "second"))
    manager.compose_runner.fail = "first"

    def fail_snapshot_restore(self, services):
        raise OSError("snapshot write failed")

    monkeypatch.setattr(AppliedServiceModels, "restore", fail_snapshot_restore)
    with pytest.raises(ContainerError, match="recovery failed: snapshot write failed"):
        manager.compose_operations.restart(["first", "second"])
    assert manager.running_state.get_persisted() == ["first"]


def test_failed_recovery_does_not_retain_running_state_for_cleaned_new_peer(tmp_path):
    shared = [{"type": "bind", "source": str(tmp_path / "new"), "target": "/config"}]
    manager = setup_case(tmp_path, [
        ("new", {"new": {"image": "new:local", "volumes": shared}}),
        ("fail", {"fail": {"image": "fail:local", "volumes": shared}}),
    ], running=("fail",))
    old = deepcopy(manager.model)
    old["services"]["fail"]["volumes"][0]["source"] = str(tmp_path / "old")
    AppliedServiceModels(manager, old).record(("fail",))
    manager.compose_runner.fail = "fail"
    manager.compose_runner.restore_fails = True
    with pytest.raises(ContainerError, match="recovery failed: restore failed"):
        manager.compose_operations.up()
    assert ("stop", ("new",)) in manager.events
    assert "new" not in manager.running_state.get_persisted()
