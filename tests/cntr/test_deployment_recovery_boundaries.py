#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migration inputs and dependent namespaces survive failed deployments."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import pytest
import yaml

from linktools.cntr._container.compose import write_docker_compose_file
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner
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
        old = yaml.safe_load(context.saved_compose[str(compose_file)])
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
    actual.services += (SimpleNamespace(service="retired", state="running", image_id="sha256:retired",
                                       labels={}, health=None, exit_code=None),)
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
