#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migration inputs and dependent namespaces survive failed deployments."""
from pathlib import Path
from types import SimpleNamespace

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
    assert restored == [("db",), ("net",), ("child",)]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db"]
    assert manager.running_state.get_persisted() == ["child", "db", "net", "other"]
