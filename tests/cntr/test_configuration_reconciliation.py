#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Partial operations compare all candidates but apply only eligible services."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import BaseContainer
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
from linktools.cntr.runtime.images import ImagePlan


class Container(BaseContainer):
    name = ""
    dependencies = ()
    integrations = ()

    def __init__(self, name, services):
        self.name = name
        self.services = {service: {} for service in services}


class SidecarContainer(Container):
    generation_services = ("stopped",)


def reconciliation(tmp_path, monkeypatch, changed=("running", "stopped"), sidecar=False):
    target = Container("target", ("target",))
    other_type = SidecarContainer if sidecar else Container
    other = other_type("other", ("running", "stopped"))
    containers = (target, other)
    calls = []
    paths = {}
    candidates = {}
    old_model = {"services": {}}
    new_model = {"services": {}}
    for container in containers:
        path = tmp_path / (container.name + ".yml")
        old = {"services": {name: {"image": "image:one", "environment": {"VALUE": "old"}}
                            for name in container.services}}
        new = {"services": {name: {"image": "image:one", "environment": {
            "VALUE": "new" if name in changed else "old"}} for name in container.services}}
        old_model["services"].update(old["services"])
        new_model["services"].update(new["services"])
        path.write_text(yaml.safe_dump(old))
        candidates[str(path)] = ("compose", container.name, yaml.safe_dump(new))
        paths[container.name] = path
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: candidates)
    runner = SimpleNamespace(
        final_model=lambda context: dict(new_model, services={
            name: dict(spec, **manager.containers["target" if name == "target" else "other"].services[name])
            for name, spec in new_model["services"].items()}),
        apply_services=lambda context, services: calls.append(("apply", tuple(services))),
        apply_saved_services=lambda context, services, files: calls.append(("restore", tuple(services), files)),
        saved_service_models=lambda context, services: {
            service: context.service_models.previous[service] for service in services},
        wait_service_running=lambda context, service: None,
        wait_service_healthy=lambda context, service: None,
    )
    manager = SimpleNamespace(
        project_name="test", data_path=tmp_path, logger=None,
        containers={c.name: c for c in containers}, integration_snapshot={c.name: () for c in containers},
        generated_configs={}, compose_runner=runner,
        environ=SimpleNamespace(locks=SimpleNamespace(process_lock=lambda key: nullcontext())),
        lifecycle=SimpleNamespace(notify_start=lambda context: nullcontext(), notify_remove=lambda context: nullcontext()),
        image_preparer=SimpleNamespace(plan=lambda model, services, **kwargs:
            ImagePlan(build=(), pull=(), targets=tuple(services))),
        artifact_index=SimpleNamespace(record=lambda entries, remove=(): None),
        running_state=SimpleNamespace(mark_started=lambda context: None),
        resolver=SimpleNamespace(resolve_dependencies=lambda selected: [c for c in containers if c in selected]),
        docker_inspector=SimpleNamespace(get_project_state=lambda containers: ProjectRuntimeState("test", (
            ServiceRuntimeState(("other",), "running", "other-runtime", "running", None, "image:one", None, {},
                                image_id="sha256:running"),
        ), "docker")),
    )
    from linktools.cntr.artifacts import AppliedServiceModels
    AppliedServiceModels(manager, old_model).record(old_model["services"])
    operations = ComposeOperations(manager)
    monkeypatch.setattr(operations, "select", lambda *args, **kwargs:
                        ComposeSelection(containers, (target,), ("target",), False))
    return operations, manager, calls, paths


def test_partial_up_does_not_prepare_unrelated_images(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    planned = []

    def image_plan(model, services, force_pull=False):
        planned.extend(services)
        assert "stopped" not in services
        return ImagePlan(build=(), pull=(), targets=tuple(services))

    manager.image_preparer.plan = image_plan
    operations.up(["target"])
    assert planned == ["target"]


def test_running_generated_owner_sync_does_not_build_stopped_sibling(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    owner = manager.containers["other"]
    manager.generated_configs = {"other": owner}
    validated = []
    owner.on_prepare_config = lambda context: validated.append("prepare")
    owner.validate_config = lambda context, candidate: validated.append("validate")
    owner.apply_config = lambda context, candidate, services: calls.append(("apply", tuple(services)))
    monkeypatch.setattr("linktools.cntr.artifacts.GeneratedCandidate",
                        lambda container, render: SimpleNamespace(
                            container=container, generation_id="saved", previous_id="saved",
                            changed=False, publish=lambda: None, prune=lambda: None))
    planned = []

    def image_plan(model, services, force_pull=False):
        planned.extend(services)
        return ImagePlan(build=(), pull=(), targets=tuple(services))

    manager.image_preparer.plan = image_plan
    operations.up(["target"])
    assert planned == ["target"]
    assert validated == ["prepare", "validate"]
    assert ("apply", ("running",)) not in calls
    assert not any("stopped" in call[1] for call in calls)


def test_changed_running_generation_prepares_its_image_only_after_change(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    owner = manager.containers["other"]
    manager.generated_configs = {"other": owner}
    owner.on_prepare_config = lambda context: None
    owner.validate_config = lambda context, candidate: None
    owner.apply_config = lambda context, candidate, services: calls.append(("apply", tuple(services)))
    monkeypatch.setattr("linktools.cntr.artifacts.GeneratedCandidate",
                        lambda container, render: SimpleNamespace(
                            container=container, generation_id="next", previous_id="previous",
                            changed=True, publish=lambda: None, prune=lambda: None))
    planned = []

    def plan(model, services, force_pull=False):
        planned.append(tuple(services))
        return ImagePlan(build=(), pull=(), targets=tuple(services))

    manager.image_preparer.plan = plan
    operations.up(["target"])
    assert planned == [("target",), ("running",)]
    assert ("apply", ("running",)) in calls
    assert not any("stopped" in call[1] for call in calls)


def test_unrelated_running_sidecar_does_not_prepare_owner_config(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=(), sidecar=True)
    owner = manager.containers["other"]
    manager.generated_configs = {"other": owner}
    owner.on_prepare_config = lambda context: pytest.fail("sidecar must not prepare native config")
    owner.render_config = lambda version: pytest.fail("sidecar must not render native config")
    planned = []
    manager.image_preparer.plan = lambda model, services, **kwargs: (
        planned.append(tuple(services)) or ImagePlan(build=(), pull=(), targets=tuple(services)))
    operations.up(["target"])
    assert planned == [("target",)]
    assert not any("running" in call[1] for call in calls)


def test_changed_running_sidecar_updates_without_native_generation(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=("running",), sidecar=True)
    owner = manager.containers["other"]
    manager.generated_configs = {"other": owner}
    owner.on_prepare_config = lambda context: pytest.fail("sidecar update must not prepare generated config")
    owner.render_config = lambda version: pytest.fail("sidecar update must not render generated config")
    operations.up(["target"])
    assert ("apply", ("target",)) in calls
    assert ("apply", ("running",)) in calls
    assert not any("stopped" in call[1] for call in calls)


@pytest.mark.parametrize("changed", [(), ("stopped",), ("running",), ("running", "stopped")])
def test_partial_up_applies_pending_running_config_without_starting_stopped_sibling(tmp_path, monkeypatch, changed):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed)
    operations.up(["target"])
    assert ("apply", ("target",)) in calls
    assert (("apply", ("running",)) in calls) is ("running" in changed)
    assert not any("stopped" in call[1] for call in calls)
    applied = Path(manager.data_path) / "compose/applied/other.yml"
    if "running" in changed:
        snapshot = yaml.safe_load(applied.read_text())["services"]
        assert snapshot["running"]["environment"]["VALUE"] == "new"
        assert snapshot["stopped"]["environment"]["VALUE"] == "old"
    else:
        assert not applied.exists()


def test_failed_pending_update_restores_only_previously_running_services(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch)
    previous = paths["other"].read_text()

    def apply(context, services):
        calls.append(("apply", tuple(services)))
        if "running" in services:
            paths["other"].write_text(context.compose_files[str(paths["other"])])
            raise RuntimeError("pending configuration failed")

    manager.compose_runner.apply_services = apply
    with pytest.raises(RuntimeError, match="pending configuration failed"):
        operations.up(["target"])
    restored = [call for call in calls if call[0] == "restore"]
    assert len(restored) == 1
    assert restored[0][1] == ("running",)
    assert paths["other"].read_text() == previous
    restored_model = yaml.safe_load(next(iter(restored[0][2].values())))
    assert restored_model["services"]["running"]["environment"]["VALUE"] == "old"
    assert not (Path(manager.data_path) / "compose/applied/other.yml").exists()


def test_unchanged_generated_owner_does_not_apply_while_changed_owner_does(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    explicit = operations.select(["target"])
    context = SimpleNamespace(initial_running_services={"running"}, changed_compose_services=set())
    assert operations._reconcile_selection(explicit, context).services == ("target",)
    assert operations._reconcile_selection(explicit, context, {"other"}).services == ("target", "running")


def test_pending_service_starts_only_its_actual_stopped_dependency(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    manager.containers["other"].services["running"]["depends_on"] = {
        "stopped": {"condition": "service_healthy"}}
    operations.up(["target"])
    assert ("apply", ("stopped",)) in calls
    assert calls.index(("apply", ("stopped",))) < calls.index(("apply", ("running",)))


def test_shared_effective_network_change_reconciles_running_service(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    original = manager.compose_runner.final_model
    manager.compose_runner.final_model = lambda context: dict(original(context), networks={"shared": {"name": "new-network"}})
    operations.up(["target"])
    assert ("apply", ("running",)) in calls
    assert not any("stopped" in call[1] for call in calls)


def test_resolved_environment_change_reconciles_and_rolls_back_old_value(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    original = manager.compose_runner.final_model

    def model(context):
        result = original(context)
        result["services"]["running"]["environment"] = {"VALUE": "resolved-env-file-new"}
        return result

    def apply(context, services):
        calls.append(("apply", tuple(services)))
        if "running" in services:
            raise RuntimeError("bad env file")

    manager.compose_runner.final_model = model
    manager.compose_runner.apply_services = apply
    with pytest.raises(RuntimeError, match="bad env file"):
        operations.up(["target"])
    restored = next(call for call in calls if call[0] == "restore")
    restored_model = yaml.safe_load(next(iter(restored[2].values())))
    assert restored_model["services"]["running"]["environment"]["VALUE"] == "old"


def test_startup_hook_environment_is_captured_after_preparation(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from linktools.cntr.artifacts import AppliedServiceModels

    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    original = manager.compose_runner.final_model
    prepared = []

    @contextmanager
    def start(context):
        prepared.append(True)
        yield

    def model(context):
        assert prepared, "env_file must be prepared before resolving Compose"
        result = original(context)
        result["services"]["target"]["environment"] = {"VALUE": "hook-new"}
        return result

    manager.lifecycle.notify_start = start
    manager.compose_runner.final_model = model
    operations.up(["target"])
    stored = AppliedServiceModels(manager, model(None)).previous["target"]
    assert yaml.safe_load(stored)["services"]["target"]["environment"]["VALUE"] == "hook-new"


def test_service_dag_can_interleave_container_owners(tmp_path, monkeypatch):
    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    target = manager.containers["target"]
    other = manager.containers["other"]
    target.services["target"]["depends_on"] = {"running": {}}
    target.services["database"] = {}
    other.services["running"]["depends_on"] = {"database": {}}
    selected = operations.start_selection(operations.select(["target"]))
    assert selected.services.index("database") < selected.services.index("running") < selected.services.index("target")
    original = manager.compose_runner.final_model
    def model(context):
        result = original(context)
        result["services"]["database"] = {"image": "database:test"}
        return result
    manager.compose_runner.final_model = model
    operations.up(["target"])
    assert calls.index(("apply", ("database",))) < calls.index(("apply", ("running",))) < calls.index(("apply", ("target",)))


def test_service_edge_preserves_its_owners_strong_group_dependencies():
    first = Container("b", ("b",))
    second = Container("a", ("a",))
    dependency = Container("c", ("c",))
    first.services["b"]["depends_on"] = {"a": {}}
    second.dependencies = ("c",)
    project = (first, second, dependency)
    manager = SimpleNamespace(integration_snapshot={c.name: () for c in project},
        resolver=SimpleNamespace(resolve_dependencies=lambda selected: (first, dependency, second)))
    selected = ComposeOperations(manager).start_selection(ComposeSelection(project, (first,), ("b",), False))
    assert selected.services == ("c", "a", "b")


def test_running_owner_prepares_inputs_but_after_start_only_visits_applied_targets(tmp_path, monkeypatch):
    from linktools.cntr.lifecycle import HookRegistry, LifecycleDispatcher

    operations, manager, calls, paths = reconciliation(tmp_path, monkeypatch, changed=())
    events = []
    manager.environ.debug = False
    manager.hooks = HookRegistry(owner=manager, scope="manager")
    manager.lifecycle = LifecycleDispatcher(manager)
    monkeypatch.setattr(manager.lifecycle, "notify_remove", lambda context: nullcontext())
    for container in manager.containers.values():
        container.manager = manager
        container.on_check = lambda context, name=container.name: events.append(("check", name))
        container.on_starting = lambda context, name=container.name: events.append(("prepare", name))
        container.on_started = lambda context, name=container.name: events.append(("started", name))
    operations.up(["target"])
    assert ("check", "other") in events
    assert ("prepare", "other") in events
    assert ("started", "other") not in events
    assert ("started", "target") in events
    assert calls == [("apply", ("target",))]
