#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Service-scoped preflight checks and recoverable updates of running services."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from _harness import builtin_container_type
from linktools.cntr import ContainerError, EventContext
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.artifacts import AppliedServiceModels, ArtifactIndex, compose_candidate
from linktools.cntr.runtime.images import ImagePlan


class _Container:
    dependencies = ()
    application_priority = 0
    bootstrap_services = ()
    docker_file = None

    def __init__(self, name: str, services: "tuple[str, ...]") -> None:
        self.name = name
        self.services = {service: {"image": service + ":new"} for service in services}
        self.generation_services = services

    @property
    def docker_compose(self) -> dict:
        return {"services": self.services}

    def get_runtime_requirements(self, required: "set[str]") -> dict:
        return {}

    def on_check(self, context: EventContext) -> None:
        pass

    def on_starting(self, context: EventContext) -> None:
        self.manager.events.append(("prepare", self.name))

    def on_service_started(self, context: EventContext, service: str) -> None:
        pass


def _case(root: Path, containers: "tuple[_Container, ...]", selected: "tuple[str, ...]",
          states: "dict[str, str]") -> SimpleNamespace:
    events = []
    model = {"services": {name: spec for owner in containers for name, spec in owner.services.items()}}
    manager = SimpleNamespace(
        project_name="test", data_path=root, logger=None, events=events,
        containers={owner.name: owner for owner in containers}, generated_configs={},
        environ=SimpleNamespace(locks=SimpleNamespace(process_lock=lambda key: nullcontext())),
        image_preparer=SimpleNamespace(plan=lambda model, services, **kwargs:
            ImagePlan(build=(), pull=(), targets=tuple(services))),
        docker_inspector=SimpleNamespace(get_project_state=lambda owners: SimpleNamespace(
            running_container_names=[owner.name for owner in containers
                                     if any(name in states for name in owner.services)],
            services=tuple(SimpleNamespace(service=name, state=state, health=None, image_id="sha256:" + name)
                           for name, state in states.items()))),
        running_state=SimpleNamespace(
            mark_started=lambda context: events.append(("started", tuple(c.name for c in context.target_containers))),
            mark_stopped=lambda context: events.append(("stopped", tuple(c.name for c in context.target_containers)))),
    )
    manager.artifact_index = ArtifactIndex(manager)
    for owner in containers:
        owner.manager = manager

    @contextmanager
    def notify_start(context):
        events.append(("preflight", context.target_services))
        for owner in context.target_containers:
            owner.on_check(context)
        for owner in context.target_containers:
            owner.on_starting(context)
        yield
        events.append(("after", context.target_services))

    manager.lifecycle = SimpleNamespace(
        notify_start=notify_start, notify_stop=lambda context: nullcontext(),
        notify_remove=lambda context: nullcontext())
    manager.compose_runner = SimpleNamespace(
        final_model=lambda context: deepcopy(model),
        apply_services=lambda context, services: events.append(("apply", tuple(services))),
        stop=lambda context, services: events.append(("stop", tuple(services))),
        apply_saved_services=lambda context, services, files: events.append(
            ("restore", tuple(services), files)),
        saved_service_models=lambda context, services: {
            service: context.service_models.previous.get(service) or
            next(iter(context.saved_compose.values())) for service in services},
        wait_service_running=lambda context, service: None,
        wait_service_healthy=lambda context, service: None,
    )
    operations = ComposeOperations(manager)
    targets = tuple(owner for owner in containers if owner.name in selected)
    services = tuple(name for owner in targets for name in owner.services)
    operations.select = lambda *args, **kwargs: ComposeSelection(containers, targets, services, False)
    manager.operations = operations
    manager.model = model
    return manager


@pytest.mark.parametrize("services", [None, (), ("authelia-redis",), ("authelia",), ("authelia-admin",)])
def test_authelia_https_check_only_applies_to_native_consumers(services: "tuple[str, ...] | None") -> None:
    native = builtin_container_type("102-authelia")
    reads = []
    owner = SimpleNamespace(generation_services=native.generation_services,
                            get_config=lambda key: reads.append(key) or False)
    context = EventContext(target_services=services)
    required = services is None or any(name in services for name in native.generation_services)
    if required:
        with pytest.raises(ContainerError, match="Authelia requires HTTPS"):
            native.on_check(owner, context)
        assert reads == ["NGINX_HTTPS_ENABLE"]
    else:
        native.on_check(owner, context)
        assert reads == []


@pytest.mark.parametrize("changed", [False, True])
def test_redis_reconciliation_preserves_preparation_without_https_requirement(tmp_path: Path, changed: bool) -> None:
    target = _Container("lldap", ("lldap",))
    owner = _Container("authelia", ("authelia", "authelia-admin", "authelia-redis"))
    native = builtin_container_type("102-authelia")
    owner.generation_services = native.generation_services
    reads = []
    owner.get_config = lambda key: reads.append(key) or False
    owner.on_check = lambda context: native.on_check(owner, context)
    case = _case(tmp_path, (target, owner), ("lldap",), {"authelia-redis": "running"})
    case.generated_configs = {"authelia": owner}
    AppliedServiceModels(case, case.model).record(("authelia-redis",))
    if changed:
        owner.services["authelia-redis"]["environment"] = {"VALUE": "new"}
    case.operations.up(["lldap"])
    assert ("prepare", "authelia") in case.events
    assert ("apply", ("lldap",)) in case.events
    assert (("apply", ("authelia-redis",)) in case.events) is changed
    assert not any(event[0] == "apply" and "authelia" in event[1] for event in case.events)
    before = next(event[1] for event in case.events if event[0] == "preflight")
    after = next(event[1] for event in case.events if event[0] == "after")
    assert set(before) == {"lldap", "authelia-redis"}
    assert set(after) == ({"lldap", "authelia-redis"} if changed else {"lldap"})
    assert reads == []


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("state", ["running", "restarting"])
def test_plain_running_service_without_old_model_is_not_replaced(tmp_path: Path, action: str, state: str) -> None:
    owner = _Container("app", ("app",))
    case = _case(tmp_path, (owner,), ("app",), {"app": state})
    with pytest.raises(ContainerError, match="Cannot replace running service app without a previous Compose model"):
        getattr(case.operations, action)(["app"])
    assert not any(event[0] in ("stop", "apply", "stopped", "started", "restore") for event in case.events)


@pytest.mark.parametrize("source", ["snapshot", "legacy"])
@pytest.mark.parametrize("action", ["up", "restart"])
def test_plain_running_service_with_old_model_remains_recoverable(tmp_path: Path, source: str, action: str) -> None:
    owner = _Container("app", ("app",))
    case = _case(tmp_path, (owner,), ("app",), {"app": "running"})
    old = {"services": {"app": {"image": "app:old"}}}
    if source == "snapshot":
        AppliedServiceModels(case, old).record(("app",))
    else:
        path, _ = compose_candidate(owner)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(old))

    def fail(context, services):
        case.events.append(("apply", tuple(services)))
        raise RuntimeError("new configuration failed")

    case.compose_runner.apply_services = fail
    with pytest.raises(RuntimeError, match="new configuration failed"):
        getattr(case.operations, action)(["app"])
    restored = [event for event in case.events if event[0] == "restore"]
    assert len(restored) == 1
    assert restored[0][1] == ("app",)
    assert yaml.safe_load(next(iter(restored[0][2].values())))["services"]["app"]["image"] == "app:old"


def test_stopped_unrelated_legacy_file_is_not_a_rollback_requirement(tmp_path: Path) -> None:
    app = _Container("app", ("app",))
    other = _Container("other", ("other",))
    case = _case(tmp_path, (app, other), ("app",), {})
    path, _ = compose_candidate(other)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("services: [invalid")
    case.operations.up(["app"])
    assert ("apply", ("app",)) in case.events
    assert not any(event[0] == "apply" and "other" in event[1] for event in case.events)


@pytest.mark.parametrize("full", [False, True])
def test_context_exposes_exact_service_selection(full: bool) -> None:
    owner = _Container("app", ("app", "sidecar"))
    selection = ComposeSelection((owner,), (owner,), () if full else ("sidecar",), full)
    context = ComposeOperations(SimpleNamespace())._make_context("up", selection)
    assert context.target_services == (("app", "sidecar") if full else ("sidecar",))


def test_implicit_native_provider_is_prepared_before_its_first_application(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lldap = _Container("lldap", ("lldap",))
    nginx = _Container("nginx", ("nginx",))
    authelia = _Container("authelia", ("authelia", "authelia-redis"))
    nginx.generation_services = ("nginx",)
    authelia.generation_services = ("authelia",)
    nginx.get_runtime_requirements = lambda roots: (
        {"authelia": ("authelia",)} if "nginx" in roots else {})
    authelia.dependencies = ("lldap",)
    case = _case(tmp_path, (nginx, lldap, authelia), ("lldap",), {"nginx": "running"})
    case.generated_configs = {"nginx": nginx, "authelia": authelia}
    AppliedServiceModels(case, case.model).record(("nginx",))
    events = case.events
    original_model = case.compose_runner.final_model
    case.compose_runner.final_model = lambda context: events.append(("final-model",)) or original_model(context)
    authelia.on_check = lambda context: events.append(("check", "authelia"))
    authelia.on_starting = lambda context: events.append(("starting", "authelia"))

    for owner in (nginx, authelia):
        owner.on_prepare_config = lambda ctx, name=owner.name: events.append(("prepared", name))
        owner.render_config = lambda generation_id: {"config": generation_id}
        owner.validate_config = lambda ctx, candidate: None
        owner.apply_config = lambda ctx, candidate, services, name=owner.name: events.append(
            ("native-applied", name, tuple(services)))

    def candidate(owner, render):
        return SimpleNamespace(container=owner, changed=True, previous_id=None,
                               generation_id="next-" + owner.name, changed_files=(),
                               publish=lambda: None, restore=lambda: None,
                               prune=lambda: events.append(("pruned", owner.name)))

    monkeypatch.setattr("linktools.cntr.artifacts.GeneratedCandidate", candidate)
    case.operations.up(["lldap"])
    assert ("prepared", "authelia") in events
    assert ("native-applied", "authelia", ("authelia",)) in events
    assert events.index(("check", "authelia")) < events.index(("starting", "authelia"))
    assert events.index(("starting", "authelia")) < events.index(("final-model",))
    assert events.index(("final-model",)) < events.index(("prepared", "authelia"))
    assert events.index(("prepared", "authelia")) < events.index(
        ("native-applied", "authelia", ("authelia",)))
    assert events.index(("native-applied", "authelia", ("authelia",))) < events.index(
        ("pruned", "authelia"))
    assert not any(event == ("native-applied", "authelia", ("authelia-redis",))
                   for event in events)
