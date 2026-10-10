#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks at mocked lifecycle and raw-Docker boundaries."""
import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from linktools.cntr import BaseContainer
from _harness import builtin_container_type
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.artifacts import AppliedServiceModels, GeneratedCandidate
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner, service_dependencies
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
from linktools.cntr.runtime.images import ImagePlan
from linktools.cntr.state.running import RunningStateStore

if TYPE_CHECKING:
    from typing import AbstractSet, Iterable, Mapping
    from linktools.cntr import ContainerManager, OperationContext


BuiltinNginxContainer = builtin_container_type("100-nginx")


class Container(BaseContainer):
    dependencies = ()
    docker_file = None
    sites = {}

    def __init__(self, name, services, path):
        self._name, self.services, self.path = name, services, path
        self.docker_compose = {"services": services}
        self.logger = SimpleNamespace(info=lambda *args: None)

    def get_app_path(self, *parts):
        return self.path.joinpath(*parts)

    def get_config(self, key, **kwargs):
        return {"NGINX_HTTP_PORT": 80, "NGINX_HTTPS_ENABLE": True,
                "NGINX_ROOT_DOMAIN": "example.test", "ACME_DNS_API": "dns_test",
                "ACME_SERVER": "letsencrypt", "ACME_ACCOUNT_EMAIL": ""}[key]


def manager_at(root, containers, states=(), model=None):
    calls, restored = [], []
    states = tuple(replace(state, image_id=state.image_id or "sha256:" + state.service) for state in states)

    def process(selected, *args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(check_call=lambda: 0)

    def docker(*args, **kwargs):
        files = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        contents = [Path(path).read_text() for path in files]
        resolved = {}
        for content in contents:
            current = yaml.safe_load(content)
            for key, value in current.items():
                if key == "services":
                    for service, spec in value.items():
                        resolved.setdefault(key, {}).setdefault(service, {}).update(spec)
                else:
                    resolved[key] = value
        if "up" in args:
            restored.extend(contents[:-1])
            original = next(state.image_id for state in states if state.service == args[-1])
            assert resolved["services"][args[-1]]["image"] == original
        return SimpleNamespace(check_call=lambda: 0, model=resolved)

    manager = SimpleNamespace(project_name="test", data_path=root,
        logger=SimpleNamespace(warning=lambda *args: None, error=lambda *args: None),
        containers={c.name: c for c in containers}, integration_snapshot={c.name: () for c in containers},
        generated_configs={}, iter_integrations=lambda consumer: iter(()),
        environ=SimpleNamespace(locks=SimpleNamespace(process_lock=lambda key: nullcontext())),
        lifecycle=SimpleNamespace(notify_start=lambda ctx: nullcontext(), notify_stop=lambda ctx: nullcontext(),
                                  notify_remove=lambda ctx: nullcontext()),
        image_preparer=SimpleNamespace(plan=lambda model, services, **kw:
            ImagePlan(pull=(), build=(), targets=tuple(services))),
        artifact_index=SimpleNamespace(record=lambda entries, remove=(): None, load=lambda: {}),
        running_state=SimpleNamespace(mark_started=lambda ctx: None, mark_stopped=lambda ctx: None),
        resolver=SimpleNamespace(resolve_dependencies=lambda selected: [c for c in containers if c in selected]),
        docker_inspector=SimpleNamespace(get_project_state=lambda selected:
            ProjectRuntimeState("test", tuple(states), "docker")),
        runtime=SimpleNamespace(create_docker_compose_process=process, create_docker_process=docker),
        structured_runner=SimpleNamespace(execute_json=lambda process, **kwargs: process.model))
    stored = {"RUNNING_CONTAINERS": sorted({name for state in states for name in state.logical_containers
                                             if state.state in ("running", "restarting")})}
    manager.cache = SimpleNamespace(get=stored.get, set=stored.__setitem__)
    manager.running_state = RunningStateStore(manager)
    for container in containers:
        container.manager = manager
    runner = manager.compose_runner = ComposeRunner(manager)
    model = model or {"services": {name: spec for c in containers for name, spec in c.services.items()}}
    runner.final_model = lambda ctx: copy.deepcopy(model)
    operations = ComposeOperations(manager)
    operations.select = lambda *a, **kw: ComposeSelection(
        tuple(containers), tuple(containers), tuple(name for c in containers for name in c.services), False)
    return operations, manager, runner, calls, restored


class NginxContainer(Container, BuiltinNginxContainer):
    pass


def running_nginx():
    return ServiceRuntimeState(("nginx",), "nginx", "nginx-runtime", "running", "healthy", "nginx:old", None, {})


def test_full_restart_bootstraps_nginx_after_stop(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:target"}}, tmp_path / "nginx")
    operations, manager, runner, calls, _ = manager_at(
        tmp_path, (nginx,), (running_nginx(),))
    operations.select = lambda *args, **kwargs: ComposeSelection((nginx,), (nginx,), (), True)
    manager.generated_configs["nginx"] = nginx
    nginx.render_config = lambda generation: {"nginx.conf": "final " + generation}
    nginx.render_bootstrap = lambda generation: {"nginx.conf": "bootstrap " + generation}
    old = GeneratedCandidate(nginx, nginx.render_config)
    old.publish()
    AppliedServiceModels(manager, {"services": {"nginx": {"image": "nginx:old"}}}).record(("nginx",))
    nginx.on_prepare_config = lambda context: None
    nginx.validate_config = lambda *args: None
    nginx.confirm = lambda *args: None
    runner.exec_service = lambda *args, **kwargs: SimpleNamespace(succeeded=True, stdout="")
    ready = []
    runner.wait_service_healthy = lambda ctx, service: ready.append((service, tuple(calls)))

    operations.restart()

    assert calls[0] == ("stop",)
    assert ready and all(any(command[0] == "up" for command in seen) for _, seen in ready)


@pytest.mark.parametrize("failure", ["acknowledgment", "application", "readiness"])
def test_restart_bootstrap_failure_restores_generation_and_exact_runtime_snapshot(tmp_path, monkeypatch, failure):
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    old_model = {"services": {"nginx": {"image": "nginx:old", "environment": {"VALUE": "price$$USD"}}},
                 "volumes": {"certs": {"name": "certs-old"}}}
    new_model = {"services": {"nginx": {"image": "nginx:new", "environment": {"VALUE": "new$$VALUE"}}}}
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (nginx,), (running_nginx(),), new_model)
    AppliedServiceModels(manager, old_model).record(("nginx",))
    previous = AppliedServiceModels(manager, new_model).previous["nginx"]
    owner = manager.generated_configs["nginx"] = nginx
    owner.render_config = lambda generation: {"nginx.conf": "serving " + generation}
    prior = GeneratedCandidate(nginx, owner.render_config)
    prior.publish()
    owner.on_prepare_config = lambda context: None
    owner.validate_config = lambda context, candidate: None
    runner.exec_service = lambda *args, **kwargs: SimpleNamespace(succeeded=True, stdout=prior.generation_id)
    confirmed = []

    def confirm(context, generation_id):
        confirmed.append(generation_id)
        raise ContainerError("bootstrap acknowledgment failed")

    owner.confirm = confirm
    original = runner.apply_service

    def apply(context, service, recreate=False):
        if context.generated_candidates["nginx"].generation_id != prior.generation_id:
            raise ContainerError("bootstrap application failed")
        return original(context, service, recreate)

    if failure == "application":
        runner.apply_service = apply
    if failure == "readiness":
        wait_healthy = runner.wait_service_healthy

        def wait(context, service):
            if context.generated_candidates["nginx"].generation_id != prior.generation_id:
                raise ContainerError("bootstrap readiness failed")
            return wait_healthy(context, service)

        runner.wait_service_healthy = wait
    with pytest.raises(ContainerError, match="bootstrap " + failure + " failed"):
        operations.restart(["nginx"])
    assert calls[0] == ("stop", "nginx")
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) == prior.generation_id
    assert restored == [previous]
    assert yaml.safe_load(restored[0]) == old_model
    assert AppliedServiceModels(manager, new_model).previous["nginx"] == previous
    assert manager.running_state.get_persisted() == ["nginx"]
    assert (bool(confirmed)) is (failure == "acknowledgment")


def test_restart_bootstrap_validation_failure_preserves_running_generation_and_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    old_model = {"services": {"nginx": {"image": "nginx:old", "environment": {"VALUE": "old$$value"}}}}
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,), (running_nginx(),))
    manager.generated_configs["nginx"] = nginx
    AppliedServiceModels(manager, old_model).record(("nginx",))
    previous_models = dict(AppliedServiceModels(manager, runner.final_model(None)).previous)
    nginx.render_config = lambda generation: {"nginx.conf": "full " + generation}
    nginx.render_bootstrap = lambda generation: {"nginx.conf": "bootstrap " + generation}
    previous = GeneratedCandidate(nginx, nginx.render_config)
    previous.publish()
    nginx.on_prepare_config = lambda context: None
    validated = []

    def validate(context, candidate):
        content = Path(candidate.path, "nginx.conf").read_text()
        validated.append(content.split()[0])
        if content.startswith("bootstrap "):
            raise ContainerError("bootstrap validation failed")

    def unexpected(*args):
        raise AssertionError("Validation failure must not apply any generation")

    nginx.validate_config = validate
    nginx.apply_config = unexpected
    with pytest.raises(ContainerError, match="bootstrap validation failed"):
        operations.restart(["nginx"])
    assert validated == ["full", "bootstrap"]
    assert calls == []
    assert restored == []
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) == previous.generation_id
    assert AppliedServiceModels(manager, runner.final_model(None)).previous == previous_models
    assert manager.running_state.get_persisted() == ["nginx"]


def test_restart_bootstrap_and_rollback_failures_are_both_reported(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,), (running_nginx(),))
    AppliedServiceModels(manager, {"services": {"nginx": {"image": "nginx:old"}}}).record(("nginx",))
    owner = manager.generated_configs["nginx"] = nginx
    owner.render_config = lambda generation: {"nginx.conf": "serving " + generation}
    prior = GeneratedCandidate(nginx, owner.render_config)
    prior.publish()
    owner.on_prepare_config = lambda context: None
    owner.validate_config = lambda context, candidate: None

    def apply(context, candidate, services):
        if candidate.generation_id == prior.generation_id:
            raise ContainerError("old runtime rejected")
        raise ContainerError("bootstrap rejected")

    owner.apply_config = apply
    with pytest.raises(ContainerError, match="bootstrap rejected.*rollback failed: old runtime rejected"):
        operations.restart(["nginx"])
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) == prior.generation_id
    assert manager.running_state.get_persisted() == []


def test_first_upgrade_failure_restores_legacy_runtime_not_bootstrap(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    old_model = {"services": {"nginx": {"image": "nginx:legacy"}}}
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (nginx,), (running_nginx(),))
    manager.generated_configs["nginx"] = nginx
    AppliedServiceModels(manager, old_model).record(("nginx",))
    previous = AppliedServiceModels(manager, runner.final_model(None)).previous["nginx"]
    nginx.render_config = lambda generation: {"nginx.conf": "final " + generation}
    nginx.render_bootstrap = lambda generation: {"nginx.conf": "bootstrap " + generation}
    nginx.on_prepare_config = lambda context: None
    nginx.validate_config = lambda context, candidate: None
    applied = []

    def apply(context, candidate, services):
        content = Path(candidate.path, "nginx.conf").read_text()
        applied.append(content.split()[0])
        if content.startswith("final "):
            raise ContainerError("final failed")

    nginx.apply_config = apply
    with pytest.raises(ContainerError, match="final failed"):
        operations.restart(["nginx"])
    assert applied == ["bootstrap", "final"]
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) is None
    assert restored == [previous]
    assert manager.running_state.get_persisted() == ["nginx"]


def test_first_upgrade_without_restore_model_fails_before_stopping(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (nginx,), (running_nginx(),))
    manager.generated_configs["nginx"] = nginx
    nginx.render_config = lambda generation: {"nginx.conf": "final " + generation}
    nginx.render_bootstrap = lambda generation: {"nginx.conf": "bootstrap " + generation}
    nginx.on_prepare_config = lambda context: None
    nginx.validate_config = lambda context, candidate: None
    with pytest.raises(ContainerError, match="Cannot replace running service nginx"):
        operations.restart(["nginx"])
    assert calls == []
    assert restored == []
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) is None


def test_cold_nginx_retains_acknowledged_bootstrap_on_final_failure(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    operations, manager, runner, calls, _ = manager_at(tmp_path, (nginx,))
    manager.generated_configs["nginx"] = nginx
    nginx.render_config = lambda generation: {"nginx.conf": "final " + generation}
    nginx.render_bootstrap = lambda generation: {"nginx.conf": "bootstrap " + generation}
    nginx.on_prepare_config = lambda context: None
    nginx.validate_config = lambda context, candidate: None
    applied = []

    def apply(context, candidate, services):
        content = Path(candidate.path, "nginx.conf").read_text()
        applied.append(content.split()[0])
        if content.startswith("final "):
            raise ContainerError("final rejected")

    nginx.apply_config = apply
    with pytest.raises(ContainerError, match="final rejected"):
        operations.up(["nginx"])
    current = GeneratedCandidate.current_id(str(nginx.get_app_path("generated")))
    assert current is not None
    assert (nginx.get_app_path("generated") / "current/nginx.conf").read_text().startswith("bootstrap ")
    assert applied == ["bootstrap", "final", "bootstrap"]
    assert manager.running_state.get_persisted() == ["nginx"]


def test_acme_install_and_runtime_share_config_home():
    path = Path(__file__).parents[2] / "linktools-cntr/src/linktools/assets/containers/100-nginx/Dockerfile"
    text = path.read_text()
    assert "--home /opt/acme --config-home /root/.acme.sh" in text
    assert "ln -s /opt/acme/acme.sh /usr/bin/acme.sh" in text
    assert "--config-home /root/.acme.sh --nocron" in text
    assert "nginx-certificates renew" in text
    assert text.count("FROM nginx:") == 1 and "--issue" in text
    assert "> /etc/crontabs/root" in text


@pytest.mark.parametrize("relation", [
    {"depends_on": {"database": {"condition": "service_healthy"}}},
    {"links": ["database:cache"]},
    {"volumes_from": ["database:ro", "container:external:ro"]},
    {"network_mode": "service:database"},
    {"ipc": "service:database"},
    {"pid": "service:database"},
])
def test_final_model_dependencies_start_before_consumer_without_stopped_siblings(tmp_path, monkeypatch, relation):
    app = Container("app", {"app": {}}, tmp_path / "app")
    providers = Container("providers", {"database": {}, "unrelated": {}}, tmp_path / "providers")
    app.application_priority = -100
    providers.application_priority = 100
    model = {"services": {"app": relation, "database": {}, "unrelated": {}}}
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app, providers), model=model)
    operations.select = lambda *args, **kwargs: ComposeSelection((app, providers), (app,), ("app",), False)
    healthy = []
    runner.wait_service_healthy = lambda context, service, timeout=30: healthy.append(service)
    operations.up(["app"])
    assert [call[-1] for call in calls if call[0] == "up"] == ["database", "app"]
    assert all("--no-deps" in call for call in calls if call[0] == "up")
    assert healthy == (["database"] if "depends_on" in relation else [])


def test_dependency_normalization_preserves_explicit_conditions_and_external_links():
    assert service_dependencies({"depends_on": {"database": {"condition": "service_healthy"}},
        "links": ["database:cache"], "volumes_from": ["container:external:ro"],
        "external_links": ["external"]}) == {"database": {"condition": "service_healthy"}}


def test_container_policy_adds_only_its_required_provider_services(tmp_path, monkeypatch) -> None:
    class MetricsContainer(Container):
        application_priority = 10

        def get_runtime_requirements(self, required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
            assert self.name == "metrics"
            return {"storage": ("database",)} if "metrics" in required else {}

    app = MetricsContainer("metrics", {"metrics": {}}, tmp_path / "metrics")
    storage = Container("storage", {"database": {}, "idle": {}}, tmp_path / "storage")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (storage, app))
    operations.select = lambda *args, **kwargs: ComposeSelection((storage, app), (app,), ("metrics",), False)
    operations.up(["metrics"])
    assert [call[-1] for call in calls if call[0] == "up"] == ["database", "metrics"]


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_container_policy_controls_bootstrap_order_and_changed_running_updates(
        tmp_path, monkeypatch, running, changed) -> None:
    events = []

    class IndexContainer(Container):
        generates_config = True
        application_priority = 100
        bootstrap_services = ("indexer",)

        def generation_label(self, service: str, generation_id: str) -> "str | None":
            return None

        def on_prepare_config(self, context: "OperationContext") -> None:
            pass

        def render_config(self, generation_id: str) -> "dict[str, str]":
            return {"config": self.content, "generation": generation_id}

        def validate_config(self, context: "OperationContext", candidate: "GeneratedCandidate") -> None:
            pass

        def render_bootstrap(self, generation_id: str) -> "dict[str, str]":
            return {"config": "bootstrap " + generation_id}

        def apply_config(self, context: "OperationContext", candidate: "GeneratedCandidate",
                  services: "Iterable[str]") -> None:
            for service in services:
                self.manager.compose_runner.apply_service(context, service)
                content = Path(candidate.path, "config").read_text()
                events.append("bootstrap" if content.startswith("bootstrap ") else "apply")

    app = Container("target", {"target": {}}, tmp_path / "target")
    indexer = IndexContainer("indexer", {"indexer": {}}, tmp_path / "indexer")
    state = ServiceRuntimeState(("indexer",), "indexer", "index-runtime", "running", None, "image", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (indexer, app), (state,) if running else ())
    healthy = []
    runner.wait_service_healthy = lambda context, service, timeout=30: healthy.append(service)
    owner = indexer
    owner.content = "old runtime input"
    manager.generated_configs["indexer"] = owner
    previous = GeneratedCandidate(indexer, owner.render_config)
    previous.publish()
    if changed:
        owner.content = "new runtime input"
    AppliedServiceModels(manager, runner.final_model(None)).record(("indexer", "target"))
    selected = (app,) if running else (app, indexer)
    selected_services = ("target",) if running else ("indexer", "target")
    operations.select = lambda *args, **kwargs: ComposeSelection((indexer, app), selected, selected_services, False)
    operations.up([container.name for container in selected])
    expected_events = (["apply"] if changed else []) if running else ["bootstrap", "apply"]
    assert events == expected_events
    assert healthy == (["indexer"] if running and changed else [])
    expected_starts = (["target", "indexer"] if changed else ["target"]) if running else ["indexer", "target", "indexer"]
    assert [call[-1] for call in calls if call[0] == "up"] == expected_starts
    current = GeneratedCandidate.current_id(str(indexer.get_app_path("generated")))
    assert (current != previous.generation_id) is changed
    assert indexer.get_app_path("generated", "current", "config").read_text() == owner.content


def test_isolated_raw_arguments_decode_only_compose_serialization_and_leave_snapshot_unchanged():
    runner = ComposeRunner(SimpleNamespace(project_name="test"))
    model = {"services": {"app": {"image": "test:image", "environment": {
        "TEXT": "price$$USD", "DOUBLE": "two$$$$signs", "OVERRIDE": "serialized$$value"},
        "volumes": [{"type": "bind", "source": "/host/$$data", "target": "/app/$$data"},
                    {"type": "volume", "source": "data", "target": "/app/$$volume"}],
        "secrets": [{"source": "secret", "target": "$$secret"}],
        "configs": [{"source": "config", "target": "/$$config"}],
        "user": "$$user:$$group", "working_dir": "/app/$$work"}},
        "volumes": {"data": {"name": "volume$$name"}},
        "secrets": {"secret": {"file": "/host/$$secret"}},
        "configs": {"config": {"file": "/host/$$config"}}}
    previous = copy.deepcopy(model)
    command = ("echo", "literal$$command")
    args = runner.isolated_service_args(model, "app", command, {"OVERRIDE": "literal$$override"})
    assert "TEXT=price$USD" in args
    assert "DOUBLE=two$$signs" in args
    assert "OVERRIDE=literal$$override" in args
    assert "type=bind,source=/host/$data,target=/app/$data" in args
    assert "type=volume,source=volume$name,target=/app/$volume" in args
    assert "type=bind,source=/host/$secret,target=/run/secrets/$secret,readonly" in args
    assert "type=bind,source=/host/$config,target=/$config,readonly" in args
    assert "$user:$group" in args
    assert "/app/$work" in args
    assert args[-1] == "literal$$command"
    assert model == previous
    persisted = []

    def process(*args, **kwargs):
        paths = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        if "up" in args:
            persisted.extend(Path(path).read_text() for path in paths[:-1])
        return SimpleNamespace(check_call=lambda: 0)

    runner.manager.runtime = SimpleNamespace(create_docker_process=process)
    runner.manager.structured_runner = SimpleNamespace(execute_json=lambda process, **kwargs: copy.deepcopy(model))
    serialized = yaml.safe_dump(model)
    owner = Container("app", model["services"], Path("/original"))
    context = SimpleNamespace(containers=(owner,), compose_files={"/original/compose.yml": serialized},
                              native_running_images={"app": "sha256:old"})
    runner.apply_saved_services(context, ("app",), {"previous.yml": serialized})
    assert persisted == [serialized]
    assert yaml.safe_load(persisted[0]) == previous


@pytest.mark.parametrize("failure", ["validation", "bootstrap", "final", None])
def test_cold_bootstrap_uses_shared_validation_application_and_final_accounting(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: "str | None") -> None:
    events = []

    class BootstrapContainer(Container):
        generates_config = True
        bootstrap_services = ("example",)

        def render_config(self, generation_id: str) -> "dict[str, str]":
            return {"config": "final", "generation": generation_id}

        def render_bootstrap(self, generation_id: str) -> "dict[str, str]":
            return {"config": "bootstrap", "generation": generation_id}

        def validate_config(self, context: "OperationContext", candidate: "GeneratedCandidate") -> None:
            phase = Path(candidate.path, "config").read_text()
            events.append(("validate", phase))
            if failure == "validation" and phase == "bootstrap":
                raise ContainerError("bootstrap validation failed")

        def apply_config(self, context: "OperationContext", candidate: "GeneratedCandidate",
                         services: "Iterable[str]") -> None:
            phase = Path(candidate.path, "config").read_text()
            events.append(("apply", phase))
            assert "example" not in AppliedServiceModels(self.manager, context.compose_model).previous
            if phase == failure:
                raise ContainerError(phase + " application failed")
            self.manager.compose_runner.apply_services(context, tuple(services))

    container = BootstrapContainer("example", {"example": {"image": "example:new"}}, tmp_path / "example")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (container,))
    manager.generated_configs["example"] = container
    if failure:
        with pytest.raises(ContainerError, match="application failed|validation failed"):
            operations.up(["example"])
    else:
        operations.up(["example"])

    assert events[:2] == [("validate", "final"), ("validate", "bootstrap")]
    expected_apply = {"validation": [], "bootstrap": ["bootstrap"],
                      "final": ["bootstrap", "final", "bootstrap"], None: ["bootstrap", "final"]}
    assert [phase for action, phase in events if action == "apply"] == expected_apply[failure]
    root = container.get_app_path("generated")
    if failure in ("validation", "bootstrap"):
        assert GeneratedCandidate.current_id(str(root)) is None
        assert not calls
    else:
        assert (root / "current/config").read_text() == ("bootstrap" if failure else "final")
    models = AppliedServiceModels(manager, runner.final_model(None)).previous
    assert ("example" in models) is (failure is None)


@pytest.mark.parametrize("failure", ["apply", "readiness"])
def test_plain_first_start_failure_stops_new_service(tmp_path, failure):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,))

    def apply(context, services):
        runner.apply_service(context, services[0])
        if failure == "apply":
            raise ContainerError("plain apply failed")

    runner.apply_services = apply

    def ready(context, service):
        if failure == "readiness":
            raise ContainerError("plain readiness failed")

    app.on_service_started = ready
    with pytest.raises(ContainerError, match="plain " + failure + " failed"):
        operations.up(["app"])
    assert [entry for entry in calls if entry[0] in ("up", "stop")] == [
        next(entry for entry in calls if entry[0] == "up"),
        ("stop", "app"),
    ]
    assert manager.running_state.get_persisted() == []
    assert not (tmp_path / "compose/applied/services" / "617070.yml").exists()


def test_plain_first_start_cleanup_failure_reports_both_errors(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,))

    def fail_readiness(context, service):
        raise ContainerError("health check failed")

    def fail_stop(context, services):
        raise ContainerError("stop failed")

    app.on_service_started = fail_readiness
    runner.stop = fail_stop
    with pytest.raises(ContainerError, match="health check failed.*Compose rollback failed: stop failed"):
        operations.up(["app"])


def test_plain_cold_multi_service_failure_keeps_successful_sibling_running(tmp_path):
    app = Container("app", {"first": {"image": "app:first"},
                            "second": {"image": "app:second"}}, tmp_path / "app")
    operations, manager, runner, calls, _ = manager_at(tmp_path, (app,))

    def started(context, service):
        if service == "second":
            raise ContainerError("second service unhealthy")

    app.on_service_started = started
    with pytest.raises(ContainerError, match="second service unhealthy"):
        operations.up(["app"])
    assert ("stop", "second") in calls
    assert ("stop", "first") not in calls
    assert manager.running_state.get_persisted() == ["app"]


def test_plain_partial_start_failure_preserves_running_sibling(tmp_path):
    app = Container("app", {"existing": {"image": "app:old"},
                             "new": {"image": "app:new"}}, tmp_path / "app")
    current = ServiceRuntimeState(("app",), "existing", "existing-runtime",
                                  "running", "healthy", "app:old", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,), (current,))
    AppliedServiceModels(manager, runner.final_model(None)).record(("existing",))
    operations.select = lambda *args, **kwargs: ComposeSelection(
        (app,), (app,), ("new",), False)
    app.on_service_started = lambda context, service: (_ for _ in ()).throw(
        ContainerError("new service unhealthy"))
    with pytest.raises(ContainerError, match="new service unhealthy"):
        operations.up(["app"])
    assert ("stop", "new") in calls
    assert ("stop", "existing") not in calls
    assert manager.running_state.get_persisted() == ["app"]


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_plain_restart_rollback_records_only_restored_owner(tmp_path, monkeypatch, rollback_fails):
    app = Container("app", {"app": {"image": "app:new"}, "idle": {}}, tmp_path / "app")
    later = Container("later", {"later": {}}, tmp_path / "later")
    untouched = Container("untouched", {"untouched": {}}, tmp_path / "untouched")
    states = tuple(ServiceRuntimeState((name,), name, name, "running", None, "old", None, {})
                   for name in ("app", "later", "untouched"))
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app, later, untouched), states)
    old_model = runner.final_model(None)
    old_model["services"]["app"]["image"] = "app:old"
    AppliedServiceModels(manager, old_model).record(tuple(old_model["services"]))
    previous = dict(AppliedServiceModels(manager, runner.final_model(None)).previous)
    operations.select = lambda *args, **kwargs: ComposeSelection(
        (app, later, untouched), (app, later), ("app", "later"), False)
    applied = []

    def fail_apply(context, services):
        applied.append(tuple(services))
        raise ContainerError("new runtime rejected")

    runner.apply_services = fail_apply
    original = runner.apply_saved_services
    restored_services = []

    def restore(context, services, files):
        restored_services.append(tuple(services))
        if rollback_fails:
            raise ContainerError("old runtime rejected")
        return original(context, services, files)

    runner.apply_saved_services = restore
    message = "new runtime rejected"
    if rollback_fails:
        message += ".*Compose rollback failed: old runtime rejected"
    with pytest.raises(ContainerError, match=message):
        operations.restart(["app", "later"])
    assert calls == [("stop", "app", "later")]
    assert applied == [("app",)]
    assert restored_services == [("app",), ("later",)]
    assert restored == ([] if rollback_fails else [previous["app"], previous["later"]])
    assert AppliedServiceModels(manager, runner.final_model(None)).previous == previous
    assert manager.running_state.get_persisted() == (
        ["untouched"] if rollback_fails else ["app", "later", "untouched"])


@pytest.mark.parametrize("loaded_generation", ["new", "old"])
def test_nginx_waits_for_health_before_generation_probe_or_reload(tmp_path, monkeypatch, loaded_generation):
    nginx = NginxContainer("nginx", {"nginx": {}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,))
    clock, events = [0.0], []
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    statuses = iter(("starting", "starting", "healthy"))

    def inspect(containers):
        health = next(statuses)
        events.append(health)
        state = ServiceRuntimeState(("nginx",), "nginx", "nginx", "running", health, "image", None, {})
        return ProjectRuntimeState("test", (state,), "docker")

    manager.docker_inspector.get_project_state = inspect
    runner.apply_service = lambda *args: events.append("apply")
    responses = iter((loaded_generation, "old", "new"))

    def execute(context, service, command, check=True):
        assert events[:4] == ["apply", "starting", "starting", "healthy"]
        if command[0] == "nginx":
            assert check
            events.append("reload")
            return SimpleNamespace(succeeded=True)
        assert not check
        events.append("probe")
        return SimpleNamespace(succeeded=True, stdout=next(responses))

    runner.exec_service = execute
    nginx.apply_config(SimpleNamespace(containers=[nginx]), SimpleNamespace(generation_id="new", path=str(tmp_path / "generation")), ("nginx",))
    assert events[4:] == (["probe"] if loaded_generation == "new" else ["probe", "reload", "probe", "probe"])
    assert clock[0] == (1.0 if loaded_generation == "new" else 1.25)


@pytest.mark.parametrize("state,health", [("running", "starting"), ("exited", "unhealthy")])
def test_nginx_readiness_timeout_never_attempts_reload(tmp_path, monkeypatch, state, health):
    nginx = NginxContainer("nginx", {"nginx": {}}, tmp_path / "nginx")
    runtime = ServiceRuntimeState(("nginx",), "nginx", "nginx", state, health, "image", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,), (runtime,))
    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    runner.apply_service = lambda *args: None

    def unexpected(*args, **kwargs):
        raise AssertionError("An unready nginx must not be probed or reloaded")

    runner.exec_service = unexpected
    with pytest.raises(ContainerError, match="Service nginx did not become healthy"):
        nginx.apply_config(SimpleNamespace(containers=[nginx]), SimpleNamespace(generation_id="new", path=str(tmp_path / "generation")), ("nginx",))
    assert clock[0] == 30.0


def test_nginx_healthy_old_generation_still_requires_bounded_acknowledgment(tmp_path, monkeypatch):
    nginx = NginxContainer("nginx", {"nginx": {}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,), (running_nginx(),))
    clock, commands = [0.0], []
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    runner.apply_service = lambda *args: None

    def execute(context, service, command, check=True):
        commands.append(command[0])
        return SimpleNamespace(succeeded=True, stdout="old")

    runner.exec_service = execute
    with pytest.raises(ContainerError, match="Nginx did not acknowledge the generated configuration"):
        nginx.apply_config(SimpleNamespace(containers=[nginx]), SimpleNamespace(generation_id="new", path=str(tmp_path / "generation")), ("nginx",))
    assert commands.count("nginx") == 1
    assert clock[0] == 30.0


def test_restart_restores_unattempted_targets_after_stop_hook_failure(tmp_path):
    from contextlib import contextmanager

    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    other = Container("other", {"other": {"image": "other:new"}}, tmp_path / "other")
    states = tuple(ServiceRuntimeState((name,), name, name, "running", None, "old", None, {})
                   for name in ("app", "other"))
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app, other), states)
    old_model = runner.final_model(None)
    AppliedServiceModels(manager, old_model).record(("app", "other"))
    saved = dict(AppliedServiceModels(manager, old_model).previous)

    @contextmanager
    def broken_stop_hook(context):
        yield
        raise ContainerError("on_stopped failed")

    manager.lifecycle.notify_stop = broken_stop_hook
    with pytest.raises(ContainerError, match="on_stopped failed"):
        operations.restart()
    assert calls == [("stop", "app", "other")]
    assert restored == [saved["app"], saved["other"]]
    assert manager.running_state.get_persisted() == ["app", "other"]


def test_restart_recovers_unattempted_services_in_old_dependency_order(tmp_path):
    provider = Container("db", {"db": {"image": "db:new"}}, tmp_path / "db")
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    app.services["app"]["depends_on"] = {"db": {"condition": "service_started"}}
    states = tuple(ServiceRuntimeState((name,), name, name, "running", None, "old", None, {})
                   for name in ("db", "app"))
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app, provider), states)
    old = runner.final_model(None)
    AppliedServiceModels(manager, old).record(("db", "app"))

    from contextlib import contextmanager

    @contextmanager
    def fail_after_stop(context):
        yield
        raise ContainerError("post-stop rejected")

    manager.lifecycle.notify_stop = fail_after_stop
    with pytest.raises(ContainerError, match="post-stop rejected"):
        operations.restart()
    assert calls == [("stop", "app", "db")]
    previous = AppliedServiceModels(manager, old).previous
    assert restored == [previous["db"], previous["app"]]
    assert manager.running_state.get_persisted() == ["app", "db"]


@pytest.mark.parametrize("stopped_service", ["app", "db", None])
def test_restart_partial_stop_failure_recovers_only_stopped_targets(tmp_path, stopped_service):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    db = Container("db", {"db": {"image": "db:new"}}, tmp_path / "db")
    observer = Container("observer", {"observer": {"image": "observer:old"}}, tmp_path / "observer")
    states = tuple(ServiceRuntimeState((name,), name, name, "running", None, "old", None, {})
                   for name in ("app", "db", "observer"))
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (app, db, observer), states)
    prior = runner.final_model(None)
    AppliedServiceModels(manager, prior).record(("app", "db", "observer"))
    original = dict(AppliedServiceModels(manager, prior).previous)
    operations.select = lambda *args, **kwargs: ComposeSelection(
        (app, db, observer), (app, db), ("app", "db"), False)

    actual = [replace(state, image_id="sha256:" + state.service) for state in states]
    manager.docker_inspector.get_project_state = lambda owners: ProjectRuntimeState(
        "test", tuple(actual), "docker")

    def stop_partially(context, services):
        calls.append(("stop", *services))
        if stopped_service is not None:
            offset = ("app", "db", "observer").index(stopped_service)
            actual[offset] = replace(actual[offset], state="exited", exit_code=0)
        raise ContainerError("partial stop failed")

    runner.stop = stop_partially
    saved_apply = runner.apply_saved_services

    def restore(context, services, files):
        saved_apply(context, services, files)
        for index, state in enumerate(actual):
            if state.service in services:
                actual[index] = replace(state, state="running")

    runner.apply_saved_services = restore

    with pytest.raises(ContainerError, match="partial stop failed"):
        operations.restart(["app", "db"])

    assert calls == [("stop", "app", "db")]
    assert restored == ([] if stopped_service is None else [original[stopped_service]])
    assert all(state.state == "running" for state in actual)
    assert manager.running_state.get_persisted() == ["app", "db", "observer"]


def test_partial_stop_failure_preserves_inspection_failure(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    state = ServiceRuntimeState(("app",), "app", "app", "running", None, "old", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,), (state,))
    AppliedServiceModels(manager, runner.final_model(None)).record(("app",))
    inspections = [0]

    def inspect(containers):
        inspections[0] += 1
        if inspections[0] == 1:
            return ProjectRuntimeState("test", (replace(state, image_id="sha256:app"),), "docker")
        raise RuntimeError("Docker unavailable")

    manager.docker_inspector.get_project_state = inspect
    runner.stop = lambda context, services: (_ for _ in ()).throw(
        ContainerError("partial stop failed"))
    with pytest.raises(ContainerError, match="partial stop failed.*recovery inspection failed.*Docker unavailable"):
        operations.restart(["app"])
    assert restored == []


def test_partial_stop_restores_service_when_only_one_replica_was_stopped(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    states = (
        ServiceRuntimeState(("app",), "app", "app-1", "running", None, "old", None, {}),
        ServiceRuntimeState(("app",), "app", "app-2", "running", None, "old", None, {}),
    )
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,), states)
    model = runner.final_model(None)
    AppliedServiceModels(manager, model).record(("app",))
    old = AppliedServiceModels(manager, model).previous["app"]
    actual = [replace(state, image_id="sha256:app") for state in states]
    manager.docker_inspector.get_project_state = lambda owners: ProjectRuntimeState(
        "test", tuple(actual), "docker")

    def fail_during_stop(context, services):
        calls.append(("stop", *services))
        actual[0] = replace(actual[0], state="exited", exit_code=0)
        raise ContainerError("replica stop failed")

    runner.stop = fail_during_stop
    original = runner.apply_saved_services

    def recover(context, services, files):
        original(context, services, files)
        for index, state in enumerate(actual):
            actual[index] = replace(state, state="running")

    runner.apply_saved_services = recover
    with pytest.raises(ContainerError, match="replica stop failed"):
        operations.restart(["app"])
    assert calls == [("stop", "app")]
    assert restored == [old]
    assert all(state.state == "running" for state in actual)
    assert manager.running_state.get_persisted() == ["app"]


def test_restart_uses_old_dependency_order_after_new_apply_rejected(tmp_path):
    a = Container("a", {"a": {"image": "a:new"}}, tmp_path / "a")
    b = Container("b", {"b": {"image": "b:new"}}, tmp_path / "b")
    old = {"services": {
        "a": {"image": "a:old", "depends_on": {
            "b": {"condition": "service_healthy"}}},
        "b": {"image": "b:old", "healthcheck": {"test": ["CMD", "true"]}}}}
    states = [
        ServiceRuntimeState((name,), name, name, "running",
                            "healthy" if name == "b" else None, "old", None, {})
        for name in ("a", "b")]
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (a, b), tuple(states))
    AppliedServiceModels(manager, old).record(("a", "b"))
    actual = [replace(state, image_id="sha256:" + state.service) for state in states]
    manager.docker_inspector.get_project_state = lambda containers: ProjectRuntimeState(
        "test", tuple(actual), "docker")
    stop = runner.stop

    def stop_targets(context, services):
        result = stop(context, services)
        for index, item in enumerate(actual):
            if item.service in services:
                actual[index] = replace(item, state="exited", exit_code=0)
        return result

    runner.stop = stop_targets
    runner.apply_services = lambda context, services: (_ for _ in ()).throw(
        ContainerError("new a rejected"))
    apply_saved = runner.apply_saved_services

    def restore(context, services, files):
        result = apply_saved(context, services, files)
        for index, item in enumerate(actual):
            if item.service in services:
                actual[index] = replace(item, state="running",
                    health="healthy" if item.service == "b" else None, exit_code=None)
        return result

    runner.apply_saved_services = restore
    with pytest.raises(ContainerError, match="new a rejected"):
        operations.restart()
    assert [item.service for item in actual if item.state == "running"] == ["a", "b"]
    assert restored == [AppliedServiceModels(manager, old).previous[name] for name in ("b", "a")]
    assert manager.running_state.get_persisted() == ["a", "b"]


def test_first_migration_restores_legacy_target_without_unrelated_corrupt_file(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    stopped = Container("stopped", {"stopped": {"image": "stopped:new"}}, tmp_path / "stopped")
    state = ServiceRuntimeState(("app",), "app", "app", "running",
                                None, "app:old", None, {})
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (app, stopped), (state,))
    operations.select = lambda *args, **kwargs: ComposeSelection(
        (app, stopped), (app,), ("app",), False)
    root = tmp_path / "compose"
    root.mkdir(parents=True)
    (root / "app.yml").write_text("services:\n  app:\n    image: app:old\n")
    (root / "stopped.yml").write_text("services: [invalid")
    runner.apply_services = lambda context, services: (_ for _ in ()).throw(
        ContainerError("new app rejected"))

    with pytest.raises(ContainerError, match="new app rejected"):
        operations.restart(["app"])
    assert restored and "app:old" in restored[0]
    assert not any("stopped" in content for content in restored)
    assert manager.running_state.get_persisted() == ["app"]


def test_first_migration_invalid_required_old_file_fails_before_stop(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    state = ServiceRuntimeState(("app",), "app", "app", "running",
                                None, "app:old", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,), (state,))
    root = tmp_path / "compose"
    root.mkdir(parents=True)
    (root / "app.yml").write_text("services: [invalid")
    with pytest.raises(Exception):
        operations.restart(["app"])
    assert not any(item[0] == "stop" for item in calls)
    assert manager.running_state.get_persisted() == ["app"]


def test_partial_stop_failed_recovery_records_actual_stopped_state(tmp_path):
    app = Container("app", {"app": {"image": "app:new"}}, tmp_path / "app")
    state = ServiceRuntimeState(("app",), "app", "app", "running",
                                None, "app:old", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app,), (state,))
    AppliedServiceModels(manager, runner.final_model(None)).record(("app",))
    actual = [replace(state, image_id="sha256:app")]
    manager.docker_inspector.get_project_state = lambda owners: ProjectRuntimeState(
        "test", tuple(actual), "docker")

    def fail_stop(context, services):
        actual[0] = replace(actual[0], state="exited", exit_code=0)
        raise ContainerError("partial stop failed")

    runner.stop = fail_stop
    runner.apply_saved_services = lambda *args: (_ for _ in ()).throw(
        ContainerError("old image rejected"))
    with pytest.raises(ContainerError, match="partial stop failed.*recovery failed"):
        operations.restart()
    assert actual[0].state == "exited"
    assert manager.running_state.get_persisted() == []
