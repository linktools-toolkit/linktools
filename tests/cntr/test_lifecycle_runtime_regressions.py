#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks at mocked lifecycle and raw-Docker boundaries."""
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from linktools.cntr.integration import IntegrationConsumer
from _harness import builtin_consumer_type
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.artifacts import AppliedServiceModels, GeneratedCandidate
from linktools.cntr.container import ContainerError
from linktools.cntr.runtime.compose import ComposeRunner, service_dependencies
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState
from linktools.cntr.state.running import RunningStateStore

if TYPE_CHECKING:
    from typing import AbstractSet, Iterable, Mapping
    from linktools.cntr import ContainerManager, EventContext


NginxGeneration = builtin_consumer_type("100-nginx")


class Container:
    dependencies = ()
    docker_file = None
    sites = {}
    integration_consumer = None

    def __init__(self, name, services, path):
        self.name, self.services, self.path = name, services, path
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

    def process(selected, *args, **kwargs):
        calls.append(tuple(args))
        return SimpleNamespace(check_call=lambda: 0)

    def docker(*args, **kwargs):
        files = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        restored.extend(Path(path).read_text() for path in files)
        return SimpleNamespace(check_call=lambda: 0)

    manager = SimpleNamespace(project_name="test", data_path=root, logger=None,
        containers={c.name: c for c in containers}, integration_snapshot={c.name: () for c in containers},
        integration_consumers={}, generated_configs={}, iter_integrations=lambda consumer: iter(()),
        environ=SimpleNamespace(locks=SimpleNamespace(process_lock=lambda key: nullcontext())),
        lifecycle=SimpleNamespace(notify_start=lambda ctx: nullcontext(), notify_stop=lambda ctx: nullcontext(),
                                  notify_remove=lambda ctx: nullcontext()),
        image_preparer=SimpleNamespace(plan=lambda *a, **kw: SimpleNamespace(pull=(), build=())),
        artifact_index=SimpleNamespace(record=lambda entries: None),
        running_state=SimpleNamespace(mark_started=lambda ctx: None, mark_stopped=lambda ctx: None),
        resolver=SimpleNamespace(resolve_dependencies=lambda selected: [c for c in containers if c in selected]),
        docker_inspector=SimpleNamespace(get_project_state=lambda selected:
            ProjectRuntimeState("test", tuple(states), "docker")),
        runtime=SimpleNamespace(create_docker_compose_process=process, create_docker_process=docker))
    stored = {"RUNNING_CONTAINERS": sorted({name for state in states for name in state.logical_containers
                                             if state.state in ("running", "restarting")})}
    manager.cache = SimpleNamespace(get=stored.get, set=stored.__setitem__)
    manager.running_state = RunningStateStore(manager)
    for container in containers:
        container.manager = manager
        consumer = container.integration_consumer
        if consumer is not None:
            manager.integration_consumers[container.name] = consumer
            if consumer.generated:
                manager.generated_configs[container.name] = consumer
    runner = manager.compose_runner = ComposeRunner(manager)
    model = model or {"services": {name: spec for c in containers for name, spec in c.services.items()}}
    runner.final_model = lambda ctx: copy.deepcopy(model)
    operations = ComposeOperations(manager)
    operations.select = lambda *a, **kw: ComposeSelection(
        tuple(containers), tuple(containers), tuple(name for c in containers for name in c.services), False)
    return operations, manager, runner, calls, restored


def running_nginx():
    return ServiceRuntimeState(("nginx",), "nginx", "nginx-runtime", "running", "healthy", "nginx:old", None, {})


@pytest.mark.parametrize("failure", ["acknowledgment", "application"])
def test_restart_bootstrap_failure_restores_generation_and_exact_runtime_snapshot(tmp_path, monkeypatch, failure):
    nginx = Container("nginx", {"nginx": {"image": "nginx:new"}}, tmp_path / "nginx")
    old_model = {"services": {"nginx": {"image": "nginx:old", "environment": {"VALUE": "price$$USD"}}},
                 "volumes": {"certs": {"name": "certs-old"}}}
    new_model = {"services": {"nginx": {"image": "nginx:new", "environment": {"VALUE": "new$$VALUE"}}}}
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (nginx,), (running_nginx(),), new_model)
    AppliedServiceModels(manager, old_model).record(("nginx",))
    previous = AppliedServiceModels(manager, new_model).previous["nginx"]
    owner = manager.generated_configs["nginx"] = NginxGeneration(nginx)
    nginx.integration_consumer = owner
    manager.integration_consumers["nginx"] = owner
    owner.render = lambda generation: {"nginx.conf": "serving " + generation}
    prior = GeneratedCandidate(nginx, owner.render)
    prior.publish()
    owner.prepare = lambda context: None
    owner.validate = lambda candidate, context: None
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})
    runner.exec_service = lambda *args, **kwargs: SimpleNamespace(succeeded=True, stdout=prior.generation_id)
    confirmed = []

    def confirm(context, generation_id):
        confirmed.append(generation_id)
        raise ContainerError("bootstrap acknowledgment failed")

    owner.confirm = confirm
    original = runner.apply_service

    def apply(context, service, recreate=False):
        if recreate:
            raise ContainerError("bootstrap application failed")
        return original(context, service, recreate)

    if failure == "application":
        runner.apply_service = apply
    with pytest.raises(ContainerError, match="bootstrap " + failure + " failed"):
        operations.restart(["nginx"])
    assert calls[0] == ("stop", "nginx")
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) == prior.generation_id
    assert restored == [previous]
    assert yaml.safe_load(restored[0]) == old_model
    assert AppliedServiceModels(manager, new_model).previous["nginx"] == previous
    assert manager.running_state.get_persisted() == ["nginx"]
    assert (bool(confirmed)) is (failure == "acknowledgment")


def test_restart_bootstrap_and_rollback_failures_are_both_reported(tmp_path, monkeypatch):
    nginx = Container("nginx", {"nginx": {}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,), (running_nginx(),))
    owner = manager.generated_configs["nginx"] = NginxGeneration(nginx)
    nginx.integration_consumer = owner
    manager.integration_consumers["nginx"] = owner
    owner.render = lambda generation: {"nginx.conf": "serving " + generation}
    prior = GeneratedCandidate(nginx, owner.render)
    prior.publish()
    owner.prepare = lambda context: None
    owner.validate = lambda candidate, context: None
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})

    def failed_bootstrap(context, generation_id):
        raise ContainerError("bootstrap rejected")

    def failed_rollback(*args):
        raise ContainerError("old runtime rejected")

    owner.confirm = failed_bootstrap
    owner.apply = failed_rollback
    with pytest.raises(ContainerError, match="bootstrap rejected.*rollback failed: old runtime rejected"):
        operations.restart(["nginx"])
    assert GeneratedCandidate.current_id(str(nginx.get_app_path("generated"))) == prior.generation_id
    assert manager.running_state.get_persisted() == []


def test_same_generation_certificate_renewal_forces_reload_and_acknowledgment(tmp_path):
    nginx = Container("nginx", {"nginx": {}}, tmp_path / "nginx")
    operations, manager, runner, calls, restored = manager_at(tmp_path, (nginx,))
    owner = NginxGeneration(nginx)
    render = lambda generation: {"nginx.conf": "generation " + generation + "\nssl_certificate /etc/certs/example.test_fullchain.pem;"}
    prior = GeneratedCandidate(nginx, render)
    prior.publish()
    nginx.get_app_path("certs").mkdir()
    certificate = nginx.get_app_path("certs", "example.test_fullchain.pem")
    certificate.write_text("old-certificate")
    nginx.get_app_path("certs", "example.test_key.pem").write_text("old-key")
    validation, execution, confirmation = [], [], []

    def validate(context, service, command, **kwargs):
        validation.append(command[-1])
        if "openssl x509" in command[-1]:
            raise ContainerError("renewal due")
        certificate.write_text("new-certificate")

    def execute(context, service, command, **kwargs):
        execution.append(tuple(command))
        return SimpleNamespace(succeeded=True, stdout=prior.generation_id)

    runner.validate_service = validate
    runner.exec_service = execute
    owner.confirm = lambda context, generation_id: confirmation.append(generation_id)
    context = SimpleNamespace(initial_services={"nginx"}, containers=(nginx,), is_full_containers=False)
    owner.prepare(context)
    candidate = GeneratedCandidate(nginx, render)
    assert not candidate.changed
    assert context.nginx_certificate_replaced
    owner.apply(candidate, context, ("nginx",))
    assert certificate.read_text() == "new-certificate"
    assert "--reloadcmd" in validation[-1]
    assert "[ -s /var/run/nginx.pid ]" in validation[-1]
    assert "nginx.conf -t &&" in validation[-1]
    assert any(command[-2:] == ("-s", "reload") for command in execution)
    assert confirmation == [candidate.generation_id]
    assert not context.nginx_certificate_replaced
    execution.clear()
    owner.apply(candidate, context, ("nginx",))
    assert len(execution) == 1
    assert execution[0][0] == "curl"


@pytest.mark.parametrize("running", [False, True])
def test_renewal_reconciles_only_running_nginx_for_unrelated_partial_up(tmp_path, monkeypatch, running):
    nginx = Container("nginx", {"nginx": {}}, tmp_path / "nginx")
    target = Container("target", {"target": {}}, tmp_path / "target")
    operations, manager, runner, calls, restored = manager_at(
        tmp_path, (target, nginx), (running_nginx(),) if running else ())
    owner = manager.generated_configs["nginx"] = NginxGeneration(nginx)
    nginx.integration_consumer = owner
    manager.integration_consumers["nginx"] = owner
    owner.render = lambda generation: {"nginx.conf": "serving " + generation}
    prior = GeneratedCandidate(nginx, owner.render)
    prior.publish()
    owner.prepare = lambda context: setattr(context, "nginx_certificate_replaced", True)
    owner.validate = lambda candidate, context: None
    owner.confirm = lambda context, generation_id: None
    runner.exec_service = lambda *args, **kwargs: SimpleNamespace(succeeded=True, stdout=prior.generation_id)
    AppliedServiceModels(manager, runner.final_model(None)).record(("nginx", "target"))
    operations.select = lambda *args, **kwargs: ComposeSelection((target, nginx), (target,), ("target",), False)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})
    operations.up(["target"])
    assert [call[-1] for call in calls if call[0] == "up"] == (["target", "nginx"] if running else ["target"])


def test_acme_install_and_runtime_share_config_home():
    path = Path(__file__).parents[2] / "linktools-cntr/src/linktools/assets/containers/100-nginx/Dockerfile"
    text = path.read_text()
    assert "--home /opt/acme --config-home /root/.acme.sh" in text
    assert "ln -s /opt/acme/acme.sh /usr/bin/acme.sh" in text


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
    model = {"services": {"app": relation, "database": {}, "unrelated": {}}}
    operations, manager, runner, calls, restored = manager_at(tmp_path, (app, providers), model=model)
    operations.select = lambda *args, **kwargs: ComposeSelection((app, providers), (app,), ("app",), False)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})
    healthy = []
    runner.wait_service_healthy = lambda context, service: healthy.append(service)
    operations.up(["app"])
    assert [call[-1] for call in calls if call[0] == "up"] == ["database", "app"]
    assert all("--no-deps" in call for call in calls if call[0] == "up")
    assert healthy == (["database"] if "depends_on" in relation else [])


def test_dependency_normalization_preserves_explicit_conditions_and_external_links():
    assert service_dependencies({"depends_on": {"database": {"condition": "service_healthy"}},
        "links": ["database:cache"], "volumes_from": ["container:external:ro"],
        "external_links": ["external"]}) == {"database": {"condition": "service_healthy"}}


def test_consumer_policy_adds_only_its_required_provider_services(tmp_path, monkeypatch) -> None:
    class MetricsConsumer(IntegrationConsumer):
        application_order = 10

        @classmethod
        def runtime_requirements(cls, manager: "ContainerManager",
                                 required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
            return {"storage": ("database",)} if "metrics" in required else {}

    app = Container("metrics", {"metrics": {}}, tmp_path / "metrics")
    storage = Container("storage", {"database": {}, "idle": {}}, tmp_path / "storage")
    app.integration_consumer = MetricsConsumer(app)
    operations, manager, runner, calls, restored = manager_at(tmp_path, (storage, app))
    operations.select = lambda *args, **kwargs: ComposeSelection((storage, app), (app,), ("metrics",), False)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})
    operations.up(["metrics"])
    assert [call[-1] for call in calls if call[0] == "up"] == ["database", "metrics"]


@pytest.mark.parametrize("running", [False, True])
def test_consumer_policy_controls_bootstrap_order_and_runtime_only_updates(tmp_path, monkeypatch, running) -> None:
    events = []

    class IndexConsumer(IntegrationConsumer):
        generated = True
        application_order = 100
        uses_generation_label = False

        def prepare(self, context: "EventContext") -> None:
            context.index_changed = True

        def render(self, generation_id: str) -> "dict[str, str]":
            return {"config": "serving " + generation_id}

        def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
            pass

        def needs_apply(self, candidate: "GeneratedCandidate", context: "EventContext") -> bool:
            assert not candidate.changed
            return context.index_changed

        def needs_bootstrap(self, services: "Iterable[str]", running_services: "AbstractSet[str]") -> bool:
            return "indexer" in services and "indexer" not in running_services

        def bootstrap(self, context: "EventContext") -> str:
            candidate = GeneratedCandidate(self.container, lambda generation: {"config": "bootstrap " + generation})
            candidate.publish()
            self.container.manager.compose_runner.apply_service(context, "indexer", recreate=True)
            events.append("bootstrap")
            return candidate.generation_id

        def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
                  services: "Iterable[str]") -> None:
            for service in services:
                self.container.manager.compose_runner.apply_service(context, service)
                events.append("apply")

    app = Container("target", {"target": {}}, tmp_path / "target")
    indexer = Container("indexer", {"indexer": {}}, tmp_path / "indexer")
    indexer.integration_consumer = IndexConsumer(indexer)
    state = ServiceRuntimeState(("indexer",), "indexer", "index-runtime", "running", None, "image", None, {})
    operations, manager, runner, calls, restored = manager_at(tmp_path, (indexer, app), (state,) if running else ())
    owner = indexer.integration_consumer
    previous = GeneratedCandidate(indexer, owner.render)
    previous.publish()
    AppliedServiceModels(manager, runner.final_model(None)).record(("indexer", "target"))
    selected = (app,) if running else (app, indexer)
    selected_services = ("target",) if running else ("indexer", "target")
    operations.select = lambda *args, **kwargs: ComposeSelection((indexer, app), selected, selected_services, False)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {})
    operations.up([container.name for container in selected])
    assert events == (["apply"] if running else ["bootstrap", "apply"])
    assert [call[-1] for call in calls if call[0] == "up"] == (
        ["target", "indexer"] if running else ["indexer", "target", "indexer"])
    assert GeneratedCandidate.current_id(str(indexer.get_app_path("generated"))) == previous.generation_id


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

    def process(*args):
        paths = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--file"]
        persisted.extend(Path(path).read_text() for path in paths)
        return SimpleNamespace(check_call=lambda: 0)

    runner.manager.runtime = SimpleNamespace(create_docker_process=process)
    serialized = yaml.safe_dump(model)
    runner.apply_saved_services(SimpleNamespace(), ("app",), {"previous.yml": serialized})
    assert persisted == [serialized]
    assert yaml.safe_load(persisted[0]) == previous
