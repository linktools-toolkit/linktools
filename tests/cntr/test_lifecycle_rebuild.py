#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import OperationContext
from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.errors import ContainerError
from linktools.cntr.lifecycle.dispatcher import LifecycleDispatcher
from linktools.cntr.lifecycle.hooks import HookPhase, HookRegistry
from linktools.cntr.runtime.compose import ComposeRunner, order_services
from linktools.cntr.runtime.inspect import ProjectRuntimeState


class Hooks(HookRegistry):
    def __init__(self, events, owner):
        super().__init__(owner=owner)
        self.events, self.event_owner = events, owner

    def call(self, phase, context, reverse=False):
        self.events.append((phase.value, self.event_owner))
        super().call(phase, context, reverse=reverse)


class Index:
    def __init__(self):
        self.values = {}

    def record(self, values, remove=()):
        self.values.update(values)
        for key in remove:
            self.values.pop(key, None)

    def load(self):
        return dict(self.values)


class Owner:
    def __init__(self, name, services, manager):
        self.name, self.services, self.manager = name, services, manager
        self.dependencies = ()
        self.docker_file = None
        self.docker_compose = {"services": services}
        self.hooks = Hooks(manager.events, name)

    def add_start_hook(self, key, hook, **kwargs):
        self.hooks.register(HookPhase.BEFORE_START, hook, key=key, **kwargs)

    def on_prepare(self):
        pass

    def reset_rendered_inputs(self):
        pass

    def on_starting(self, context):
        self.manager.events.append(("prepare", self.name))

    def on_check(self, context):
        self.manager.events.append(("check-callback", self.name))

    def on_started(self, context):
        self.manager.events.append(("after-callback", self.name))

    def on_stopping(self, context):
        self.manager.events.append(("stopping-callback", self.name))

    def on_stopped(self, context):
        self.manager.events.append(("stopped-callback", self.name))

    def on_removed(self, context):
        self.manager.events.append(("removed-callback", self.name))

    def register_fields(self):
        pass

    def get_app_path(self, *parts):
        return self.manager.data_path.joinpath("app", self.name, *parts)


class Runner:
    def __init__(self, manager):
        self.manager = manager
        self.fail = None
        self.restore_fails = False

    def final_model(self, context):
        self.manager.events.append(("model",))
        return deepcopy(self.manager.model)

    def pull_args(self, services):
        return ["pull", *services]

    def options_for_build(self, services, pull=False):
        return SimpleNamespace(services=services, pull=pull)

    def build_args(self, options):
        return ["build", *options.services]

    def pull(self, context, services):
        self.manager.events.append(("images", tuple(services)))

    def build(self, context, options):
        self.manager.events.append(("images", tuple(options.services)))

    def apply_service(self, context, service, recreate=False):
        self.manager.events.append(("apply", service, recreate))
        if self.fail == service:
            raise ContainerError("apply failed " + service)

    def apply_services(self, context, services):
        self.applied = []
        updated, recreated = set(), set()
        initial = context.initial_runtime_state.running_services
        from linktools.cntr.runtime.compose import namespace_dependencies, service_dependencies
        for service in services:
            spec = context.compose_model["services"][service]
            changed = (service in context.service_models.changed_services or
                       context.initial_runtime_state.running_images.get(service) !=
                       self.manager.image_preparer.image_id(spec["image"]) or
                       bool(set(namespace_dependencies(spec)) & recreated))
            cascade = any(options.get("restart") and dependency in updated for
                          dependency, options in service_dependencies(spec).items())
            if cascade and service in initial and not changed:
                self.restart_service(context, service)
            else:
                self.apply_service(context, service, recreate=changed)
            self.applied.append(service)
            if changed or service not in initial:
                recreated.add(service)
            if changed or cascade or service not in initial:
                updated.add(service)

    def ready_services(self, context, services, state, image_ids):
        return set(self.applied)

    def wait_services_ready(self, context, services, image_ids):
        active = {service for service in services if self.wait_service_ready(context, service)}
        return ProjectRuntimeState("test", tuple(SimpleNamespace(
            service=service, state="running", labels={}) for service in active), "docker")

    def restart_service(self, context, service, model=None):
        self.manager.events.append(("restart", service))
        if self.fail == service:
            raise ContainerError("restart failed " + service)

    def wait_service_ready(self, context, service, model=None):
        self.manager.events.append(("ready", service))
        return True

    def stop(self, context, services):
        self.manager.events.append(("stop", tuple(services)))
        self.stopped = set(getattr(self, "stopped", ())) | set(services)

    def down(self, context, services):
        self.manager.events.append(("down", tuple(services)))

    def saved_service_models(self, context, services):
        self.manager.events.append(("capture-restore", tuple(services)))
        missing = set(services) - set(context.service_models.previous)
        if missing:
            raise ContainerError("Missing restore input " + ",".join(missing))
        specifications = {name: yaml.safe_load(context.service_models.previous[name])["services"][name]
                          for name in services}
        from linktools.cntr.runtime.compose import order_service_subset
        return {name: context.service_models.previous[name]
                for name in order_service_subset(context.project_containers, specifications)}

    def apply_saved_services(self, context, services, files, *, image_ids=None):
        self.manager.events.append(("restore", tuple(services)))
        if self.restore_fails:
            raise ContainerError("restore failed")


def setup_case(tmp_path, groups, running=()):
    manager = SimpleNamespace(events=[], data_path=tmp_path, project_name="test", cache={})
    manager.logger = SimpleNamespace(warning=lambda *args: None, error=lambda *args: None,
                                     info=lambda *args: None, debug=lambda *args: None)
    manager.environ = SimpleNamespace(locks=SimpleNamespace(process_lock=lambda name: nullcontext()), debug=False)
    manager.artifact_index = Index()
    owners = [Owner(name, services, manager) for name, services in groups]
    manager.containers = {owner.name: owner for owner in owners}
    manager.integration_snapshot = {owner.name: () for owner in owners}
    manager.load_installed_config_metadata = lambda: owners
    manager.resolver = SimpleNamespace(resolve_dependencies=lambda selected: selected)
    manager.model = {"services": {name: spec for owner in owners for name, spec in owner.services.items()}}
    state = ProjectRuntimeState("test", tuple(
        SimpleNamespace(service=name, state="running", image_id="sha256:old-" + name, labels={}, health=None,
                        exit_code=None) for name in running), "docker")
    def observed_state(containers, **kwargs):
        applied = set(getattr(manager.compose_runner, "applied", ()))
        stopped = set(getattr(manager.compose_runner, "stopped", ())) - applied
        previous = tuple(SimpleNamespace(**dict(vars(item), state="exited", exit_code=0))
                         if item.service in stopped else item for item in state.services if item.service not in applied)
        return ProjectRuntimeState("test", previous + tuple(
            SimpleNamespace(service=name, state="running", health=None, exit_code=None,
                            image_id=manager.image_preparer.image_id(manager.model["services"][name]["image"]),
                            labels={"com.docker.compose.config-hash": "new-" + name}) for name in applied), "docker")
    manager.docker_inspector = SimpleNamespace(get_project_state=observed_state)
    stored = {owner.name for owner in owners if set(owner.services) & set(running)}
    manager.running_state = SimpleNamespace(
        get_persisted=lambda: sorted(stored), remove=lambda names: stored.difference_update(names),
        mark_started=lambda context: stored.update(owner.name for owner in context.target_containers),
        mark_stopped=lambda context: stored.difference_update(owner.name for owner in context.target_containers),
    )
    manager.image_preparer = SimpleNamespace(
        with_build_revisions=lambda model, containers, services: model,
        plan=lambda model, services, **kwargs: SimpleNamespace(pull=tuple(services), build=()),
        verify_builds=lambda model, services, **kwargs: None,
        image_id=lambda image: "sha256:old-" + image.split(":")[0])
    manager.image_preparer.inspect_images = lambda images: {
        image: SimpleNamespace(image_id=manager.image_preparer.image_id(image), revision=None) for image in images}
    manager.hooks = Hooks(manager.events, "manager")
    manager.lifecycle = LifecycleDispatcher(manager)
    manager.compose_runner = Runner(manager)
    manager.compose_operations = ComposeOperations(manager)
    if running:
        old = deepcopy(manager.model)
        for name in running:
            old["services"][name]["image"] = name + ":old"
        AppliedServiceModels(manager, old).record(running)
    return manager


def test_preparation_images_check_apply_and_notification_have_fixed_order(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})])
    manager.compose_operations.up(["app"])
    events = manager.events
    phases = [event[0] for event in events]
    assert phases.index("prepare") < phases.index("before-start") < phases.index("model")
    assert phases.index("model") < phases.index("images") < phases.index("check-callback")
    assert phases.index("check-callback") < phases.index("apply") < phases.index("ready") < phases.index("after-callback")
    assert manager.running_state.get_persisted() == ["app"]


def test_failed_check_does_not_stop_or_apply_restart_targets(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    manager.containers["app"].on_check = lambda context: (_ for _ in ()).throw(ContainerError("invalid candidate"))
    with pytest.raises(ContainerError, match="invalid candidate"):
        manager.compose_operations.restart(["app"])
    assert not any(event[0] in ("stop", "apply", "restore") for event in manager.events)
    assert manager.running_state.get_persisted() == ["app"]


def test_unrelated_prepare_and_check_do_not_run(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}}),
                                    ("other", {"other": {"image": "other:new"}})], running=("other",))
    manager.containers["other"].on_starting = lambda ctx: pytest.fail("unrelated preparation")
    manager.containers["other"].on_check = lambda ctx: pytest.fail("unrelated validation")
    manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]


def test_cold_second_service_failure_preserves_first_and_its_model(tmp_path):
    manager = setup_case(tmp_path, [("app", {"first": {"image": "first:new"}, "second": {"image": "second:new"}})])
    manager.compose_runner.fail = "second"
    with pytest.raises(ContainerError, match="apply failed second"):
        manager.compose_operations.up(["app"])
    assert ("stop", ("second",)) in manager.events
    assert ("stop", ("first",)) not in manager.events
    assert manager.running_state.get_persisted() == ["app"]
    models = AppliedServiceModels(manager, manager.model)
    assert "first" in models.previous and "second" not in models.previous


def test_restart_restores_failed_and_unattempted_services_not_successful_one(tmp_path):
    manager = setup_case(tmp_path, [("app", {name: {"image": name + ":new"} for name in ("first", "second", "third")})],
                         running=("first", "second", "third"))
    manager.compose_runner.fail = "second"
    with pytest.raises(ContainerError, match="apply failed second"):
        manager.compose_operations.restart(["app"])
    restored = [event[1] for event in manager.events if event[0] == "restore"]
    assert restored == [("second",), ("third",)]
    snapshots = AppliedServiceModels(manager, manager.model).previous
    assert yaml.safe_load(snapshots["first"])["services"]["first"]["image"] == "first:new"
    assert yaml.safe_load(snapshots["second"])["services"]["second"]["image"] == "second:old"


def test_recovery_failure_reports_both_errors(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    manager.compose_runner.fail = "app"
    manager.compose_runner.restore_fails = True
    with pytest.raises(ContainerError, match="Operation failed: apply failed app; recovery failed: restore failed"):
        manager.compose_operations.up(["app"])


def test_after_start_failure_is_reported_after_commit_without_rollback(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})])
    manager.containers["app"].on_started = lambda context: (_ for _ in ()).throw(RuntimeError("notification failed"))
    with pytest.raises(ContainerError, match="Services were applied; after-start callback failed: notification failed"):
        manager.compose_operations.up(["app"])
    assert not any(event[0] in ("stop", "restore") for event in manager.events)
    assert manager.running_state.get_persisted() == ["app"]
    assert "app" in AppliedServiceModels(manager, manager.model).previous


def test_down_does_not_run_preparation_checks_or_images(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    manager.containers["app"].on_starting = lambda context: pytest.fail("down prepared config")
    manager.containers["app"].on_check = lambda context: pytest.fail("down checked new config")
    manager.compose_operations.down(["app"])
    assert not any(event[0] in ("prepare", "check-callback", "images", "apply") for event in manager.events)
    assert not manager.running_state.get_persisted()


def test_missing_original_image_blocks_before_stopping(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})], running=("app",))
    manager.docker_inspector.get_project_state(None).services[0].image_id = None
    with pytest.raises(ContainerError, match="without its original image ID"):
        manager.compose_operations.restart(["app"])
    assert not any(event[0] in ("stop", "apply") for event in manager.events)


def test_container_co_selection_does_not_create_runtime_cycle(tmp_path):
    manager = setup_case(tmp_path, [("nginx", {"nginx": {"image": "nginx:new"}}),
                                    ("auth", {"auth": {"image": "auth:new", "depends_on": {"nginx": {}}}})])
    nginx, auth = manager.containers["nginx"], manager.containers["auth"]
    nginx.dependencies = ("auth",)
    auth.dependencies = ("nginx",)
    selected = manager.compose_operations.start_selection(manager.compose_operations.select(["nginx", "auth"]))
    assert selected.services == ("nginx", "auth")


def test_real_compose_cycle_is_rejected_without_bootstrap(tmp_path):
    manager = setup_case(tmp_path, [("app", {"a": {"depends_on": ["b"]}, "b": {"depends_on": ["a"]}})])
    with pytest.raises(ContainerError, match="dependency cycle"):
        manager.compose_operations.start_selection(manager.compose_operations.select(["app"]))


def test_completed_job_not_marked_as_running(tmp_path):
    manager = setup_case(tmp_path, [("job", {"job": {"image": "job:new"}})])
    manager.compose_runner.wait_service_ready = lambda *args, **kwargs: False
    manager.compose_operations.up(["job"])
    assert manager.running_state.get_persisted() == []
    assert "job" in AppliedServiceModels(manager, manager.model).previous


@pytest.mark.parametrize("state,health,code,expected", [
    ("running", None, None, True), ("running", "healthy", None, True),
    ("exited", None, 0, False), ("exited", None, 1, "error"),
    ("running", "unhealthy", None, "error"), ("exited", "healthy", 0, "error"),
])
def test_effective_health_and_job_readiness(state, health, code, expected):
    actual = ProjectRuntimeState(project="test", backend="docker", services=(SimpleNamespace(service="app", state=state, health=health, exit_code=code, labels={}),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda c, services=None: actual)))
    ctx = OperationContext(project_containers=[], target_services=("app",), compose_model={"services": {"app": {}}})
    if expected == "error":
        with pytest.raises(ContainerError, match="failed to become ready"):
            runner.wait_service_ready(ctx, "app", timeout=0)
    else:
        assert runner.wait_service_ready(ctx, "app", timeout=0) is expected


def test_explicit_completed_dependency_requires_completion():
    actual = ProjectRuntimeState(project="test", backend="docker", services=(SimpleNamespace(service="migrate", state="running", health=None, exit_code=None, labels={}),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda c, services=None: actual)))
    ctx = OperationContext(project_containers=[], target_services=("migrate", "web"), compose_model={"services": {
        "migrate": {}, "web": {"depends_on": {"migrate": {"condition": "service_completed_successfully"}}}}})
    with pytest.raises(ContainerError, match="did not complete successfully"):
        runner.wait_service_ready(ctx, "migrate", timeout=0)


def test_unselected_completed_dependency_does_not_change_selected_service_policy():
    actual = ProjectRuntimeState(project="test", backend="docker", services=(SimpleNamespace(service="app", state="running", health=None, exit_code=None, labels={}),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=lambda c, services=None: actual)))
    ctx = OperationContext(project_containers=[], target_services=("app",), compose_model={"services": {
        "app": {}, "unrelated": {"depends_on": {"app": {"condition": "service_completed_successfully"}}}}})
    assert runner.wait_service_ready(ctx, "app", timeout=0)


def test_restore_pins_original_image_and_does_not_inject_generation_label(tmp_path):
    commands = []
    documents = []

    def process(*args, **kwargs):
        commands.append(args)
        documents.append([Path(args[index + 1]).read_text() for index, value in enumerate(args[:-1]) if value == "--file"])
        return SimpleNamespace(check_call=lambda: 0)

    manager = SimpleNamespace(data_path=tmp_path, project_name="test", runtime=SimpleNamespace(create_docker_process=process))
    runner = ComposeRunner(manager)
    runner._resolved_model = lambda process: {"services": {"app": {"image": "sha256:old"}}}
    runner.wait_service_dependencies = lambda *args, **kwargs: None
    ctx = OperationContext(project_containers=[SimpleNamespace(name="app", services={"app": {}})])
    ctx.initial_runtime_state = ProjectRuntimeState("test", (
        SimpleNamespace(service="app", state="running", image_id="sha256:old", labels={}),), "docker")
    runner.apply_saved_services(ctx, ("app",), {"previous.yml": "services:\n  app:\n    image: mutable:tag\n"})
    overlay = yaml.safe_load(documents[-1][-1])
    assert overlay == {"services": {"app": {"image": "sha256:old"}}}
    assert "--force-recreate" in commands[-1]
    assert "generation" not in "\n".join(documents[-1])


def test_implicit_sidecar_does_not_promote_its_owners_group_dependencies(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new", "depends_on": ["redis"]}}),
        ("auth", {"redis": {"image": "redis:new"}, "auth": {"image": "auth:new"}}),
        ("nginx", {"nginx": {"image": "nginx:new"}}),
    ])
    manager.containers["auth"].dependencies = ("nginx",)
    selected = manager.compose_operations.start_selection(manager.compose_operations.select(["app"]))
    assert selected.services == ("app",)
    with pytest.raises(ContainerError, match="redis"):
        manager.compose_operations.up(["app"])
    assert not any(event[0] in ("images", "apply") for event in manager.events)


def test_declared_running_consumer_prepares_before_checks(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}}),
                                    ("nav", {"nav": {"image": "nav:new"}})], running=("nav",))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="nav"),)
    manager.compose_operations.up(["app"])
    assert ("prepare", "nav") not in manager.events
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]


def test_completed_service_uses_compose_conditions_without_new_task_deadline(monkeypatch):
    observations, sleeps = [], []

    def inspect(containers, services=None):
        observations.append(services)
        state = "running" if len(observations) == 1 else "exited"
        return ProjectRuntimeState("test", (SimpleNamespace(
            service="job", state=state, health=None, exit_code=0, labels={}),), "docker")

    monkeypatch.setattr("time.monotonic", lambda: pytest.fail("Completion must not add a task deadline"))
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect)))
    ctx = OperationContext(target_services=("job", "web"), compose_model={"services": {
        "job": {}, "web": {"depends_on": {"job": {"condition": "service_completed_successfully"}}}}})
    assert runner.wait_service_ready(ctx, "job") is False
    assert observations == [("job",), ("job",)]
    assert sleeps == [0.5]


def test_hook_cannot_change_operation_scope(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})])
    manager.containers["app"].on_starting = lambda context: setattr(context, "target_services", ())
    with pytest.raises(ContainerError, match="cannot change the resolved operation targets"):
        manager.compose_operations.up(["app"])
    assert not any(event[0] in ("images", "check-callback", "apply", "stop") for event in manager.events)


def test_failed_shared_file_consumer_restores_updated_running_peer(tmp_path):
    old_file, new_file = tmp_path / "old.cfg", tmp_path / "new.cfg"
    old_file.write_text("old")
    new_file.write_text("new")
    old_mount = {"type": "bind", "source": str(old_file), "target": "/shared.cfg"}
    new_mount = dict(old_mount, source=str(new_file))
    manager = setup_case(tmp_path, [("app", {
        name: {"image": name + ":new", "volumes": [new_mount]} for name in ("first", "second")
    })], running=("first", "second"))
    original = deepcopy(manager.model)
    for name in original["services"]:
        original["services"][name]["volumes"] = [old_mount]
    AppliedServiceModels(manager, original).record(("first", "second"))
    manager.compose_runner.fail = "second"
    with pytest.raises(ContainerError, match="apply failed second"):
        manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("first",), ("second",)]
    saved = AppliedServiceModels(manager, manager.model).previous
    for name in saved:
        assert yaml.safe_load(saved[name])["services"][name]["volumes"][0]["source"] == str(old_file)


def test_shared_changed_input_recovery_follows_transitive_consumers(tmp_path):
    old_x, old_y, new_x, new_y = [tmp_path / name for name in ("oldx", "oldy", "newx", "newy")]
    for path in (old_x, old_y, new_x, new_y):
        path.write_text(path.name)
    def spec(*paths):
        return {"image": "app:new", "volumes": [
            {"type": "bind", "source": str(path), "target": "/" + path.name[-1:]} for path in paths]}
    manager = setup_case(tmp_path, [("app", {"a": spec(new_x), "b": spec(new_x, new_y), "c": spec(new_y)})],
                         running=("a", "b", "c"))
    old = {"services": {"a": spec(old_x), "b": spec(old_x, old_y), "c": spec(old_y)}}
    AppliedServiceModels(manager, old).record(("a", "b", "c"))
    manager.compose_runner.fail = "c"
    with pytest.raises(ContainerError, match="apply failed c"):
        manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("a",), ("b",), ("c",)]


def test_cold_shared_input_failure_stops_its_new_peer(tmp_path):
    common = tmp_path / "common"
    common.write_text("new")
    spec = {"image": "app:new", "volumes": [{"type": "bind", "source": str(common), "target": "/common"}]}
    manager = setup_case(tmp_path, [("app", {"a": deepcopy(spec), "b": deepcopy(spec)})])
    manager.compose_runner.fail = "b"
    with pytest.raises(ContainerError, match="apply failed b"):
        manager.compose_operations.up(["app"])
    assert ("stop", ("b", "a")) in manager.events
    assert not manager.running_state.get_persisted()
    assert not AppliedServiceModels(manager, manager.model).previous


def test_plan_and_execution_share_project_and_file_argument_builder(tmp_path):
    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="project"))
    assert runner.compose_args(("logical.yml",)) == [
        "compose", "--project-directory", str(tmp_path / "compose"),
        "--project-name", "project", "--file", "logical.yml"]
    with runner._saved_compose_args(OperationContext(), ("services: {}\n",)) as args:
        assert args[:5] == runner.compose_args(("logical.yml",))[:5]
        actual = Path(args[-1])
        assert actual.read_text() == "services: {}\n"
    assert not actual.exists()
    assert not (tmp_path / "compose").exists()


def test_stop_uses_prepared_model_without_serializing_live_template_files(tmp_path):
    documents = []
    def process(*args, **kwargs):
        documents.append([Path(args[index + 1]).read_text() for index, value in enumerate(args[:-1]) if value == "--file"])
        return SimpleNamespace(check_call=lambda: 0)
    runner = ComposeRunner(SimpleNamespace(data_path=tmp_path, project_name="test", runtime=SimpleNamespace(create_docker_process=process)))
    ctx = OperationContext(compose_model={"services": {"app": {"image": "test:target"}}})
    runner.final_model = lambda context: pytest.fail("stop rendered a different model")
    runner.stop(ctx, ("app",))
    assert yaml.safe_load(documents[0][0]) == ctx.compose_model


def test_restart_true_propagates_only_after_actual_provider_change(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:local"}}),
        ("web", {"web": {"image": "web:local",
                         "depends_on": {"db": {"restart": True}}}}),
    ], running=("db", "web"))
    # A matching applied model and image must not trigger a dependent restart.
    AppliedServiceModels(manager, manager.model).record(("db", "web"))
    manager.compose_operations.up()
    assert not any(event[0] == "restart" for event in manager.events)
    manager.events.clear()
    manager.model["services"]["db"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up()
    assert ("restart", "web") in manager.events


def test_plain_dependency_does_not_restart_running_peer(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:local"}}),
        ("web", {"web": {"image": "web:local", "depends_on": ["db"]}}),
    ], running=("db", "web"))
    manager.model["services"]["db"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up(["db"])
    assert ("restart", "web") not in manager.events


def test_restart_dependency_transitive_once(tmp_path):
    manager = setup_case(tmp_path, [
        ("a", {"a": {"image": "a:local"}}),
        ("b", {"b": {"image": "b:local", "depends_on": {"a": {"restart": True}}}}),
        ("c", {"c": {"image": "c:local", "depends_on": {"a": {"restart": True},
                                                     "b": {"restart": True}}}}),
    ], running=("a", "b", "c"))
    AppliedServiceModels(manager, manager.model).record(("a", "b", "c"))
    manager.model["services"]["a"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up()
    assert [event[1] for event in manager.events if event[0] == "restart"] == ["b", "c"]


def test_recreated_namespace_provider_recreates_existing_dependent(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:local"}}),
        ("net", {"net": {"image": "net:local", "network_mode": "service:db"}}),
    ], running=("db", "net"))
    manager.model["services"]["db"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up(["db", "net"])
    assert ("apply", "net", True) in manager.events
    assert ("restart", "net") not in manager.events


def test_stopped_restart_only_dependent_not_started(tmp_path):
    manager = setup_case(tmp_path, [
        ("db", {"db": {"image": "db:local"}}),
        ("web", {"web": {"image": "web:local",
                         "depends_on": {"db": {"restart": True}}}}),
    ], running=("db",))
    manager.model["services"]["db"]["environment"] = {"VERSION": "2"}
    manager.compose_operations.up(["db"])
    assert ("restart", "web") not in manager.events


@pytest.mark.parametrize("action", ["up", "restart", "down", "render"])
def test_execution_prepares_each_installed_owner_before_selected_operation(tmp_path, action):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}}),
                                    ("other", {"other": {"image": "other:new"}})])
    for owner in manager.containers.values():
        owner.on_prepare = lambda owner=owner: manager.events.append(("load-prepare", owner.name))
    manager.compose_runner.config = lambda *args, **kwargs: manager.events.append(("config",))
    manager.compose_runner.down = lambda *args, **kwargs: manager.events.append(("down",))
    getattr(manager.compose_operations, action)(["app"])
    assert manager.events[0] == ("load-prepare", "app")
    assert len([event for event in manager.events if event[0] == "load-prepare"]) == (
        1 if action in ("up", "restart") else 2)
    assert len([event for event in manager.events if event == ("prepare", "app")]) == (
        1 if action in ("up", "restart") else 0)
