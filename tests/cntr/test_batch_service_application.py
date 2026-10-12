#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native batch scope and authoritative per-service completion evidence."""
from contextlib import nullcontext
from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest

from linktools.cntr.errors import ContainerError
from linktools.cntr.artifacts import AppliedServiceModels
from linktools.cntr.runtime.compose import ComposeRunner
from linktools.cntr.runtime.inspect import ProjectRuntimeState
from test_lifecycle_rebuild import setup_case


def instance(service, image="sha256:target", state="running", health=None, code=None, config="new"):
    return SimpleNamespace(service=service, image_id=image, state=state, health=health, exit_code=code,
                           labels={"com.docker.compose.config-hash": config})


def readiness_case(specifications, snapshots):
    queries = []
    def inspect(containers, services=None):
        queries.append(tuple(services))
        return ProjectRuntimeState("test", tuple(snapshots.pop(0)), "docker")
    manager = SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect))
    context = SimpleNamespace(project_containers=(), compose_model={"services": specifications})
    return ComposeRunner(manager), context, queries


def test_native_batch_keeps_selected_edges_without_python_service_loop():
    calls = []
    def process(*args):
        calls.append(args)
        return SimpleNamespace(check_call=lambda: 0)
    runner = ComposeRunner(SimpleNamespace(runtime=SimpleNamespace(create_docker_process=process)))
    context = SimpleNamespace(is_full_project=False, compose_model={"services": {
        "db": {}, "app": {"depends_on": {"db": {"condition": "service_healthy"}}}}})
    runner._model_args = lambda ctx: nullcontext(["compose"])
    runner.apply_services(context, ("db", "app"))
    assert calls == [("compose", "up", "--detach", "--no-build", "--pull", "never", "--no-deps", "db", "app")]
    assert context.compose_model["services"]["app"]["depends_on"]["db"]["condition"] == "service_healthy"


@pytest.mark.parametrize("observed", [[], [instance("app")],
    [instance("app"), instance("app", image="sha256:old")],
    [instance("app"), instance("app", health="unhealthy")]])
def test_batch_rejects_incomplete_replicas_wrong_images_and_unhealthy_instances(observed):
    runner, context, queries = readiness_case({"app": {"deploy": {"replicas": 2}}}, [observed])
    with pytest.raises(ContainerError, match="Service app did not reach"):
        runner.wait_services_ready(context, ("app",), {"app": "sha256:target"})
    assert len(queries) == 1


@pytest.mark.parametrize("health", [None, "unhealthy"])
def test_batch_waits_for_remaining_job_replicas_without_rechecking_completed_ones_individually(monkeypatch, health):
    specs = {"job": {"scale": 2}, "app": {"depends_on": {
        "job": {"condition": "service_completed_successfully"}}}}
    snapshots = [[instance("job", state="exited", code=0, health=health), instance("job", health=health), instance("app")],
                 [instance("job", state="exited", code=0, health=health),
                  instance("job", state="exited", code=0, health=health), instance("app")]]
    runner, context, queries = readiness_case(specs, snapshots)
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    result = runner.wait_services_ready(context, ("job", "app"), {"job": "sha256:target", "app": "sha256:target"})
    assert result.running_services == {"app"}
    assert queries == [("job", "app"), ("job", "app")]
    assert sleeps == [0.5]


def test_zero_replicas_requires_observed_removal():
    runner, context, _ = readiness_case({"app": {"scale": 0}}, [[]])
    assert runner.wait_services_ready(context, ("app",), {"app": "sha256:target"}).services == ()
    state = ProjectRuntimeState("test", (instance("app"),), "docker")
    assert runner.ready_services(context, ("app",), state, {"app": "sha256:target"}) == set()


def test_partial_batch_failure_preserves_successful_sibling_but_restores_mixed_config_replicas(tmp_path):
    manager = setup_case(tmp_path, [(name, {name: {"image": name + ":current", "scale": replicas,
                                                  "environment": {"VERSION": "new"}}})
                                    for name, replicas in (("good", 1), ("mixed", 3))], running=("good", "mixed"))
    old = deepcopy(manager.model)
    for spec in old["services"].values():
        spec["environment"]["VERSION"] = "old"
    AppliedServiceModels(manager, old).record(("good", "mixed"))
    original = [instance("good", image="sha256:old-good", config="old"),
                *[instance("mixed", image="sha256:old-mixed", config="old") for _ in range(3)]]
    current = list(original)
    manager.docker_inspector.get_project_state = lambda *args, **kwargs: ProjectRuntimeState("test", tuple(current), "docker")
    runner = manager.compose_runner
    def apply(context, services):
        current[:] = [instance("good", image="sha256:old-good"),
                      instance("mixed", image="sha256:old-mixed"),
                      instance("mixed", image="sha256:old-mixed"),
                      instance("mixed", image="sha256:old-mixed", config="old")]
        raise ContainerError("native partial failure")
    runner.apply_services = apply
    runner.ready_services = MethodType(ComposeRunner.ready_services, runner)
    runner._completed_services = MethodType(ComposeRunner._completed_services, runner)
    with pytest.raises(ContainerError, match="native partial failure"):
        manager.compose_operations.up()
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("mixed",)]
    receipts = AppliedServiceModels(manager, manager.model)
    assert receipts.previous_model("good")["services"]["good"]["environment"]["VERSION"] == "new"
    assert receipts.previous_model("mixed")["services"]["mixed"]["environment"]["VERSION"] == "old"


def test_maintenance_oneoff_does_not_count_as_a_service_replica():
    maintenance = instance("app", state="exited", code=0)
    maintenance.labels["com.docker.compose.oneoff"] = "True"
    runner, context, queries = readiness_case({"app": {}}, [[instance("app"), maintenance]])
    runner.wait_services_ready(context, ("app",), {"app": "sha256:target"})
    assert len(queries) == 1
    assert runner.ready_services(context, ("app",), ProjectRuntimeState("test", (maintenance,), "docker"),
                                 {"app": "sha256:target"}) == set()


def test_failed_batch_does_not_commit_or_rerun_an_untouched_previously_completed_job(tmp_path):
    manager = setup_case(tmp_path, [("job", {"job": {
        "image": "job:current", "environment": {"VERSION": "new"}}})])
    previous = deepcopy(manager.model)
    previous["services"]["job"]["environment"]["VERSION"] = "old"
    AppliedServiceModels(manager, previous).record(("job",))
    stopped = instance("job", image="sha256:old-job", state="exited", code=0, config="old")
    manager.docker_inspector.get_project_state = lambda *args, **kwargs: ProjectRuntimeState(
        "test", (stopped,), "docker")

    def fail_before_application(context, services):
        raise ContainerError("native failure before job application")

    runner = manager.compose_runner
    runner.apply_services = fail_before_application
    runner.ready_services = ComposeRunner(manager).ready_services
    with pytest.raises(ContainerError, match="native failure before job application"):
        manager.compose_operations.up()
    receipts = AppliedServiceModels(manager, manager.model)
    assert receipts.previous_model("job")["services"]["job"]["environment"]["VERSION"] == "old"
    assert not any(event[0] in ("apply", "restore", "restart") for event in manager.events)
    assert manager.running_state.get_persisted() == []
