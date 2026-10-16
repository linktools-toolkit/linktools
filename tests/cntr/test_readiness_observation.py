#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readiness batches observations only within an unchanged waiting phase."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from linktools.cntr.errors import ContainerError
from linktools.cntr.artifacts import CONFIG_REVISION_LABEL
from linktools.cntr.runtime.inspect import ProjectRuntimeState
from linktools.cntr.runtime.compose import ComposeRunner
from linktools.cntr.runtime.inspect import RuntimeInspectionOutputError


def item(name, state="running", health="healthy", exit_code=0, labels=None):
    return SimpleNamespace(service=name, state=state, health=health, exit_code=exit_code,
                           labels=labels or {}, image_id="sha256:app")


def context(dependencies):
    return SimpleNamespace(project_containers=(), target_services=("app", *dependencies),
        is_full_project=False, compose_model={"services": {
            "app": {"depends_on": dependencies}, **{name: {} for name in dependencies}}})


@pytest.mark.parametrize("condition", ["service_healthy", "service_completed_successfully", "service_started"])
def test_ready_dependency_fanout_uses_one_snapshot(condition):
    names = tuple("dep{}".format(i) for i in range(20))
    ctx = context({name: {"condition": condition} for name in names})
    calls = []
    state = "exited" if condition == "service_completed_successfully" else "running"

    def inspect(containers, services=None):
        calls.append(services)
        return ProjectRuntimeState(project="test", backend="docker", services=tuple(item(name, state=state) for name in names))

    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect)))
    runner.wait_service_dependencies(ctx, "app", model=ctx.compose_model)
    assert calls == [names]


def test_poll_refreshes_all_dependencies_without_reusing_previous_health(monkeypatch):
    ctx = context({name: {"condition": "service_healthy"} for name in ("first", "second")})
    observed, sleeps = [], []

    def inspect(containers, services=None):
        observed.append(services)
        if len(observed) == 1:
            return ProjectRuntimeState(project="test", backend="docker", services=(item("first", health="starting"), item("second")))
        return ProjectRuntimeState(project="test", backend="docker", services=(item("first"), item("second", health="unhealthy")))

    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect)))
    with pytest.raises(ContainerError, match="second is unhealthy"):
        runner.wait_service_dependencies(ctx, "app")
    assert observed == [("first", "second"), ("first", "second")]
    assert sleeps == [0.5]


@pytest.mark.parametrize("selected", [False, True])
def test_optional_dependency_never_masks_corrupt_inspection(selected):
    ctx = context({"dep": {"condition": "service_healthy", "required": False}})
    ctx.target_services = ("app", "dep") if selected else ("app",)
    calls = []

    def inspect(containers, services=None):
        calls.append(services)
        raise RuntimeInspectionOutputError("invalid fixture output")

    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect)))
    with pytest.raises(RuntimeInspectionOutputError, match="invalid fixture output"):
        runner.wait_service_dependencies(ctx, "app")
    assert calls == [("dep",)]


def test_up_and_next_wait_phase_cannot_reuse_previous_observation():
    ctx = context({"dep": {"condition": "service_healthy"}})
    calls = []

    def inspect(containers, services=None):
        calls.append(("inspect", services))
        return ProjectRuntimeState(project="test", backend="docker", services=(item("dep"), item("app")))

    process = SimpleNamespace(check_call=lambda: calls.append(("up",)))
    runner = ComposeRunner(SimpleNamespace(
        docker_inspector=SimpleNamespace(get_project_state=inspect),
        runtime=SimpleNamespace(create_docker_process=lambda *a, **kw: process)))
    runner._model_args = lambda context: nullcontext(["compose"])
    for _ in range(2):
        runner.apply_service(ctx, "app")
        assert runner.wait_service_ready(ctx, "app")
    assert calls == [("inspect", ("dep",)), ("up",), ("inspect", ("app",))] * 2


@pytest.mark.parametrize("method", ["wait_service_ready", "wait_service_healthy", "wait_service_completed"])
def test_single_service_wait_observes_every_replica(method, monkeypatch):
    ctx = context({})
    calls, sleeps = [], []

    def inspect(containers, services=None):
        calls.append(services)
        if method == "wait_service_completed":
            replicas = (item("app", state="exited"),
                        item("app", state="running" if len(calls) == 1 else "exited"))
        else:
            replicas = (item("app"), item("app", health="starting" if len(calls) == 1 else "healthy"))
        return ProjectRuntimeState(project="test", backend="docker", services=replicas)

    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(get_project_state=inspect)))
    getattr(runner, method)(ctx, "app", timeout=None)
    assert calls == [("app",), ("app",)]
    assert sleeps == [0.5]


@pytest.mark.parametrize("expected", [None, "new"])
@pytest.mark.parametrize("observed", [None, "old", "new"])
def test_batch_readiness_requires_revision_on_every_regular_replica(expected, observed, monkeypatch):
    ctx = context({})
    ctx.compose_model["services"]["app"] = {
        "scale": 2, "labels": {} if expected is None else {CONFIG_REVISION_LABEL: expected}}
    actual = ProjectRuntimeState(project="test", backend="docker", services=(
        item("app", labels={CONFIG_REVISION_LABEL: "new"}),
        item("app", labels={} if observed is None else {CONFIG_REVISION_LABEL: observed}),
        item("app", labels={"com.docker.compose.oneoff": "True", CONFIG_REVISION_LABEL: "old"})))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: actual)))
    monkeypatch.setattr("time.sleep", lambda delay: pytest.fail("Revision mismatch must not poll"))
    images = {"app": "sha256:app"}
    ready = expected is None or observed == expected
    assert runner.ready_services(ctx, ("app",), actual, images) == ({"app"} if ready else set())
    if ready:
        assert runner.wait_services_ready(ctx, ("app",), images) is actual
    else:
        with pytest.raises(ContainerError, match="desired configuration revision"):
            runner.wait_services_ready(ctx, ("app",), images)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("observed", [None, "old", "new"])
def test_recovery_readiness_uses_saved_model_revision_for_every_replica(completed, observed, monkeypatch):
    ctx = context({})
    ctx.compose_model["services"]["app"]["labels"] = {CONFIG_REVISION_LABEL: "new"}
    saved = {"services": {"app": {"labels": {CONFIG_REVISION_LABEL: "old"}}}}
    if completed:
        saved["services"]["caller"] = {"depends_on": {"app": {"condition": "service_completed_successfully"}}}
        ctx.target_services = ("app", "caller")
    state = "exited" if completed else "running"
    actual = ProjectRuntimeState(project="test", backend="docker", services=(
        item("app", state=state, labels={CONFIG_REVISION_LABEL: "old"}),
        item("app", state=state, labels={} if observed is None else {CONFIG_REVISION_LABEL: observed}),
        item("app", labels={"com.docker.compose.oneoff": "True", CONFIG_REVISION_LABEL: "new"})))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: actual)))
    monkeypatch.setattr("time.sleep", lambda delay: pytest.fail("Revision mismatch must not poll"))
    if observed == "old":
        assert runner.wait_service_ready(ctx, "app", model=saved) is (not completed)
    else:
        with pytest.raises(ContainerError, match="desired configuration revision"):
            runner.wait_service_ready(ctx, "app", model=saved)


def test_single_service_without_expected_revision_accepts_existing_label():
    ctx = context({})
    actual = ProjectRuntimeState(project="test", backend="docker", services=(
        item("app", labels={CONFIG_REVISION_LABEL: "existing"}),))
    runner = ComposeRunner(SimpleNamespace(docker_inspector=SimpleNamespace(
        get_project_state=lambda containers, services=None: actual)))
    assert runner.wait_service_ready(ctx, "app")
