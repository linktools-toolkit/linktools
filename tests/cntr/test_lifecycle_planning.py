#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle plans describe exactly the registered hooks dispatch will visit."""
import pytest

from linktools.cntr.container import BaseContainer
from linktools.cntr.context import OperationContext
from linktools.cntr.lifecycle import HookCycleError, HookPhase, HookRegistry, HookValidationError


class _Container(BaseContainer):

    def __init__(self, manager, root_path, name, events, dependencies=(), order=500):
        super().__init__(manager, root_path, name=name)
        self.events = events
        self._dependencies = dependencies
        self._order = order
        self.services = {name: {}}
        self.hooks = HookRegistry(owner=self, scope="container")

    @property
    def dependencies(self):
        return self._dependencies

    def on_check(self, context):
        self.events.append(("callback", self.name, "check"))

    def on_starting(self):
        self.events.append(("callback", self.name, "starting"))

    def on_started(self, context):
        self.events.append(("callback", self.name, "started"))

    def on_stopping(self):
        self.events.append(("callback", self.name, "stopping"))

    def on_stopped(self, context):
        self.events.append(("callback", self.name, "stopped"))


@pytest.fixture
def lifecycle_case(fresh_manager, monkeypatch, tmp_path):
    events = []
    first = _Container(fresh_manager, tmp_path, "first", events, order=900)
    second = _Container(fresh_manager, tmp_path, "second", events, dependencies=("first",), order=100)
    monkeypatch.setattr(fresh_manager, "integration_snapshot", {"first": (), "second": ()})
    monkeypatch.setattr(fresh_manager, "containers", {c.name: c for c in (second, first)})
    monkeypatch.setattr(fresh_manager, "hooks", HookRegistry(owner=fresh_manager, scope="manager"))
    monkeypatch.setattr(
        fresh_manager, "load_installed_config_metadata",
        lambda: fresh_manager.resolver.resolve_dependencies((second, first)),
    )

    def fail(*args, **kwargs):
        raise AssertionError("Planning must not prepare, run Docker, or write state")

    monkeypatch.setattr(fresh_manager, "prepare_installed_containers", fail)
    monkeypatch.setattr(fresh_manager.runtime, "create_process", fail)
    monkeypatch.setattr(fresh_manager.running_state, "mark_started", fail)
    monkeypatch.setattr(fresh_manager.running_state, "mark_stopped", fail)
    monkeypatch.setattr(fresh_manager.artifact_index, "record", fail)
    monkeypatch.setattr("linktools.cntr.execution.planner.collect_candidates",
                        lambda manager, containers: {
                            str(tmp_path / (container.name + ".yml")): (
                                "compose", container.name,
                                "services:\\n  {}:\\n    image: {}:latest\\n".format(container.name, container.name),
                            ) for container in containers})
    monkeypatch.setattr(fresh_manager.docker_inspector, "preflight_candidates",
                        lambda values: "skipped")
    context = OperationContext()
    context.target_containers = [first, second]
    return fresh_manager, (first, second), context, events


def _register(registry, phase, events, owner, key, **kwargs):
    def callback(context=None):
        events.append((phase.value, owner, key))
    registry.register(phase, callback, key=key, name=key, **kwargs)


def _execute(manager, context, action):
    if action == "down":
        with manager.lifecycle.notify_stop(context):
            pass
    else:
        with manager.lifecycle.notify_start(context):
            manager.lifecycle.check(context)
            if action == "restart":
                with manager.lifecycle.notify_stop(context):
                    pass


def _expected_hooks(action):
    start = [
        ("before-start", "first", "a"), ("before-start", "first", "b"),
        ("before-start", "second", "a"), ("before-start", "second", "b"),
        ("before-start", None, "a"), ("before-start", None, "b"),
        ("check", "first", "a"), ("check", "first", "b"),
        ("check", "second", "a"), ("check", "second", "b"),
        ("after-start", "second", "b"), ("after-start", "second", "a"),
        ("after-start", "first", "b"), ("after-start", "first", "a"),
    ]
    stop = [
        ("before-stop", "second", "b"), ("before-stop", "second", "a"),
        ("before-stop", "first", "b"), ("before-stop", "first", "a"),
        ("before-stop", None, "a"), ("before-stop", None, "b"),
        ("after-stop", "first", "a"), ("after-stop", "first", "b"),
        ("after-stop", "second", "a"), ("after-stop", "second", "b"),
        ("after-stop", None, "a"), ("after-stop", None, "b"),
    ]
    return {"up": start, "restart": start[:-4] + stop + start[-4:], "down": stop}[action]


@pytest.mark.parametrize("action", ["up", "restart", "down"])
def test_plan_matches_dependency_ordered_dispatch_and_supported_manager_phases(lifecycle_case, action):
    manager, containers, context, events = lifecycle_case
    for owner, registry in [(c.name, c.hooks) for c in containers] + [(None, manager.hooks)]:
        for phase in HookPhase:
            # Dependency order wins over registration and numeric order.
            _register(registry, phase, events, owner, "b", order=1, after=("a",))
            _register(registry, phase, events, owner, "a", order=999, opaque=True)

    plan = manager.planner.plan(action)
    assert plan.resolved_containers == ("first", "second")
    assert events == []
    _execute(manager, context, action)
    actual = [event for event in events if event[0] != "callback"]
    assert actual == _expected_hooks(action)
    assert [(h.phase, h.container, h.name) for h in plan.hooks] == actual
    assert all(h.opaque == (h.name == "a") for h in plan.hooks)


@pytest.mark.parametrize("action", ["up", "restart", "down"])
@pytest.mark.parametrize("phase", [HookPhase.CHECK, HookPhase.AFTER_START,
                                   HookPhase.AFTER_COMPOSE_RENDER, HookPhase.AFTER_REMOVE])
def test_unused_invalid_manager_phase_does_not_block_plan_or_dispatch(lifecycle_case, action, phase):
    manager, _, context, events = lifecycle_case
    _register(manager.hooks, phase, events, None, "unused", after=("missing",))

    plan = manager.planner.plan(action)
    assert plan.hooks == ()
    assert events == []
    _execute(manager, context, action)
    assert all(event[0] == "callback" for event in events)


@pytest.mark.parametrize("owner", ["container", "manager"])
@pytest.mark.parametrize("phase,action", [(HookPhase.BEFORE_START, "up"),
                                         (HookPhase.BEFORE_STOP, "restart"),
                                         (HookPhase.AFTER_STOP, "down")])
@pytest.mark.parametrize("constraint", ["before", "after", "cycle"])
def test_used_invalid_hooks_fail_planning_without_invocation(lifecycle_case, owner, phase, action, constraint):
    manager, containers, _, events = lifecycle_case
    registry = containers[0].hooks if owner == "container" else manager.hooks
    if constraint == "cycle":
        _register(registry, phase, events, owner, "a", after=("b",))
        _register(registry, phase, events, owner, "b", after=("a",))
        error = HookCycleError
    else:
        _register(registry, phase, events, owner, "a", **{constraint: ("missing",)})
        error = HookValidationError
    with pytest.raises(error):
        manager.planner.plan(action)
    assert events == []


@pytest.mark.parametrize("action,phase", [("up", HookPhase.BEFORE_START),
                                         ("restart", HookPhase.BEFORE_STOP),
                                         ("down", HookPhase.AFTER_STOP)])
@pytest.mark.parametrize("constraint", ["optional_before", "optional_after"])
def test_optional_missing_hook_dependency_remains_allowed(lifecycle_case, action, phase, constraint):
    manager, containers, context, events = lifecycle_case
    _register(containers[0].hooks, phase, events, "first", "optional", **{constraint: ("missing",)})
    plan = manager.planner.plan(action)
    assert [(h.phase, h.container, h.name) for h in plan.hooks] == [(phase.value, "first", "optional")]
    assert events == []
    _execute(manager, context, action)
    assert (phase.value, "first", "optional") in events


def test_starting_callbacks_all_finish_before_start_registry_lookup(lifecycle_case):
    manager, containers, context, events = lifecycle_case
    first, second = containers
    _register(first.hooks, HookPhase.BEFORE_START, events, "first", "b", after=("a",))

    def on_starting():
        events.append(("callback", "second", "starting"))
        _register(first.hooks, HookPhase.BEFORE_START, events, "first", "a")

    second.on_starting = on_starting
    with manager.lifecycle.notify_start(context):
        manager.lifecycle.check(context)
        events.append(("runtime", None, "up"))

    assert events == [
        ("callback", "first", "starting"), ("callback", "second", "starting"),
        ("before-start", "first", "a"), ("before-start", "first", "b"),
        ("callback", "first", "check"), ("callback", "second", "check"),
        ("runtime", None, "up"),
        ("callback", "second", "started"), ("callback", "first", "started"),
    ]


def test_partial_restart_starts_runtime_provider_without_stopping_it(lifecycle_case, monkeypatch) -> None:
    manager, containers, _, events = lifecycle_case
    provider, target = containers
    target._dependencies = ()
    monkeypatch.setattr(
        provider, "get_runtime_requirements",
        lambda names: {provider.name: tuple(provider.services)} if target.name in names else {},
    )
    for container in containers:
        for phase in (HookPhase.CHECK, HookPhase.BEFORE_START, HookPhase.AFTER_START,
                      HookPhase.BEFORE_STOP, HookPhase.AFTER_STOP):
            _register(container.hooks, phase, events, container.name, phase.value)

    plan = manager.planner.plan("restart", names=[target.name])
    assert events == []
    planned = [(hook.phase, hook.container, hook.name) for hook in plan.hooks]
    provider_phases = [phase for phase, name, _ in planned if name == provider.name]
    assert provider_phases == ["before-start", "check", "after-start"]
    assert [(phase, name) for phase, name, _ in planned if phase.endswith("stop")] == [
        ("before-stop", target.name), ("after-stop", target.name),
    ]

    selection = manager.compose_operations.select([target.name], metadata_only=True, for_start=True)
    start_selection = manager.compose_operations.start_selection(selection)
    start_context = OperationContext()
    start_context.target_containers = start_selection.target_containers
    stop_context = OperationContext()
    stop_context.target_containers = selection.target_containers
    with manager.lifecycle.notify_start(start_context):
        manager.lifecycle.check(start_context)
        with manager.lifecycle.notify_stop(stop_context):
            events.append(("runtime", None, "stop"))
        events.append(("runtime", None, "up"))

    assert [event for event in events if event[0] not in ("callback", "runtime")] == planned
    assert ("callback", provider.name, "stopping") not in events
    assert ("callback", provider.name, "stopped") not in events
    before_stop = next(index for index, event in enumerate(events) if event[0] == "before-stop")
    assert all(index < before_stop for index, event in enumerate(events)
               if event[0] in ("check", "before-start"))
