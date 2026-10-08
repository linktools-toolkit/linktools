#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle plans describe exactly the registered hooks dispatch will visit."""
import pytest

from linktools.cntr.context import EventContext
from linktools.cntr.lifecycle import HookCycleError, HookPhase, HookRegistry, HookValidationError


class _Container:
    def __init__(self, name, events, dependencies=(), order=500):
        self.name = name
        self.events = events
        self.dependencies = dependencies
        self.order = order
        self.services = {name: {}}
        self.integrations = {}
        self.hooks = HookRegistry(owner=self, scope="container")

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
def lifecycle_case(fresh_manager, monkeypatch):
    events = []
    monkeypatch.setattr(fresh_manager, "generated_configs", {})
    first = _Container("first", events, order=900)
    second = _Container("second", events, dependencies=("first",), order=100)
    monkeypatch.setattr(fresh_manager, "integration_snapshot", {"first": {}, "second": {}})
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
    monkeypatch.setattr("linktools.cntr.execution.planner.collect_candidates", lambda *args: {})
    context = EventContext()
    context.target_containers = [first, second]
    return fresh_manager, (first, second), context, events


def _register(registry, phase, events, owner, key, **kwargs):
    def callback(context=None):
        events.append((phase.value, owner, key))
    registry.register(phase, callback, key=key, name=key, **kwargs)


def _execute(manager, context, action):
    if action in ("restart", "down"):
        with manager.lifecycle.notify_stop(context):
            pass
    if action in ("restart", "up"):
        with manager.lifecycle.notify_start(context):
            pass


def _expected_hooks(action):
    start = [
        ("check", "first", "a"), ("check", "first", "b"),
        ("check", "second", "a"), ("check", "second", "b"),
        ("before-start", "first", "a"), ("before-start", "first", "b"),
        ("before-start", "second", "a"), ("before-start", "second", "b"),
        ("before-start", None, "a"), ("before-start", None, "b"),
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
    return {"up": start, "restart": stop + start, "down": stop}[action]


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
        events.append(("runtime", None, "up"))

    assert events == [
        ("callback", "first", "check"), ("callback", "second", "check"),
        ("callback", "first", "starting"), ("callback", "second", "starting"),
        ("before-start", "first", "a"), ("before-start", "first", "b"),
        ("runtime", None, "up"),
        ("callback", "second", "started"), ("callback", "first", "started"),
    ]


@pytest.mark.parametrize("reassign_at", ["on_check", "check_hook", "on_starting"])
def test_start_phases_reread_reassigned_targets(lifecycle_case, reassign_at):
    manager, containers, context, events = lifecycle_case
    first, second = containers

    def reassign(context):
        events.append(("reassign", "first", reassign_at))
        context.target_containers = [second]

    if reassign_at == "check_hook":
        first.hooks.register(HookPhase.CHECK, reassign)
    else:
        setattr(first, reassign_at, reassign)

    def removed_target_hook():
        raise AssertionError("Removed target must not run before-start hooks")

    first.hooks.register(HookPhase.BEFORE_START, removed_target_hook)
    _register(second.hooks, HookPhase.BEFORE_START, events, "second", "remaining")
    _register(manager.hooks, HookPhase.BEFORE_START, events, None, "manager")

    with manager.lifecycle.notify_start(context):
        events.append(("runtime", None, "up"))

    check_events = [("callback", "first", "check"), ("callback", "second", "check")]
    starting_events = [("callback", "second", "starting")]
    if reassign_at == "on_check":
        check_events[0] = ("reassign", "first", reassign_at)
    elif reassign_at == "check_hook":
        check_events.insert(1, ("reassign", "first", reassign_at))
    else:
        starting_events.insert(0, ("reassign", "first", reassign_at))
    assert events == check_events + starting_events + [
        ("before-start", "second", "remaining"),
        ("before-start", None, "manager"),
        ("runtime", None, "up"),
        ("callback", "second", "started"),
    ]
