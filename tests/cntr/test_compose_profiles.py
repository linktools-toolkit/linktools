#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Compose remains authoritative for profile activation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from linktools.cntr._operations import ComposeOperations, ComposeSelection
from linktools.cntr.runtime.compose import ComposeRunner
from test_service_application_order import Container


def test_full_selection_omits_disabled_owner_and_its_hooks():
    app = Container("app", {"app": {"image": "app"}})
    optional = Container("optional", {"optional": {"image": "optional", "profiles": ["debug"]}})
    model = {"services": {"app": app.services["app"]}}
    calls = []
    manager = SimpleNamespace(compose_runner=SimpleNamespace(
        final_model=lambda context, **kwargs: calls.append(context) or model))
    result = ComposeOperations(manager).start_selection(ComposeSelection((app, optional), (app, optional), (), True))
    assert result.services == ("app",)
    assert result.target_containers == (app,)
    assert len(calls) == 1 and calls[0].is_full_project


def test_native_enabled_profile_is_retained():
    app = Container("app", {"app": {"profiles": ["debug"]}})
    model = {"services": dict(app.services)}
    manager = SimpleNamespace(compose_runner=SimpleNamespace(final_model=lambda context, **kwargs: model))
    selection = ComposeSelection((app,), (app,), ("app",), False)
    assert ComposeOperations(manager).start_selection(selection).services == ("app",)


@pytest.mark.parametrize("full,services", [(True, []), (False, ["app"])])
def test_native_model_activates_explicit_profiles_without_filtering_other_services(monkeypatch, full, services):
    app = Container("app", {"app": {"profiles": ["debug"]}})
    calls = []
    manager = SimpleNamespace(runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: calls.append(args) or object()))
    runner = ComposeRunner(manager)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {
        "app.yml": ("compose", "app", "services: {}")})
    runner._saved_compose_args = lambda *args: nullcontext(["compose"])
    runner._resolved_model = lambda process: {"services": {}}
    context = SimpleNamespace(project_containers=(app,), target_services=("app",), is_full_project=full)
    runner.final_model(context)
    assert calls == [("compose", "config", "--format", "json", *services)]


def test_disabled_profile_cycle_does_not_preempt_native_selection():
    app = Container("app", {"app": {"image": "app"}})
    optional = Container("optional", {"optional": {"profiles": ["debug"], "depends_on": ["optional"]}})
    manager = SimpleNamespace(compose_runner=SimpleNamespace(
        final_model=lambda context, **kwargs: {"services": dict(app.services)}))
    result = ComposeOperations(manager).start_selection(ComposeSelection((app, optional), (app, optional), (), True))
    assert result.services == ("app",)


def test_preserved_disabled_model_keeps_missing_env_file_unresolved(monkeypatch):
    app = Container("app", {"app": {"image": "app"}})
    optional = Container("optional", {"optional": {"profiles": ["debug"]}})
    calls = []
    manager = SimpleNamespace(runtime=SimpleNamespace(
        create_docker_process=lambda *args, **kwargs: calls.append(args) or args))
    runner = ComposeRunner(manager)
    monkeypatch.setattr("linktools.cntr.artifacts.collect_candidates", lambda *args: {
        "app.yml": ("compose", "app", "services: {}")})
    runner._saved_compose_args = lambda *args: nullcontext(["compose"])
    inactive = {"image": "optional", "profiles": ["debug"], "env_file": [{"path": "/missing.env", "required": True}]}
    runner._resolved_model = lambda args: {
        "services": {"app": {"image": "app:unresolved"}, "optional": inactive}
    } if "--no-interpolate" in args else {"services": {"app": {"image": "app:resolved"}}}
    context = SimpleNamespace(project_containers=(app, optional), target_services=("app",), is_full_project=True)
    model = runner.final_model(context, preserve_disabled=True)
    assert model["services"] == {"app": {"image": "app:resolved"}, "optional": inactive}
    assert len(calls) == 2 and not any("--profile" in args for args in calls)


@pytest.mark.parametrize("restart", [False, True])
def test_full_operation_leaves_disabled_running_owner_unprepared(tmp_path, restart):
    from copy import deepcopy
    from test_lifecycle_rebuild import setup_case
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("optional", {"optional": {"image": "optional:new", "profiles": ["debug"]}}),
    ], running=("app", "optional"))
    manager.compose_runner.final_model = lambda context, preserve_disabled=False, **kwargs: deepcopy(
        manager.model if preserve_disabled else {"services": {"app": manager.model["services"]["app"]}})
    manager.containers["optional"].on_starting = lambda context, **kwargs: pytest.fail("disabled profile was prepared")
    manager.containers["optional"].on_check = lambda context, **kwargs: pytest.fail("disabled profile was checked")
    manager.containers["optional"].on_stopping = lambda context, **kwargs: pytest.fail("disabled profile was stopped")
    operation = manager.compose_operations.restart if restart else manager.compose_operations.up
    operation()
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert "optional" in manager.running_state.get_persisted()
