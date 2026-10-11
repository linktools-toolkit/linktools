#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple Compose authors retain one native service model and lifecycle scope."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr.execution.planner import ExecutionPlanner
from linktools.cntr.lifecycle.hooks import HookPhase, HookRegistry
from linktools.cntr.runtime.compose import ComposeRunner
from linktools.cntr.runtime.images import ImagePreparer, ImagePreparationError
from linktools.cntr.runtime.process import CommandSpec
from test_lifecycle_rebuild import setup_case


def shared_case(tmp_path, running=(), dependency=False, sibling=False):
    app = {"image": "app:new"}
    if dependency:
        app["depends_on"] = {"db": {"condition": "service_healthy"}}
    overlay = {"app": {"image": "app:new", "environment": {"EXTRA": "yes"}}}
    if sibling:
        overlay["other"] = {"image": "other:new"}
    groups = [("base", {"app": app}), ("overlay", overlay)]
    if dependency:
        groups.append(("database", {"db": {"image": "db:new"}}))
    manager = setup_case(tmp_path, groups, running=running)
    manager.model = deepcopy(manager.model)
    manager.model["services"]["app"].update(deepcopy(app))
    native = manager.compose_runner.final_model
    manager.compose_runner.final_model = lambda context, **kwargs: native(context)
    return manager


@pytest.mark.parametrize("action", ["up", "restart"])
def test_shared_service_runs_all_author_hooks_without_starting_siblings(tmp_path, action):
    manager = shared_case(tmp_path, running=("app",), sibling=True)
    getattr(manager.compose_operations, action)(["base"])
    assert [event[1] for event in manager.events if event[0] == "prepare"] == ["base", "overlay"]
    assert [event[1] for event in manager.events if event[0] == "check-callback"] == ["base", "overlay"]
    assert [event[1] for event in manager.events if event[0] == "after-callback"] == ["overlay", "base"]
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert manager.running_state.get_persisted() == ["base", "overlay"]
    if action == "restart":
        assert ("stop", ("app",)) in manager.events
        assert [event[1] for event in manager.events if event[0] == "stopping-callback"] == ["overlay", "base"]


def test_earlier_fragment_dependency_is_resolved_from_native_model(tmp_path):
    manager = shared_case(tmp_path, dependency=True)
    manager.compose_operations.up(["base"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["db", "app"]
    assert [event[1] for event in manager.events if event[0] == "prepare"] == ["database", "base", "overlay"]


@pytest.mark.parametrize("sibling_running", [False, True])
def test_shared_down_updates_all_authors_and_preserves_untouched_running_sibling(tmp_path, sibling_running):
    manager = shared_case(tmp_path, running=("app", "other") if sibling_running else ("app",), sibling=True)
    manager.compose_operations.down(["base"])
    assert ("down", ("app",)) in manager.events
    assert [event[1] for event in manager.events if event[0] == "stopping-callback"] == ["overlay", "base"]
    assert manager.running_state.get_persisted() == (["overlay"] if sibling_running else [])


def test_shared_profile_resolution_includes_runtime_providers_before_native_selection(tmp_path):
    manager = shared_case(tmp_path)
    provider = setup_case(tmp_path / "provider", [("provider", {"provider": {
        "image": "provider:new", "profiles": ["private"]}})]).containers["provider"]
    manager.containers["provider"] = provider
    manager.load_installed_config_metadata = lambda: list(manager.containers.values())
    manager.containers["overlay"].get_runtime_requirements = lambda required: (
        {"provider": ("provider",)} if "overlay" in required else {})
    manager.model["services"]["provider"] = deepcopy(provider.services["provider"])
    observed = []

    def native(context, **kwargs):
        observed.append((context.target_services, kwargs.get("privilege")))
        return {"services": {name: spec for name, spec in manager.model["services"].items()
                             if name == "app" or name in context.target_services}}

    manager.compose_runner.final_model = native
    selection = manager.compose_operations.start_selection(
        manager.compose_operations.select(["base"], for_start=True), privilege=False)
    assert set(selection.services) == {"app", "provider"}
    assert {owner.name for owner in selection.target_containers} == {"base", "overlay", "provider"}
    assert observed == [(("app", "provider"), False)]


def test_shared_contributor_dependencies_do_not_select_its_unrelated_siblings(tmp_path):
    manager = shared_case(tmp_path, sibling=True)
    database = setup_case(tmp_path / "database", [("database", {"db": {"image": "db:new"}})]).containers["database"]
    database.manager = manager
    manager.containers["database"] = database
    manager.load_installed_config_metadata = lambda: list(manager.containers.values())
    manager.model["services"]["db"] = deepcopy(database.services["db"])
    manager.containers["overlay"].dependencies = ("database",)
    manager.compose_operations.up(["base"])
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "db"}
    assert {event[1] for event in manager.events if event[0] == "prepare"} == {"base", "overlay", "database"}


@pytest.mark.parametrize("action", ["up", "restart", "down"])
def test_shared_plan_uses_canonical_dependencies_and_all_author_hooks(tmp_path, action):
    manager = shared_case(tmp_path, dependency=True)
    for owner in manager.containers.values():
        owner.hooks = HookRegistry(owner)
        owner.hooks.register(HookPhase.BEFORE_START, lambda: None, key="prepare", name=owner.name)
        owner.hooks.register(HookPhase.BEFORE_STOP, lambda: None, key="stop", name=owner.name)
    manager.hooks = HookRegistry(manager)
    native = ComposeRunner(manager)
    manager.compose_runner.compose_args = native.compose_args
    manager.compose_runner.apply_services_args = native.apply_services_args
    manager.docker_inspector.preflight_candidates = lambda files: "passed"
    manager.runtime = SimpleNamespace(docker_args=lambda *args, **kwargs: CommandSpec(args, False),
                                      display_args=lambda spec: spec.args)
    plan = ExecutionPlanner(manager).plan(action, names=["base"])
    assert plan.targets == ("base",)
    if action != "down":
        assert plan.commands[-1].args[-2:] == ("db", "app")
        assert {hook.container for hook in plan.hooks} == {"base", "overlay", "database"}
    if action != "up":
        assert plan.commands[0].args[-1] == "app"
        assert {hook.container for hook in plan.hooks if hook.phase == "before-stop"} == {"base", "overlay"}


def test_unique_service_start_does_not_add_native_model_queries(tmp_path):
    manager = setup_case(tmp_path, [("app", {"app": {"image": "app:new"}})])
    manager.compose_operations.up(["app"])
    assert manager.events.count(("model",)) == 1


def build_owner(name, revision, build):
    return SimpleNamespace(name=name, services={"app": {"build": build} if build is not None else {}},
                           docker_file="FROM scratch\n", get_build_revision=lambda service: revision)


@pytest.mark.parametrize("overlay_build", [None, {"args": {"EXTRA": "yes"}}])
def test_shared_service_preserves_managed_build_revision_from_earlier_author(overlay_build):
    preparer = ImagePreparer(SimpleNamespace())
    model = {"services": {"app": {"image": "app:fixed", "build": {
        "context": "/source", "dockerfile": "/generated/base.Dockerfile"}}}}
    owners = [build_owner("base", "source-revision", model["services"]["app"]["build"]),
              build_owner("overlay", None, overlay_build)]
    result = preparer.with_build_revisions(model, owners, ("app",))
    assert preparer.BUILD_LABEL in result["services"]["app"]["build"]["labels"]


def test_shared_managed_build_claims_select_effective_native_dockerfile():
    preparer = ImagePreparer(SimpleNamespace())
    first = {"context": "/first", "dockerfile": "/generated/first.Dockerfile"}
    second = {"context": "/second", "dockerfile": "/generated/second.Dockerfile"}
    model = {"services": {"app": {"image": "app:fixed", "build": second}}}
    owners = [build_owner("first", "first-revision", first), build_owner("second", "second-revision", second)]
    expected = preparer.with_build_revisions(model, owners[1:], ("app",))
    assert preparer.with_build_revisions(model, owners, ("app",)) == expected


def test_conflicting_shared_managed_build_claims_fail_before_build():
    preparer = ImagePreparer(SimpleNamespace())
    build = {"context": "/source", "dockerfile": "/generated/shared.Dockerfile"}
    model = {"services": {"app": {"image": "app:fixed", "build": build}}}
    owners = [build_owner("first", "first-revision", build), build_owner("second", "second-revision", build)]
    with pytest.raises(ImagePreparationError, match="Ambiguous.*app"):
        preparer.with_build_revisions(model, owners, ("app",))


def test_identical_shared_managed_build_claims_keep_one_identity():
    preparer = ImagePreparer(SimpleNamespace())
    build = {"context": "/source", "dockerfile": "/generated/shared.Dockerfile"}
    model = {"services": {"app": {"image": "app:fixed", "build": build}}}
    owners = [build_owner("first", "same-revision", build), build_owner("second", "same-revision", build)]
    assert preparer.with_build_revisions(model, owners, ("app",)) == preparer.with_build_revisions(model, owners[:1], ("app",))


def test_mixed_managed_context_and_dockerfile_claims_are_not_guessed():
    preparer = ImagePreparer(SimpleNamespace())
    first = {"context": "/first", "dockerfile": "/generated/first.Dockerfile"}
    second = {"context": "/second"}
    model = {"services": {"app": {"image": "app:fixed", "build": dict(first, context="/second")}}}
    owners = [build_owner("first", "first-revision", first), build_owner("second", "second-revision", second)]
    with pytest.raises(ImagePreparationError, match="Ambiguous.*app"):
        preparer.with_build_revisions(model, owners, ("app",))


@pytest.mark.parametrize("initial_app", [False, True])
def test_shared_service_failure_restores_all_author_state_without_touching_sibling(tmp_path, initial_app):
    manager = shared_case(tmp_path, running=("app", "other") if initial_app else ("other",), sibling=True)
    manager.compose_runner.fail = "app"
    with pytest.raises(Exception, match="apply failed app"):
        manager.compose_operations.up(["base"])
    assert manager.running_state.get_persisted() == (["base", "overlay"] if initial_app else ["overlay"])
    assert not any(event[0] == "apply" and event[1] == "other" for event in manager.events)
