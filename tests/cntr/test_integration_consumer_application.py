#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Running integration consumers apply only explicitly declared generated trees."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr import OperationContext, ContainerError
from linktools.cntr.artifacts import AppliedServiceModels, prepared_hooks
from test_lifecycle_rebuild import setup_case


def consumer_case(tmp_path, running=True, changed=True, historical=False, empty_prior=False):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("consumer", {"consumer": {"image": "consumer:stable", "environment": {"SETTING": "stable"}}}),
    ], running=("consumer",) if running else ())
    owner = manager.containers["consumer"]
    owner.runtime = SimpleNamespace(chmod=lambda path, mode: path.chmod(mode))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="consumer"),)
    prior = OperationContext(project_containers=list(manager.containers.values()), target_services=("consumer",))
    old_path = prior.write_files(owner, {} if empty_prior else {"config.yml": "old"})
    manager.model["services"]["consumer"]["volumes"] = [
        {"type": "bind", "source": str(old_path), "target": "/generated", "read_only": True}]
    if running:
        AppliedServiceModels(manager, deepcopy(manager.model)).record(("consumer",))
    if historical:
        manager.integration_snapshot["app"] = ()
    owner.config_path = owner.prepare_files({"config.yml": "new" if changed else "old"})
    owner.old_config_path = old_path
    manager.model["services"]["consumer"]["volumes"][0]["source"] = str(owner.config_path)
    owner.on_starting = lambda ctx: pytest.fail("collateral consumer initialization")
    owner.add_start_hook(("ordinary",), lambda: pytest.fail("collateral ordinary start hook"))
    old_image_id = manager.image_preparer.image_id
    manager.image_preparer.image_id = lambda image: image if image.startswith("sha256:") else old_image_id(image)
    manager.applied_models = []
    apply = manager.compose_runner.apply_services

    def capture(ctx, services):
        manager.applied_models.append(deepcopy(ctx.compose_model))
        apply(ctx, services)

    manager.compose_runner.apply_services = capture
    return manager


def assert_no_application(manager):
    assert not any(event[0] in ("images", "apply", "stop") for event in manager.events)


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("action", ["up", "restart"])
def test_changed_running_consumer_is_applied_without_initialization_or_image_refresh(tmp_path, historical, action):
    manager = consumer_case(tmp_path, historical=historical)
    owner = manager.containers["consumer"]
    assert not owner.config_path.exists()
    getattr(manager.compose_operations, action)(["app"], pull=True)
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
    spec = manager.applied_models[-1]["services"]["consumer"]
    assert spec["environment"] == {"SETTING": "stable"}
    assert spec["image"] == "sha256:old-consumer"
    assert spec["volumes"] == [{"type": "bind", "source": str(owner.config_path),
                               "target": "/generated", "read_only": True}]
    assert (owner.config_path / "config.yml").read_text() == "new"
    assert (owner.old_config_path / "config.yml").read_text() == "old"
    assert ("check-callback", "consumer") in manager.events
    assert ("after-callback", "consumer") in manager.events
    assert all("consumer" not in event[1] for event in manager.events if event[0] == "stop")


@pytest.mark.parametrize("running,changed", [(False, True), (True, False)])
def test_stopped_or_unchanged_consumer_is_not_applied_or_notified(tmp_path, running, changed):
    manager = consumer_case(tmp_path, running=running, changed=changed)
    owner = manager.containers["consumer"]
    hook = prepared_hooks(owner)[0]
    owner.runtime.chmod = lambda *args: pytest.fail("unselected generated files written")
    before = {key: value for key, value in manager.artifact_index.load().items()
              if value.get("container") == "consumer"}
    manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert ("check-callback", "consumer") not in manager.events
    assert ("after-callback", "consumer") not in manager.events
    assert {key: value for key, value in manager.artifact_index.load().items()
            if value.get("container") == "consumer"} == before
    assert prepared_hooks(owner) == (hook,)
    assert owner.config_path.exists() is (not changed)


def test_missing_consumer_baseline_requires_explicit_selection_before_mutation(tmp_path):
    manager = consumer_case(tmp_path)
    for path in (tmp_path / "compose/applied/services").glob("*.yml"):
        path.unlink()
    with pytest.raises(ContainerError, match="explicit"):
        manager.compose_operations.up(["app"])
    assert_no_application(manager)
    assert not manager.containers["consumer"].config_path.exists()


def test_failed_generated_preparation_does_not_publish_or_apply(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    previous = dict(AppliedServiceModels(manager, manager.model).previous)

    def fail(path, mode):
        raise ContainerError("invalid consumer input")

    owner.runtime.chmod = fail
    with pytest.raises(ContainerError, match="invalid consumer input"):
        manager.compose_operations.restart(["app"])
    assert_no_application(manager)
    assert dict(AppliedServiceModels(manager, manager.model).previous) == previous
    assert not owner.config_path.exists()
    assert (owner.old_config_path / "config.yml").read_text() == "old"


def test_consumer_partial_failure_does_not_record_unapplied_generation(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    before = AppliedServiceModels(manager, manager.model).previous["consumer"]
    manager.compose_runner.fail = "consumer"
    with pytest.raises(ContainerError, match="apply failed consumer"):
        manager.compose_operations.up(["app"])
    assert AppliedServiceModels(manager, manager.model).previous["consumer"] == before
    assert ("after-callback", "consumer") not in manager.events
    assert (owner.old_config_path / "config.yml").read_text() == "old"
    assert (owner.config_path / "config.yml").read_text() == "new"


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("pending", ["environment", "image", "dependency", "mount", "network", "build"])
def test_pending_consumer_changes_require_explicit_selection_only_for_generated_change(tmp_path, changed, pending):
    manager = consumer_case(tmp_path, changed=changed)
    spec = manager.model["services"]["consumer"]
    if pending == "environment":
        spec["environment"] = {"SETTING": "pending"}
    elif pending == "image":
        spec["image"] = "consumer:pending"
    elif pending == "dependency":
        spec["depends_on"] = {"consumer": {"condition": "service_started"}}
    elif pending == "mount":
        spec["volumes"].append({"type": "bind", "source": str(tmp_path / "pending"), "target": "/pending"})
    elif pending == "network":
        manager.model["networks"] = {"default": {"name": "pending-network"}}
    else:
        spec["build"] = {"context": str(tmp_path / "pending-build")}
    before = AppliedServiceModels(manager, manager.model).previous["consumer"]
    if changed:
        with pytest.raises(ContainerError, match="consumer.*explicit"):
            manager.compose_operations.up(["app"])
        assert_no_application(manager)
        assert not manager.containers["consumer"].config_path.exists()
    else:
        manager.compose_operations.up(["app"])
        assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
        assert ("check-callback", "consumer") not in manager.events
        assert ("after-callback", "consumer") not in manager.events
    assert AppliedServiceModels(manager, manager.model).previous["consumer"] == before


@pytest.mark.parametrize("action", ["up", "restart"])
def test_unchanged_consumer_does_not_inspect_pending_build_inputs(tmp_path, action):
    from linktools.cntr.runtime.images import ImagePreparer

    manager = consumer_case(tmp_path, changed=False)
    manager.model["services"]["consumer"]["build"] = {"context": str(tmp_path / "missing-build")}
    manager.image_preparer.with_build_revisions = ImagePreparer(manager).with_build_revisions
    getattr(manager.compose_operations, action)(["app"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert ("check-callback", "consumer") not in manager.events
    assert ("after-callback", "consumer") not in manager.events


@pytest.mark.parametrize("build_changed", [False, True])
def test_changed_consumer_still_validates_automatic_build_identity(tmp_path, build_changed):
    from linktools.cntr.runtime.images import ImagePreparer

    manager = consumer_case(tmp_path)
    prepare = ImagePreparer(manager).with_build_revisions
    manager.image_preparer.with_build_revisions = prepare
    owner = manager.containers["consumer"]
    spec = manager.model["services"]["consumer"]
    spec["build"] = {"context": str(tmp_path), "dockerfile_inline": "FROM scratch"}
    previous = AppliedServiceModels(manager, manager.model).previous_model("consumer")
    previous["services"]["consumer"]["build"] = dict(spec["build"])
    previous = prepare(previous, (owner,), ("consumer",))
    AppliedServiceModels(manager, previous).record(("consumer",))
    if build_changed:
        spec["build"]["dockerfile_inline"] = "FROM alpine"
        with pytest.raises(ContainerError, match="outside.*explicitly select"):
            manager.compose_operations.up(["app"])
        assert_no_application(manager)
    else:
        manager.compose_operations.up(["app"])
        assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}
        assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
        saved = AppliedServiceModels(manager, manager.model).previous_model("consumer")
        assert saved["services"]["consumer"]["build"] == previous["services"]["consumer"]["build"]


@pytest.mark.parametrize("action", ["up", "restart"])
@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_running_profiled_consumer_is_resolved_before_config_comparison(tmp_path, action, running, changed):
    from contextlib import nullcontext
    from linktools.cntr.runtime.compose import ComposeRunner
    from linktools.cntr.runtime.images import ImagePreparer

    manager = consumer_case(tmp_path, running=running, changed=changed)
    manager.containers["consumer"].services["consumer"]["profiles"] = ["optional"]
    manager.model["services"]["consumer"]["profiles"] = ["optional"]
    if running:
        previous = AppliedServiceModels(manager, manager.model).previous_model("consumer")
        previous["services"]["consumer"]["profiles"] = ["optional"]
        AppliedServiceModels(manager, previous).record(("consumer",))
    manager.runtime = SimpleNamespace(create_docker_process=lambda *args, **kwargs: args)
    native = ComposeRunner(manager)
    native._saved_compose_args = lambda *args: nullcontext(["compose"])
    scopes = []

    def resolve(args):
        scope = args[args.index("json") + 1:]
        scopes.append(scope)
        model = deepcopy(manager.model)
        if "--no-interpolate" not in scope and "consumer" not in scope:
            model["services"].pop("consumer")
        return model

    native._resolved_model = resolve
    manager.compose_runner.final_model = native.final_model
    manager.image_preparer.with_build_revisions = ImagePreparer(manager).with_build_revisions
    getattr(manager.compose_operations, action)(["app"], pull=True)
    assert scopes[0] == ("app",)
    assert any("consumer" in scope for scope in scopes) is running
    assert {event[1] for event in manager.events if event[0] == "apply"} == (
        {"app", "consumer"} if running and changed else {"app"})
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
    assert all("consumer" not in event[1] for event in manager.events if event[0] == "stop")
    assert (("check-callback", "consumer") in manager.events) is (running and changed)
    assert (("after-callback", "consumer") in manager.events) is (running and changed)


def test_changed_consumer_keeps_dependency_order_for_validation_and_notifications(tmp_path):
    manager = consumer_case(tmp_path)
    manager.model["services"]["app"]["depends_on"] = {"consumer": {"condition": "service_started"}}
    manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "check-callback"] == ["consumer", "app"]
    assert [event[1] for event in manager.events if event[0] == "after-callback"] == ["app", "consumer"]
    assert ("before-start", "consumer") not in manager.events


@pytest.mark.parametrize("other_state", ["exited", "running"])
def test_collateral_image_comes_only_from_unambiguous_running_replicas(tmp_path, other_state):
    from linktools.cntr.runtime.inspect import ProjectRuntimeState
    manager = consumer_case(tmp_path)
    actual = manager.docker_inspector.get_project_state(None)
    extra = SimpleNamespace(service="consumer", state=other_state, image_id="sha256:other", labels={})
    mixed = ProjectRuntimeState("test", actual.services + (extra,), "docker")
    manager.docker_inspector.get_project_state = lambda containers, **kwargs: mixed
    if other_state == "running":
        with pytest.raises(ContainerError, match="one running image"):
            manager.compose_operations.up(["app"])
        assert_no_application(manager)
    else:
        manager.compose_operations.up(["app"])
        assert manager.applied_models[-1]["services"]["consumer"]["image"] == "sha256:old-consumer"


def test_partial_failure_does_not_roll_back_successful_service_without_shared_inputs(tmp_path):
    manager = setup_case(tmp_path, [("auth", {
        "auth": {"image": "auth:stable"},
        "admin": {"image": "admin:stable", "depends_on": ["auth"]},
    })], running=("auth", "admin"))
    owner = manager.containers["auth"]
    owner.runtime = SimpleNamespace(chmod=lambda path, mode: path.chmod(mode))
    prior = OperationContext(project_containers=[owner])
    old_path = prior.write_files(owner, {"config.yml": "old"})
    manager.model["services"]["auth"]["volumes"] = [{
        "type": "bind", "source": str(old_path), "target": "/generated", "read_only": True}]
    AppliedServiceModels(manager, deepcopy(manager.model)).record(("auth", "admin"))
    before = AppliedServiceModels(manager, manager.model).previous["admin"]
    new_path = owner.prepare_files({"config.yml": "new"})
    manager.model["services"]["auth"]["volumes"][0]["source"] = str(new_path)
    manager.compose_runner.fail = "admin"
    with pytest.raises(ContainerError, match="apply failed admin"):
        manager.compose_operations.up(["auth"])
    after = AppliedServiceModels(manager, manager.model)
    assert after.previous_model("auth")["services"]["auth"]["volumes"][0]["source"] == str(new_path)
    assert after.previous["admin"] == before
    assert [event[1] for event in manager.events if event[0] == "restore"] == [("admin",)]
    assert (old_path / "config.yml").read_text() == "old"
    assert (new_path / "config.yml").read_text() == "new"


@pytest.mark.parametrize("changed", [False, True])
def test_shared_contributor_requirement_applies_only_to_changed_consumer(tmp_path, changed):
    from test_lifecycle_rebuild import Owner
    manager = consumer_case(tmp_path, changed=changed)
    manager.containers["contributor"] = Owner("contributor", manager.containers["consumer"].services, manager)
    manager.integration_snapshot["contributor"] = ()
    manager.load_installed_config_metadata = lambda: list(manager.containers.values())
    native_model = manager.compose_runner.final_model
    manager.compose_runner.final_model = lambda context, **kwargs: native_model(context)
    if changed:
        with pytest.raises(ContainerError, match="Shared integration consumer contributors.*contributor"):
            manager.compose_operations.up(["app"])
        assert_no_application(manager)
    else:
        manager.compose_operations.up(["app"])
        assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]


def test_unchanged_consumer_network_definition_is_preserved(tmp_path):
    manager = consumer_case(tmp_path)
    previous = AppliedServiceModels(manager, manager.model).previous_model("consumer")
    for model in (previous, manager.model):
        model["services"]["consumer"]["networks"] = {"private": {}}
        model["networks"] = {"private": {"name": "applied-private"}}
    AppliedServiceModels(manager, previous).record(("consumer",))
    manager.compose_operations.up(["app"])
    assert manager.applied_models[-1]["networks"]["private"] == {"name": "applied-private"}
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}


def test_consumer_snapshot_can_predate_new_explicit_producer(tmp_path):
    manager = consumer_case(tmp_path)
    previous = AppliedServiceModels(manager, manager.model).previous_model("consumer")
    previous["services"].pop("app")
    AppliedServiceModels(manager, previous).record(("consumer",))
    manager.compose_operations.up(["app"])
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}


def test_deferred_generation_is_checked_before_apply_without_rewriting_mounts(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    source = owner.config_path
    snapshot = manager.integration_snapshot
    declarations = snapshot["app"]
    assert not source.exists()
    check = owner.on_check

    def checked(context):
        assert (owner.old_config_path / "config.yml").read_text() == "old"
        assert context.file_path(owner, "config.yml") == source / "config.yml"
        assert (source / "config.yml").read_text() == "new"
        assert context.compose_model["services"]["consumer"]["volumes"][0]["source"] == str(source)
        assert manager.integration_snapshot is snapshot
        assert snapshot["app"] is declarations
        check(context)

    owner.on_check = checked
    manager.compose_operations.up(["app"])
    assert manager.model["services"]["consumer"]["volumes"][0]["source"] == str(source)
    assert not owner.get_app_path("generated/current").exists()


def test_failed_check_leaves_old_immutable_files_and_saved_model_untouched(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    before = AppliedServiceModels(manager, manager.model).previous["consumer"]

    def fail(context):
        raise ContainerError("candidate invalid")

    owner.on_check = fail
    with pytest.raises(ContainerError, match="candidate invalid"):
        manager.compose_operations.up(["app"])
    assert (owner.old_config_path / "config.yml").read_text() == "old"
    assert (owner.config_path / "config.yml").read_text() == "new"
    assert AppliedServiceModels(manager, manager.model).previous["consumer"] == before
    assert not any(event[0] == "apply" for event in manager.events)


def test_failed_apply_uses_successful_saved_model_despite_other_staged_generation(tmp_path):
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    interrupted = OperationContext(project_containers=list(manager.containers.values()))
    pending = interrupted.write_files(owner, {"config.yml": "interrupted"})
    before = AppliedServiceModels(manager, manager.model).previous["consumer"]
    manager.compose_runner.fail = "consumer"
    with pytest.raises(ContainerError, match="apply failed consumer"):
        manager.compose_operations.up(["app"])
    assert AppliedServiceModels(manager, manager.model).previous["consumer"] == before
    assert (owner.old_config_path / "config.yml").read_text() == "old"
    assert (pending / "config.yml").read_text() == "interrupted"


def test_running_sidecar_without_generated_mount_does_not_write_or_start_its_owner(tmp_path):
    manager = setup_case(tmp_path, [
        ("app", {"app": {"image": "app:new"}}),
        ("authelia", {name: {"image": name + ":new"} for name in ("authelia", "authelia-admin", "authelia-redis")}),
    ], running=("authelia-redis",))
    owner = manager.containers["authelia"]
    owner.config_path = owner.prepare_files({"config.yml": "new"})
    for name in ("authelia", "authelia-admin"):
        manager.model["services"][name]["volumes"] = [{
            "type": "bind", "source": str(owner.config_path), "target": "/generated", "read_only": True}]
    owner.on_starting = lambda ctx: pytest.fail("sidecar owner initialization")
    owner.runtime = SimpleNamespace(chmod=lambda *args: pytest.fail("sidecar generated write"))
    manager.integration_snapshot["app"] = (SimpleNamespace(consumer="authelia"),)
    manager.compose_operations.up(["app"])
    assert [event[1] for event in manager.events if event[0] == "apply"] == ["app"]
    assert not owner.config_path.exists()


def test_shared_bind_recovery_still_includes_transitive_peers(tmp_path):
    old_file, new_file = tmp_path / "old.cfg", tmp_path / "new.cfg"
    old_file.write_text("old")
    new_file.write_text("new")
    old_mount = {"type": "bind", "source": str(old_file), "target": "/shared.cfg"}
    new_mount = dict(old_mount, source=str(new_file))
    manager = setup_case(tmp_path, [
        ("peer", {"peer": {"image": "peer:new", "volumes": [new_mount]}}),
        ("auth", {
            "auth": {"image": "auth:new", "volumes": [new_mount]},
            "admin": {"image": "admin:new", "depends_on": ["auth"], "volumes": [new_mount]},
        }),
    ], running=("peer", "auth", "admin"))
    previous = deepcopy(manager.model)
    for spec in previous["services"].values():
        spec["volumes"] = [old_mount]
    AppliedServiceModels(manager, previous).record(("peer", "auth", "admin"))
    before = dict(AppliedServiceModels(manager, previous).previous)
    manager.compose_runner.fail = "admin"
    with pytest.raises(ContainerError, match="apply failed admin"):
        manager.compose_operations.up(["peer", "auth"])
    restored = {service for event in manager.events if event[0] == "restore" for service in event[1]}
    assert restored == {"peer", "auth", "admin"}
    after = AppliedServiceModels(manager, manager.model).previous
    assert all(after[name] == before[name] for name in before)


@pytest.mark.parametrize("build", [False, True])
def test_repeated_generated_updates_keep_declared_image_and_build_in_saved_model(tmp_path, build):
    from linktools.cntr.lifecycle.hooks import HookPhase
    manager = consumer_case(tmp_path)
    owner = manager.containers["consumer"]
    if build:
        definition = {"context": str(tmp_path / "consumer-source")}
        manager.model["services"]["consumer"]["build"] = definition
        previous = AppliedServiceModels(manager, manager.model).previous_model("consumer")
        previous["services"]["consumer"]["build"] = definition
        AppliedServiceModels(manager, previous).record(("consumer",))
    paths = []
    for content in ("new", "newer"):
        if paths:
            for hook in prepared_hooks(owner):
                owner.hooks.unregister(HookPhase.BEFORE_START, hook.key)
            owner.config_path = owner.prepare_files({"config.yml": content})
            manager.model["services"]["consumer"]["volumes"][0]["source"] = str(owner.config_path)
        manager.events.clear()
        manager.compose_operations.up(["app"], pull=True)
        paths.append(owner.config_path)
        execution = manager.applied_models[-1]["services"]["consumer"]
        saved = AppliedServiceModels(manager, manager.model).previous_model("consumer")["services"]["consumer"]
        assert execution["image"] == "sha256:old-consumer"
        assert "build" not in execution
        assert saved["image"] == "consumer:stable"
        assert saved.get("build") == (definition if build else None)
        assert saved["volumes"][0]["source"] == str(owner.config_path)
        assert (owner.config_path / "config.yml").read_text() == content
        assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
        assert [event[1] for event in manager.events if event[0] == "apply"] == ["app", "consumer"]
    assert paths[0] != paths[1]


@pytest.mark.parametrize("action", ["up", "restart"])
def test_empty_historical_tree_still_selects_running_consumer(tmp_path, action):
    manager = consumer_case(tmp_path, historical=True, empty_prior=True)
    getattr(manager.compose_operations, action)(["app"], pull=True)
    assert {event[1] for event in manager.events if event[0] == "apply"} == {"app", "consumer"}
    assert [event[1] for event in manager.events if event[0] == "images"] == [("app",)]
