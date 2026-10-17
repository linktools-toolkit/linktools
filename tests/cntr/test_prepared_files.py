#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerManager, Integration, OperationContext
from linktools.cntr.artifacts import (AppliedServiceModels, prepare_files, prepared_paths,
                                     prepared_hooks, stage_files, prune_prepared_files)
from linktools.cntr.errors import ContainerError
from linktools.cntr.ext import endpoint
from linktools.cntr.lifecycle.hooks import HookPhase, HookRegistry
from linktools.cntr.runtime.inspect import ProjectRuntimeState, ServiceRuntimeState


class Index:
    def __init__(self):
        self.values = {}

    def record(self, entries, remove=()):
        self.values.update(entries)
        for key in remove:
            self.values.pop(key, None)

    def load(self):
        return dict(self.values)


@pytest.fixture
def owner(tmp_path):
    manager = SimpleNamespace(data_path=tmp_path, artifact_index=Index())
    root = tmp_path / "app/auth"
    value = SimpleNamespace(name="auth", services={"auth": {}, "admin": {}}, manager=manager,
                            get_app_path=lambda *parts: root.joinpath(*parts),
                            runtime=SimpleNamespace(chmod=lambda path, mode: os.chmod(path, mode)))
    value.hooks = HookRegistry(owner=value, scope="container")
    value.add_start_hook = lambda key, hook, **kwargs: value.hooks.register(
        HookPhase.BEFORE_START, hook, key=key, **kwargs)
    return value


def model(owner, path):
    binding = {"type": "bind", "source": str(path), "target": "/config", "read_only": True}
    return {"services": {
        "auth": {"image": "auth:new", "volumes": [dict(binding)]},
        "admin": {"image": "admin:new", "volumes": [dict(binding)]},
    }}


def context(owner, full=False):
    ctx = OperationContext(project_containers=[owner], target_services=("auth", "admin"),
                           is_full_project=full)
    ctx.initial_runtime_state = ProjectRuntimeState("test", (), "docker")
    return ctx


def declaration(consumer: str) -> Integration:
    value = Integration()
    value.consumer = consumer
    return value


def set_sources(owner: SimpleNamespace, sources: dict) -> None:
    manager = object.__new__(ContainerManager)
    containers = {name: SimpleNamespace(name=name, integrations=values) for name, values in sources.items()}
    containers.setdefault(owner.name, SimpleNamespace(name=owner.name, integrations=()))
    containers.setdefault("nginx", SimpleNamespace(name="nginx", integrations=()))
    manager.__dict__["containers"] = containers
    manager.__dict__["installed_state"] = SimpleNamespace(get=lambda resolve: tuple(containers.values()))
    owner.manager.integration_snapshot = manager.integration_snapshot


@pytest.mark.parametrize("nested_endpoint", [False, True])
def test_stage_records_named_and_anonymous_sources_in_existing_artifact_index(owner, nested_endpoint):
    auth = declaration("auth")
    sources = ([endpoint(object(), domain="app.test", extensions=[(auth,)])]
               if nested_endpoint else [declaration("nginx"), auth])
    set_sources(owner, {
        "app": sources,
        "other": [declaration("auth")],
        "unrelated": [declaration("nginx")],
    })
    stage_files(owner, {"configuration.yml": "content"})
    entries = owner.manager.artifact_index.load()
    assert len(entries) == 1
    assert next(iter(entries.values()))["producers"] == ["app", "other"]


def test_reusing_prepared_tree_refreshes_source_provenance(owner):
    set_sources(owner, {"first": [declaration("auth")]})
    first = stage_files(owner, {"configuration.yml": "unchanged"})
    assert next(iter(owner.manager.artifact_index.load().values()))["producers"] == ["first"]
    set_sources(owner, {"second": [declaration("auth")]})
    reused = stage_files(owner, {"configuration.yml": "unchanged"})
    assert reused == first
    assert next(iter(owner.manager.artifact_index.load().values()))["producers"] == ["second"]
    set_sources(owner, {"second": []})
    assert stage_files(owner, {"configuration.yml": "unchanged"}) == first
    assert next(iter(owner.manager.artifact_index.load().values()))["producers"] == []


def test_stage_is_immutable_private_and_does_not_publish(owner):
    first = stage_files(owner, {"nested/config.yml": "secret"})
    assert (first / "nested/config.yml").stat().st_mode & 0o777 == 0o600
    assert not owner.get_app_path("generated/current").exists()
    assert stage_files(owner, {"nested/config.yml": "secret"}) == first
    second = stage_files(owner, {"nested/config.yml": "new"})
    assert second != first
    assert (first / "nested/config.yml").read_text() == "secret"


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../escape", "a\\b"])
def test_stage_rejects_escaping_file_names(owner, name):
    with pytest.raises(ContainerError, match="within its tree"):
        stage_files(owner, {name: "bad"})
    assert not owner.get_app_path("generated").exists()


def test_stage_detects_content_tampering(owner):
    path = stage_files(owner, {"config": "original"})
    (path / "config").write_text("changed outside the operation")
    with pytest.raises(ContainerError, match="modified"):
        stage_files(owner, {"config": "original"})


def test_stage_detects_permission_tampering(owner):
    path = stage_files(owner, {"config": "secret"})
    (path / "config").chmod(0o644)
    with pytest.raises(ContainerError, match="modified"):
        stage_files(owner, {"config": "secret"})


def test_operation_accepts_one_prepared_tree_per_owner(owner):
    ctx = context(owner)
    path = ctx.write_files(owner, {"file": "content"})
    assert ctx.file_path(owner, "file") == path / "file"
    with pytest.raises(ValueError, match="already prepared"):
        ctx.write_files(owner, {"file": "other"})
    with pytest.raises(ValueError, match="within its tree"):
        ctx.file_path(owner, "../outside")


def test_prepare_is_pure_and_materializes_frozen_files_before_validation(owner):
    files = {"nested/config.yml": "secret"}
    path = prepare_files(owner, files)
    files["nested/config.yml"] = "mutated"
    files["added.yml"] = "new"
    assert not owner.get_app_path("generated").exists()
    assert owner.manager.artifact_index.load() == {}
    assert prepared_paths(owner) == (path,)
    hook, = prepared_hooks(owner)
    assert hook.key == ("generated-config", str(path))
    assert hook.source == "generated-config"
    assert hook.metadata == {"path": str(path), "files": ["nested/config.yml"]}
    validated = []
    owner.add_start_hook(("check",), lambda ctx: validated.append(ctx.file_path(owner, "nested/config.yml").read_text()))
    ctx = context(owner)
    owner.hooks.call(HookPhase.BEFORE_START, ctx)
    assert validated == ["secret"]
    assert ctx.prepared_dirs == {owner.name: path}
    assert not (path / "added.yml").exists()
    assert next(iter(owner.manager.artifact_index.load().values()))["tree"] == str(path.relative_to(owner.manager.data_path))


def test_same_preparation_registers_one_hook_without_writing(owner):
    path = prepare_files(owner, {"a": "1", "b": "2"})
    assert prepare_files(owner, {"b": "2", "a": "1"}) == path
    assert len(prepared_hooks(owner)) == 1
    assert not path.exists()
    owner.hooks.call(HookPhase.BEFORE_START, context(owner))
    assert path == stage_files(owner, {"a": "1", "b": "2"})


def test_declared_paths_ignore_unrelated_metadata_and_wrong_phase(owner):
    prepare_files(owner, {"a": "1"})
    declared = prepared_paths(owner)
    for phase, key, source, metadata in (
        (HookPhase.BEFORE_START, ("unrelated",), "generated-config", {"path": "/unrelated"}),
        (HookPhase.BEFORE_START, ("generated-config", "/wrong-source"), "other", {"path": "/wrong-source"}),
        (HookPhase.BEFORE_START, ("generated-config", "/wrong-path"), "generated-config", {"path": "/mismatch"}),
        (HookPhase.AFTER_START, ("generated-config", "/wrong-phase"), "generated-config", {"path": "/wrong-phase"}),
    ):
        owner.hooks.register(phase, lambda: None, key=key, source=source, metadata=metadata)
    assert prepared_paths(owner) == declared


@pytest.mark.parametrize("change", ["name", "content", "mode", "group"])
def test_tree_identity_includes_files_and_permissions(owner, change):
    files, mode, group = {"config": "content"}, 0o600, None
    first = prepare_files(owner, files, mode=mode, group=group)
    if change == "name":
        files = {"renamed": "content"}
    elif change == "content":
        files = {"config": "other"}
    elif change == "mode":
        mode = 0o640
    else:
        group = 1234
    assert prepare_files(owner, files, mode=mode, group=group) != first
    assert not owner.get_app_path("generated").exists()


def test_acl_change_changes_mount_source_for_each_consumer_service(owner):
    old = stage_files(owner, {"base.yml": "same", "acl.yml": "deny"})
    previous = model(owner, old)
    AppliedServiceModels(owner.manager, previous).record(("auth", "admin"))
    new = prepare_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    current = model(owner, new)
    store = AppliedServiceModels(owner.manager, current)
    assert store.changed_services == {"auth", "admin"}
    for service in current["services"]:
        assert current["services"][service]["volumes"][0]["source"] == str(new)
        assert "labels" not in current["services"][service]
    assert not new.exists()
    assert not owner.get_app_path("generated/current").exists()


def test_same_files_have_identical_model_identity(owner):
    first = stage_files(owner, {"base.yml": "same"})
    AppliedServiceModels(owner.manager, model(owner, first)).record(("auth", "admin"))
    later = prepare_files(owner, {"base.yml": "same"})
    assert not AppliedServiceModels(owner.manager, model(owner, later)).changed_services


def apply_prepared(owner, value, services=("auth", "admin"), running=(), full=False):
    ctx = context(owner, full=full)
    ctx.target_services = tuple(services)
    ctx.initial_runtime_state = ProjectRuntimeState("test", tuple(
        ServiceRuntimeState(logical_containers=(owner.name,), service=name, runtime_name=name,
                            state="running", health=None, image=None, exit_code=0, labels={}) for name in running), "docker")
    path = ctx.write_files(owner, {"base.yml": value})
    desired = model(owner, path)
    desired["services"] = {name: desired["services"][name] for name in services}
    owner.services = {name: {} for name in services}
    store = AppliedServiceModels(owner.manager, desired, retained_services=running)
    store.record(services)
    prune_prepared_files(ctx, store)
    return path, store


def test_pruning_retains_current_and_previous_exact_tree_sources(owner):
    first, _ = apply_prepared(owner, "old")
    abandoned = stage_files(owner, {"base.yml": "abandoned"})
    later, _ = apply_prepared(owner, "new")
    assert first.exists() and later.exists()
    assert not abandoned.exists()
    assert not owner.get_app_path("generated/current").exists()
    newest, _ = apply_prepared(owner, "newest")
    assert not first.exists()
    assert later.exists() and newest.exists()


def test_removed_running_service_retains_its_exact_tree_until_stopped(owner):
    original, _ = apply_prepared(owner, "A")
    generations = []
    for value in ("B", "C", "D"):
        path, store = apply_prepared(owner, value, services=("auth",), running=("auth", "admin"))
        generations.append(path)
        assert original.exists()
        assert "admin" in store.previous and "admin" not in store.current
    assert not generations[0].exists()
    assert generations[1].exists()
    entry = str((original / "base.yml").relative_to(owner.manager.data_path))
    assert entry in owner.manager.artifact_index.load()
    apply_prepared(owner, "E", services=("auth",), running=("auth",))
    assert not original.exists()
    assert entry not in owner.manager.artifact_index.load()


@pytest.mark.parametrize("source", ["generated", "generated/current", "generated/current/base.yml"])
def test_uncertain_legacy_mount_preserves_generations_and_pointer(owner, source):
    original = stage_files(owner, {"base.yml": "old"})
    current = owner.get_app_path("generated/current")
    current.symlink_to(original.name)
    legacy = model(owner, owner.get_app_path(source))
    AppliedServiceModels(owner.manager, legacy).record(("auth", "admin"))
    unknown = stage_files(owner, {"base.yml": "unknown"})
    for value in ("B", "C"):
        apply_prepared(owner, value, services=("auth",), running=("auth", "admin"))
        assert original.exists() and unknown.exists()
        assert current.resolve() == original


def test_untracked_running_service_prevents_destructive_pruning(owner):
    original, _ = apply_prepared(owner, "old", services=("auth",))
    abandoned = stage_files(owner, {"base.yml": "unknown"})
    for value in ("B", "C"):
        _, store = apply_prepared(owner, value, services=("auth",), running=("auth", "admin"))
        assert store.untracked_services == {"admin"}
        assert original.exists() and abandoned.exists()
    apply_prepared(owner, "D", services=("auth",), running=("auth",))
    assert not original.exists() and not abandoned.exists()


def test_unindexed_legacy_tree_is_never_deleted(owner):
    original = stage_files(owner, {"config": "legacy"})
    for value in owner.manager.artifact_index.values.values():
        value.pop("tree")
    apply_prepared(owner, "new")
    assert original.exists()


def test_snapshot_restore_does_not_undo_successful_sibling(owner):
    original = {"services": {"a": {"image": "a:old"}, "b": {"image": "b:old"}}}
    desired = {"services": {"a": {"image": "a:new"}, "b": {"image": "b:new"}}}
    AppliedServiceModels(owner.manager, original).record(("a", "b"))
    state = AppliedServiceModels(owner.manager, desired)
    state.record(("a", "b"))
    state.restore(("b",))
    current = AppliedServiceModels(owner.manager, desired)
    assert yaml.safe_load(current.previous["a"])["services"]["a"]["image"] == "a:new"
    assert yaml.safe_load(current.previous["b"])["services"]["b"]["image"] == "b:old"
    assert current.changed_services == {"b"}


def test_pruning_retains_tree_for_exact_file_sources_across_services(owner):
    original = stage_files(owner, {"base.yml": "original", "nested/acl.yml": "deny"})
    applied = model(owner, original / "base.yml")
    applied["services"]["admin"]["volumes"][0]["source"] = str(original / "nested/acl.yml")
    AppliedServiceModels(owner.manager, applied).record(("auth", "admin"))
    for value in ("B", "C", "D"):
        _, store = apply_prepared(owner, value, services=("auth",), running=("auth", "admin"))
        assert store.previous_model("admin")["services"]["admin"]["volumes"][0]["source"] == str(original / "nested/acl.yml")
        assert original.exists()
    apply_prepared(owner, "E", services=("auth",), running=("auth",))
    assert not original.exists()
