#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import OperationContext
from linktools.cntr.artifacts import (AppliedServiceModels, stage_files, bind_prepared_files,
                                     publish_prepared_files, prune_prepared_files)
from linktools.cntr.errors import ContainerError


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
    return SimpleNamespace(name="auth", services={"auth": {}, "admin": {}}, manager=manager,
                           get_app_path=lambda *parts: root.joinpath(*parts),
                           runtime=SimpleNamespace(chmod=lambda path, mode: os.chmod(path, mode)))


def model(owner, base=True, acl=True):
    root = owner.get_app_path("generated/current")
    bindings = lambda names: [{"type": "bind", "source": str(root / name), "target": "/" + name,
                               "read_only": True} for name in names]
    return {"services": {
        "auth": {"image": "auth:new", "volumes": bindings(["base.yml", "acl.yml"])},
        "admin": {"image": "admin:new", "volumes": bindings(["base.yml"])},
    }}


def context(owner, full=False):
    ctx = OperationContext(project_containers=[owner], target_services=("auth", "admin"),
                           is_full_project=full)
    ctx.initial_running_services = set()
    ctx.service_models = SimpleNamespace(previous={}, untracked_services=frozenset())
    return ctx


def test_stage_records_sources_in_existing_artifact_index(owner):
    owner.manager.integration_snapshot = {
        "app": (SimpleNamespace(consumer="nginx", expose=SimpleNamespace(consumer="auth")),),
        "other": (SimpleNamespace(consumer="auth", expose=None),),
    }
    stage_files(owner, {"configuration.yml": "content"})
    entries = owner.manager.artifact_index.load()
    assert len(entries) == 1
    assert next(iter(entries.values()))["producers"] == ["app", "other"]


def test_reusing_prepared_tree_refreshes_source_provenance(owner):
    owner.manager.integration_snapshot = {
        "first": (SimpleNamespace(consumer="auth", expose=None),),
    }
    first = stage_files(owner, {"configuration.yml": "unchanged"})
    assert next(iter(owner.manager.artifact_index.load().values()))["producers"] == ["first"]
    owner.manager.integration_snapshot = {
        "second": (SimpleNamespace(consumer="auth", expose=None),),
    }
    reused = stage_files(owner, {"configuration.yml": "unchanged"})
    assert reused == first
    assert next(iter(owner.manager.artifact_index.load().values()))["producers"] == ["second"]


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


def test_acl_change_recreates_only_its_actual_consumer(owner):
    old = context(owner)
    old.write_files(owner, {"base.yml": "same", "acl.yml": "deny"})
    old.compose_model = bind_prepared_files(old, model(owner), {})
    AppliedServiceModels(owner.manager, old.compose_model).record(("auth", "admin"))
    new = context(owner)
    new.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    store = AppliedServiceModels(owner.manager, model(owner))
    new_model = bind_prepared_files(new, model(owner), store.previous)
    store.set_model(new_model)
    assert store.changed_services == {"auth"}
    assert new_model["services"]["admin"] == old.compose_model["services"]["admin"]
    assert new_model["services"]["auth"]["volumes"][0] == old.compose_model["services"]["auth"]["volumes"][0]
    assert new_model["services"]["auth"]["volumes"][1] != old.compose_model["services"]["auth"]["volumes"][1]


def test_inactive_profile_input_is_not_rebound_or_required_in_active_prepared_tree(owner):
    ctx = context(owner, full=True)
    ctx.target_services = ("auth",)
    candidate = ctx.write_files(owner, {"base.yml": "current", "acl.yml": "allow"})
    desired = model(owner)
    inactive = desired["services"]["admin"]
    inactive["profiles"] = ["debug"]
    inactive["volumes"][0]["source"] = str(owner.get_app_path("generated/current/old.yml"))
    bound = bind_prepared_files(ctx, desired, {})
    assert bound["services"]["admin"] == inactive
    assert bound["services"]["auth"]["volumes"][0]["source"] == str(candidate / "base.yml")


def test_no_change_reuses_identical_paths_and_model(owner):
    first = context(owner)
    first.write_files(owner, {"base.yml": "same", "acl.yml": "same"})
    original = bind_prepared_files(first, model(owner), {})
    AppliedServiceModels(owner.manager, original).record(("auth", "admin"))
    later = context(owner)
    later.write_files(owner, {"base.yml": "same", "acl.yml": "same"})
    store = AppliedServiceModels(owner.manager, model(owner))
    store.set_model(bind_prepared_files(later, model(owner), store.previous))
    assert not store.changed_services


def test_missing_prepared_mount_fails_before_any_current_switch(owner):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "base"})
    with pytest.raises(ContainerError, match="Missing prepared input"):
        bind_prepared_files(ctx, model(owner), {})
    assert not owner.get_app_path("generated/current").exists()


def test_confirmed_model_and_previous_snapshot_are_retained(owner):
    first = context(owner)
    first.write_files(owner, {"base.yml": "old", "acl.yml": "old"})
    first.compose_model = bind_prepared_files(first, model(owner), {})
    AppliedServiceModels(owner.manager, first.compose_model).record(("auth", "admin"))
    later = context(owner)
    later.write_files(owner, {"base.yml": "new", "acl.yml": "new"})
    store = AppliedServiceModels(owner.manager, model(owner))
    later.compose_model = bind_prepared_files(later, model(owner), store.previous)
    later.service_models = store
    store.set_model(later.compose_model)
    abandoned = stage_files(owner, {"base.yml": "abandoned", "acl.yml": "abandoned"})
    store.record(("auth", "admin"))
    publish_prepared_files(later, ("auth", "admin"))
    prune_prepared_files(later, store)
    assert first.prepared_dirs["auth"].exists()
    assert later.prepared_dirs["auth"].exists()
    assert not abandoned.exists()
    assert owner.get_app_path("generated/current").resolve() == later.prepared_dirs["auth"]


def test_unselected_legacy_consumer_keeps_current_pointer(owner):
    old = stage_files(owner, {"base.yml": "old", "acl.yml": "old"})
    owner.get_app_path("generated/current").symlink_to(old.name)
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "new", "acl.yml": "new"})
    ctx.compose_model = bind_prepared_files(ctx, model(owner), {})
    ctx.initial_running_services = {"auth", "admin"}
    legacy = {"services": {"admin": {"volumes": [{"type": "bind", "target": "/generated",
                                                  "source": str(owner.get_app_path("generated"))}]}}}
    ctx.service_models.previous = {"admin": yaml.safe_dump(legacy)}
    publish_prepared_files(ctx, ("auth",))
    assert owner.get_app_path("generated/current").resolve() == old
    assert Path(ctx.compose_model["services"]["auth"]["volumes"][0]["source"]).read_text() == "new"


def _apply_prepared(owner, value, services, running=(), desired_services=(), full=False, declared_services=()):
    ctx = context(owner, full=full)
    ctx.target_services = tuple(services)
    ctx.initial_running_services = frozenset(running)
    ctx.write_files(owner, {"base.yml": value, "acl.yml": value})
    desired = model(owner)
    desired["services"] = {name: desired["services"][name] for name in desired_services or services}
    owner.services = {name: {} for name in declared_services or desired["services"]}
    models = AppliedServiceModels(owner.manager, desired, retained_services=running)
    ctx.compose_model = bind_prepared_files(ctx, desired, models.previous)
    ctx.service_models = models
    models.set_model(ctx.compose_model)
    models.record(services)
    publish_prepared_files(ctx, services)
    prune_prepared_files(ctx, models)
    return ctx


def test_running_removed_service_retains_inputs_across_generations(owner):
    first = _apply_prepared(owner, "A", ("auth", "admin"))
    original = first.prepared_dirs["auth"]
    generations = []
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"))
        generations.append(ctx.prepared_dirs["auth"])
        assert (original / "base.yml").read_text() == "A"
        assert "admin" in ctx.service_models.previous
        assert "admin" not in ctx.service_models.current
        assert "admin" not in ctx.service_models.changed_services
        assert owner.get_app_path("generated/current").resolve() == ctx.prepared_dirs["auth"]
    assert not generations[0].exists()
    assert generations[1].exists()
    retained_entry = str((original / "base.yml").relative_to(owner.manager.data_path))
    assert retained_entry in owner.manager.artifact_index.load()

    # Once the removed service is no longer running, its old snapshot is not a live reference.
    stopped = _apply_prepared(owner, "E", ("auth",), running=("auth",))
    assert "admin" not in stopped.service_models.previous
    assert not original.exists()
    assert retained_entry not in owner.manager.artifact_index.load()


@pytest.mark.parametrize("source", ["generated", "generated/current", "generated/current/base.yml"])
def test_running_removed_legacy_service_keeps_current_pointer(owner, source):
    original = stage_files(owner, {"base.yml": "A", "acl.yml": "A"})
    owner.get_app_path("generated/current").symlink_to(original.name)
    legacy = model(owner)
    legacy["services"]["admin"]["volumes"] = [
        {"type": "bind", "source": str(owner.get_app_path(source)), "target": "/legacy"}]
    AppliedServiceModels(owner.manager, legacy).record(("auth", "admin"))

    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"))
        assert owner.get_app_path("generated/current").resolve() == original
        assert owner.get_app_path("generated/current/base.yml").read_text() == "A"
        assert Path(ctx.compose_model["services"]["auth"]["volumes"][0]["source"]).read_text() == value


@pytest.mark.parametrize("desired_services", [("auth",), ("auth", "admin")])
def test_unknown_unselected_service_preserves_current_and_all_generated_trees(owner, desired_services):
    first = _apply_prepared(owner, "A", ("auth",))
    original = first.prepared_dirs["auth"]
    unknown_input = stage_files(owner, {"base.yml": "orphan input", "acl.yml": "orphan input"})
    unknown_entry = str((unknown_input / "base.yml").relative_to(owner.manager.data_path))
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"),
                              desired_services=desired_services)
        assert ctx.service_models.untracked_services == {"admin"}
        assert owner.get_app_path("generated/current").resolve() == original
        assert owner.get_app_path("generated/current/base.yml").read_text() == "A"
        assert (unknown_input / "base.yml").read_text() == "orphan input"
        assert unknown_entry in owner.manager.artifact_index.load()
        assert Path(ctx.compose_model["services"]["auth"]["volumes"][0]["source"]).read_text() == value

    stopped = _apply_prepared(owner, "E", ("auth",), running=("auth",))
    assert not stopped.service_models.untracked_services
    assert owner.get_app_path("generated/current").resolve() == stopped.prepared_dirs["auth"]
    assert not original.exists()
    assert not unknown_input.exists()
    assert unknown_entry not in owner.manager.artifact_index.load()


def test_successful_first_migration_publishes_and_prunes_prepared_files(owner):
    original = stage_files(owner, {"base.yml": "A", "acl.yml": "A"})
    owner.get_app_path("generated/current").symlink_to(original.name)
    ctx = _apply_prepared(owner, "B", ("auth",), running=("auth",))
    assert ctx.service_models.untracked_services == {"auth"}
    assert owner.get_app_path("generated/current").resolve() == ctx.prepared_dirs["auth"]
    assert not original.exists()
    stored = AppliedServiceModels(owner.manager, model(owner), retained_services=("auth",))
    assert not stored.untracked_services


@pytest.mark.parametrize("known_orphan", [False, True])
def test_full_apply_retires_orphan_before_publishing_and_cleanup(owner, known_orphan):
    first = _apply_prepared(owner, "A", ("auth",))
    original = first.prepared_dirs["auth"]
    orphan_input = stage_files(owner, {"base.yml": "orphan input", "acl.yml": "orphan input"})
    if known_orphan:
        AppliedServiceModels(owner.manager, model(owner)).record(("admin",))

    ctx = _apply_prepared(owner, "B", ("auth",), running=("auth", "admin"), full=True)
    assert owner.get_app_path("generated/current").resolve() == ctx.prepared_dirs["auth"]
    assert owner.get_app_path("generated/current/base.yml").read_text() == "B"
    assert original.exists()
    if known_orphan:
        assert "admin" in ctx.service_models.previous
    else:
        assert ctx.service_models.untracked_services == {"admin"}
        assert not orphan_input.exists()
        orphan_entry = str((orphan_input / "base.yml").relative_to(owner.manager.data_path))
        assert orphan_entry not in owner.manager.artifact_index.load()


@pytest.mark.parametrize("source", ["generated", "generated/current", "generated/current/base.yml"])
def test_full_apply_preserves_defined_unapplied_legacy_consumer(owner, source):
    original = stage_files(owner, {"base.yml": "A", "acl.yml": "A"})
    owner.get_app_path("generated/current").symlink_to(original.name)
    legacy = model(owner)
    legacy["services"]["admin"]["volumes"] = [
        {"type": "bind", "source": str(owner.get_app_path(source)), "target": "/legacy"}]
    AppliedServiceModels(owner.manager, legacy).record(("auth", "admin"))
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"), full=True,
                              declared_services=("auth", "admin"))
        assert owner.get_app_path("generated/current").resolve() == original
        assert owner.get_app_path("generated/current/base.yml").read_text() == "A"
        assert Path(ctx.compose_model["services"]["auth"]["volumes"][0]["source"]).read_text() == value


def test_full_apply_retains_unknown_running_disabled_profile_inputs(owner):
    first = _apply_prepared(owner, "A", ("auth",))
    original = first.prepared_dirs["auth"]
    unknown_input = stage_files(owner, {"base.yml": "untracked", "acl.yml": "untracked"})
    entry = str((unknown_input / "base.yml").relative_to(owner.manager.data_path))
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"), full=True,
                              declared_services=("auth", "admin"))
        assert ctx.service_models.untracked_services == {"admin"}
        assert owner.get_app_path("generated/current").resolve() == original
        assert unknown_input.exists() and original.exists()
        assert entry in owner.manager.artifact_index.load()
        assert Path(ctx.compose_model["services"]["auth"]["volumes"][0]["source"]).read_text() == value

    removed = _apply_prepared(owner, "E", ("auth",), running=("auth", "admin"), full=True)
    assert owner.get_app_path("generated/current").resolve() == removed.prepared_dirs["auth"]
    assert not unknown_input.exists()


def test_snapshot_restore_does_not_undo_successful_sibling(owner):
    original = {"services": {"a": {"image": "a:old"}, "b": {"image": "b:old"}}}
    desired = {"services": {"a": {"image": "a:new"}, "b": {"image": "b:new"}}}
    AppliedServiceModels(owner.manager, original).record(("a", "b"))
    state = AppliedServiceModels(owner.manager, desired)
    state.record(("a",))
    state.record(("b",))
    state.restore(("b",))
    current = AppliedServiceModels(owner.manager, desired)
    assert yaml.safe_load(current.previous["a"])["services"]["a"]["image"] == "a:new"
    assert yaml.safe_load(current.previous["b"])["services"]["b"]["image"] == "b:old"
    assert current.changed_services == {"b"}
