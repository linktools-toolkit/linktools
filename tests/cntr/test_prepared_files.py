#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerManager, Integration, OperationContext
from linktools.cntr.artifacts import (AppliedServiceModels, stage_files, apply_config_revisions,
                                     switch_config_revisions, prune_prepared_files, CONFIG_REVISION_LABEL)
from linktools.cntr.errors import ContainerError
from linktools.cntr.ext import endpoint
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
    ctx.initial_runtime_state = ProjectRuntimeState("test", (), "docker")
    ctx.service_models = SimpleNamespace(previous={}, untracked_services=frozenset())
    owner.get_config_revision = lambda service: (ctx.prepared_dirs[owner.name].name
        if service in owner.services and owner.name in ctx.prepared_dirs else None)
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


def test_acl_change_updates_all_services_using_the_same_generated_tree(owner):
    old = context(owner)
    old.write_files(owner, {"base.yml": "same", "acl.yml": "deny"})
    old.compose_model = apply_config_revisions(old, model(owner))
    AppliedServiceModels(owner.manager, old.compose_model).record(("auth", "admin"))
    new = context(owner)
    candidate = new.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    store = AppliedServiceModels(owner.manager, model(owner))
    new_model = apply_config_revisions(new, model(owner))
    store.set_model(new_model)
    assert store.changed_services == {"auth", "admin"}
    for service in ("auth", "admin"):
        assert new_model["services"][service]["volumes"] == old.compose_model["services"][service]["volumes"]
        assert new_model["services"][service]["labels"][CONFIG_REVISION_LABEL] == candidate.name


def test_inactive_profile_revision_is_not_added_to_active_selection(owner):
    ctx = context(owner, full=True)
    ctx.target_services = ("auth",)
    candidate = ctx.write_files(owner, {"base.yml": "current", "acl.yml": "allow"})
    desired = model(owner)
    inactive = desired["services"]["admin"]
    inactive["profiles"] = ["debug"]
    inactive["volumes"][0]["source"] = str(owner.get_app_path("generated/current/old.yml"))
    revised = apply_config_revisions(ctx, desired)
    assert revised["services"]["admin"] == inactive
    assert revised["services"]["auth"]["labels"][CONFIG_REVISION_LABEL] == candidate.name
    assert revised["services"]["auth"]["volumes"] == desired["services"]["auth"]["volumes"]


def test_no_change_reuses_identical_revision_and_model(owner):
    first = context(owner)
    first.write_files(owner, {"base.yml": "same", "acl.yml": "same"})
    original = apply_config_revisions(first, model(owner))
    AppliedServiceModels(owner.manager, original).record(("auth", "admin"))
    later = context(owner)
    later.write_files(owner, {"base.yml": "same", "acl.yml": "same"})
    store = AppliedServiceModels(owner.manager, model(owner))
    store.set_model(apply_config_revisions(later, model(owner)))
    assert not store.changed_services
    assert not owner.get_app_path("generated/current").exists()


def test_missing_prepared_generation_fails_before_any_current_switch(owner):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "base"})
    with pytest.raises(ContainerError, match="Missing generated configuration revision"):
        switch_config_revisions(ctx, {owner.name: "f" * 64})
    assert not owner.get_app_path("generated/current").exists()


def _apply_prepared(owner, value, services, running=(), desired_services=(), full=False,
                    declared_services=(), activate=True):
    ctx = context(owner, full=full)
    ctx.target_services = tuple(services)
    ctx.initial_runtime_state = ProjectRuntimeState("test", tuple(
        ServiceRuntimeState(logical_containers=(owner.name,), service=name, runtime_name=name,
                            state="running", health=None, image=None, exit_code=0, labels={}) for name in running), "docker")
    candidate = ctx.write_files(owner, {"base.yml": value, "acl.yml": value})
    desired = model(owner)
    desired["services"] = {name: desired["services"][name] for name in desired_services or services}
    owner.services = {name: {} for name in declared_services or desired["services"]}
    models = AppliedServiceModels(owner.manager, desired, retained_services=running)
    ctx.compose_model = apply_config_revisions(ctx, desired)
    ctx.service_models = models
    models.set_model(ctx.compose_model)
    if activate:
        switch_config_revisions(ctx, {owner.name: candidate.name})
        models.record(services)
    prune_prepared_files(ctx, models)
    return ctx


def test_confirmed_and_rollback_revisions_are_retained(owner):
    first = _apply_prepared(owner, "old", ("auth", "admin"))
    abandoned = stage_files(owner, {"base.yml": "abandoned", "acl.yml": "abandoned"})
    later = _apply_prepared(owner, "new", ("auth", "admin"))
    assert first.prepared_dirs["auth"].exists()
    assert later.prepared_dirs["auth"].exists()
    assert not abandoned.exists()
    assert owner.get_app_path("generated/current").resolve() == later.prepared_dirs["auth"]


def test_combined_site_file_switch_preserves_previous_generation_and_can_restore_it(owner):
    native_model = {"services": {"auth": {"image": "nginx:test", "volumes": [{
        "type": "bind", "source": str(owner.get_app_path("generated/current")),
        "target": "/etc/nginx/managed", "read_only": True,
    }]}}}
    old = context(owner)
    old.target_services = ("auth",)
    original = old.write_files(owner, {"nginx.conf": "include sites/web.conf;",
                                      "sites/web.conf": "server { server_name old.test; }"})
    old.compose_model = apply_config_revisions(old, native_model)
    AppliedServiceModels(owner.manager, old.compose_model).record(("auth",))
    switch_config_revisions(old, {owner.name: original.name})
    custom = owner.get_app_path("custom.conf")
    custom.write_text("user owned")
    current = context(owner)
    current.target_services = ("auth",)
    candidate = current.write_files(owner, {"nginx.conf": "include sites/generated.conf;",
                                           "sites/generated.conf": "server { server_name new.test; }"})
    store = AppliedServiceModels(owner.manager, native_model)
    current.compose_model = apply_config_revisions(current, native_model)
    store.set_model(current.compose_model)
    assert store.changed_services == {"auth"}
    assert owner.get_app_path("generated/current").resolve() == original
    previous = switch_config_revisions(current, {owner.name: candidate.name})
    assert previous == {owner.name: original.name}
    assert owner.get_app_path("generated/current").resolve() == candidate
    assert not (candidate / "sites/web.conf").exists()
    assert (original / "sites/web.conf").read_text() == "server { server_name old.test; }"
    assert current.compose_model["services"]["auth"]["volumes"] == native_model["services"]["auth"]["volumes"]
    switch_config_revisions(current, previous)
    store.restore(("auth",))
    assert owner.get_app_path("generated/current").resolve() == original
    assert store.previous_model("auth")["services"]["auth"]["labels"][CONFIG_REVISION_LABEL] == original.name
    assert custom.read_text() == "user owned"


def test_running_removed_service_retains_revision_across_generations(owner):
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
    assert not generations[0].exists()
    assert generations[1].exists()
    entry = str((original / "base.yml").relative_to(owner.manager.data_path))
    assert entry in owner.manager.artifact_index.load()
    stopped = _apply_prepared(owner, "E", ("auth",), running=("auth",))
    assert "admin" not in stopped.service_models.previous
    assert not original.exists()
    assert entry not in owner.manager.artifact_index.load()


@pytest.mark.parametrize("source", ["generated", "generated/current", "generated/current/base.yml"])
@pytest.mark.parametrize("full", [False, True])
def test_unresolved_legacy_service_preserves_current_and_prevents_pruning(owner, source, full):
    original = stage_files(owner, {"base.yml": "A", "acl.yml": "A"})
    owner.get_app_path("generated/current").symlink_to(original.name)
    legacy = model(owner)
    legacy["services"]["admin"]["volumes"] = [
        {"type": "bind", "source": str(owner.get_app_path(source)), "target": "/legacy"}]
    AppliedServiceModels(owner.manager, legacy).record(("auth", "admin"))
    unknown = stage_files(owner, {"base.yml": "untracked", "acl.yml": "untracked"})
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"), full=full,
                              declared_services=("auth", "admin"), activate=False)
        assert owner.get_app_path("generated/current").resolve() == original
        assert owner.get_app_path("generated/current/base.yml").read_text() == "A"
        assert unknown.exists() and original.exists()
        assert (ctx.prepared_dirs["auth"] / "base.yml").read_text() == value


@pytest.mark.parametrize("desired_services", [("auth",), ("auth", "admin")])
@pytest.mark.parametrize("full", [False, True])
def test_unknown_running_service_preserves_all_generations_until_removed(owner, desired_services, full):
    first = _apply_prepared(owner, "A", ("auth",))
    original = first.prepared_dirs["auth"]
    unknown = stage_files(owner, {"base.yml": "untracked", "acl.yml": "untracked"})
    entry = str((unknown / "base.yml").relative_to(owner.manager.data_path))
    for value in ("B", "C", "D"):
        ctx = _apply_prepared(owner, value, ("auth",), running=("auth", "admin"),
                              desired_services=desired_services, full=full,
                              declared_services=("auth", "admin"), activate=False)
        assert ctx.service_models.untracked_services == {"admin"}
        assert owner.get_app_path("generated/current").resolve() == original
        assert unknown.exists() and original.exists()
        assert entry in owner.manager.artifact_index.load()
        assert (ctx.prepared_dirs["auth"] / "base.yml").read_text() == value
    removed = _apply_prepared(owner, "E", ("auth",), running=("auth", "admin") if full else ("auth",), full=full)
    assert owner.get_app_path("generated/current").resolve() == removed.prepared_dirs["auth"]
    assert not unknown.exists()
    assert entry not in owner.manager.artifact_index.load()


def test_successful_first_revision_migration_publishes_and_prunes(owner):
    original = stage_files(owner, {"base.yml": "A", "acl.yml": "A"})
    owner.get_app_path("generated/current").symlink_to(original.name)
    ctx = _apply_prepared(owner, "B", ("auth",), running=("auth",))
    assert ctx.service_models.untracked_services == {"auth"}
    assert owner.get_app_path("generated/current").resolve() == ctx.prepared_dirs["auth"]
    assert not original.exists()
    stored = AppliedServiceModels(owner.manager, model(owner), retained_services=("auth",))
    assert not stored.untracked_services


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
