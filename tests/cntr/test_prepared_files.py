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
    return SimpleNamespace(name="auth", manager=manager, get_app_path=lambda *parts: root.joinpath(*parts),
                           runtime=SimpleNamespace(chmod=lambda path, mode: os.chmod(path, mode)))


def model(owner, base=True, acl=True):
    root = owner.get_app_path("generated/current")
    bindings = lambda names: [{"type": "bind", "source": str(root / name), "target": "/" + name,
                               "read_only": True} for name in names]
    return {"services": {
        "auth": {"image": "auth:new", "volumes": bindings(["base.yml", "acl.yml"])},
        "admin": {"image": "admin:new", "volumes": bindings(["base.yml"])},
    }}


def context(owner):
    ctx = OperationContext(containers=[owner], target_services=("auth", "admin"))
    ctx.initial_running_services = set()
    ctx.service_models = SimpleNamespace(previous={})
    return ctx


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
    assert first.prepared_files["auth"].exists()
    assert later.prepared_files["auth"].exists()
    assert not abandoned.exists()
    assert owner.get_app_path("generated/current").resolve() == later.prepared_files["auth"]


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
