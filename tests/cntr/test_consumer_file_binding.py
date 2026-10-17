#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extra consumers can change only exactly declared generated-tree sources."""
from copy import deepcopy

import pytest

from linktools.cntr.artifacts import AppliedServiceModels, stage_files, validate_consumer_config
from linktools.cntr.errors import ContainerError
from test_prepared_files import context, model, owner


def saved(owner):
    old = stage_files(owner, {"base.yml": "same", "acl.yml": "deny"})
    applied = model(owner, old)
    applied["services"]["auth"].update(environment={"VALUE": "applied"}, ports=[{"target": 80, "published": "8080"}])
    AppliedServiceModels(owner.manager, applied).record(("auth", "admin"))
    return old, applied


def changed_model(owner, applied, *, files=None, mode=0o600):
    path = owner.prepare_files(files or {"base.yml": "same", "acl.yml": "allow"}, mode=mode)
    desired = deepcopy(applied)
    for spec in desired["services"].values():
        spec["volumes"][0]["source"] = str(path)
    return path, desired


def validate(owner, desired, services=("auth",)):
    return validate_consumer_config(context(owner), desired, AppliedServiceModels(owner.manager, desired), services)


def test_declared_tree_changes_are_returned_without_rewriting_models(owner):
    _, applied = saved(owner)
    path, desired = changed_model(owner, applied)
    original = deepcopy(desired)
    store = AppliedServiceModels(owner.manager, desired)
    ctx = context(owner)
    ctx.target_services = ("admin",)
    assert validate_consumer_config(ctx, desired, store, ("auth", "auth")) == ("auth",)
    assert desired == original
    assert store.previous_model("auth") == applied
    assert ctx.target_services == ("admin",)
    assert not path.exists()


def test_permission_change_is_visible_through_exact_tree_source(owner):
    old, applied = saved(owner)
    path, desired = changed_model(owner, applied, files={"base.yml": "same", "acl.yml": "deny"}, mode=0o640)
    assert path != old
    assert validate(owner, desired, ("auth", "admin")) == ("auth", "admin")


@pytest.mark.parametrize("field,value", [
    ("image", "auth:pending"), ("environment", {"VALUE": "pending"}),
    ("ports", []), ("build", {"context": "/pending"}), ("command", ["pending"]),
    ("labels", {"pending": "label"}), ("depends_on", {"admin": {"condition": "service_started"}}),
])
def test_changed_consumer_rejects_other_pending_service_settings(owner, field, value):
    _, applied = saved(owner)
    _, desired = changed_model(owner, applied)
    desired["services"]["auth"][field] = value
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


@pytest.mark.parametrize("field,value", [("target", "/different"), ("read_only", False),
                                         ("type", "volume"), ("bind", {"propagation": "rshared"})])
def test_generated_mount_metadata_is_not_rewritten(owner, field, value):
    _, applied = saved(owner)
    _, desired = changed_model(owner, applied)
    desired["services"]["auth"]["volumes"][0][field] = value
    if field == "type":
        # It no longer consumes a declared bind tree, so it is not an extra consumer.
        assert validate(owner, desired) == ()
    else:
        with pytest.raises(ContainerError, match="outside.*explicitly select"):
            validate(owner, desired)


def test_consumer_with_unchanged_generated_source_ignores_unrelated_pending_changes(owner):
    old, applied = saved(owner)
    path, desired = changed_model(owner, applied, files={"base.yml": "same", "acl.yml": "deny"})
    assert path == old
    desired["services"]["auth"]["image"] = "pending"
    desired["name"] = "pending-project"
    assert validate(owner, desired) == ()


def test_sidecar_without_declared_generated_mount_is_ignored(owner):
    _, applied = saved(owner)
    _, desired = changed_model(owner, applied)
    desired["services"]["admin"] = {"image": "redis:pending", "environment": {"PENDING": "yes"}}
    assert validate(owner, desired, ("admin",)) == ()


def test_consumer_without_saved_model_requires_explicit_selection(owner):
    path = owner.prepare_files({"base.yml": "new"})
    with pytest.raises(ContainerError, match="No applied Compose model.*auth.*explicitly select"):
        validate(owner, model(owner, path))


@pytest.mark.parametrize("legacy", ["no-tree", "wrong-owner", "wrong-kind", "unknown-path"])
def test_old_source_must_have_exact_tree_provenance(owner, legacy):
    old, applied = saved(owner)
    if legacy in ("no-tree", "wrong-owner", "wrong-kind"):
        for entry in owner.manager.artifact_index.values.values():
            if entry.get("kind") != "generated-config":
                continue
            if legacy == "no-tree":
                entry.pop("tree")
            elif legacy == "wrong-owner":
                entry["container"] = "other"
            else:
                entry["kind"] = "compose"
    else:
        applied["services"]["auth"]["volumes"][0]["source"] = str(old.parent / "current")
        AppliedServiceModels(owner.manager, applied).record(("auth",))
    _, desired = changed_model(owner, applied)
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


def test_nonregistered_new_file_path_is_not_inferred_as_generated(owner):
    _, applied = saved(owner)
    path, desired = changed_model(owner, applied)
    desired["services"]["auth"]["volumes"][0]["source"] = str(path / "undeclared.yml")
    assert validate(owner, desired) == ()


def test_user_owned_mount_change_alongside_generated_change_requires_selection(owner):
    _, applied = saved(owner)
    applied["services"]["auth"]["volumes"].append({"type": "bind", "source": "/user-owned", "target": "/manual"})
    AppliedServiceModels(owner.manager, applied).record(("auth",))
    _, desired = changed_model(owner, applied)
    desired["services"]["auth"]["volumes"][1]["source"] = "/different"
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


def support_model(owner):
    _, applied = saved(owner)
    spec = applied["services"]["auth"]
    spec.update(networks={"auth-net": {}}, secrets=[{"source": "auth-secret", "target": "secret"}],
                configs=["auth-config"])
    spec["volumes"].append({"type": "volume", "source": "auth-data", "target": "/data"})
    applied.update(networks={"auth-net": {"name": "applied-net"}},
                   volumes={"auth-data": {"name": "applied-data"}},
                   secrets={"auth-secret": {"file": "/applied/secret"}},
                   configs={"auth-config": {"file": "/applied/config"}}, name="project")
    AppliedServiceModels(owner.manager, applied).record(("auth",))
    return applied


@pytest.mark.parametrize("category,name", [("networks", "auth-net"), ("volumes", "auth-data"),
                                           ("secrets", "auth-secret"), ("configs", "auth-config")])
def test_referenced_support_changes_require_explicit_selection(owner, category, name):
    applied = support_model(owner)
    _, desired = changed_model(owner, applied)
    desired[category][name] = {"external": True}
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


def test_project_support_changes_require_explicit_selection(owner):
    applied = support_model(owner)
    _, desired = changed_model(owner, applied)
    desired["name"] = "different"
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


def test_unreferenced_support_changes_do_not_block_generated_update(owner):
    applied = support_model(owner)
    _, desired = changed_model(owner, applied)
    desired["networks"]["unrelated"] = {"name": "pending-net"}
    assert validate(owner, desired) == ("auth",)


def test_exact_file_mounts_follow_declared_tree_identity(owner):
    old, applied = saved(owner)
    applied["services"]["auth"]["volumes"][0]["source"] = str(old / "acl.yml")
    applied["services"]["admin"]["volumes"][0]["source"] = str(old / "base.yml")
    AppliedServiceModels(owner.manager, applied).record(("auth", "admin"))
    path = owner.prepare_files({"base.yml": "same", "acl.yml": "allow"})
    desired = deepcopy(applied)
    desired["services"]["auth"]["volumes"][0]["source"] = str(path / "acl.yml")
    desired["services"]["admin"]["volumes"][0]["source"] = str(path / "base.yml")
    assert validate(owner, desired, ("auth", "admin")) == ("auth", "admin")


@pytest.mark.parametrize("old_name,new_name", [("base.yml", "acl.yml"), ("", "base.yml"), ("base.yml", "")])
def test_cannot_swap_distinct_generated_files_or_root_and_file(owner, old_name, new_name):
    old, applied = saved(owner)
    applied["services"]["auth"]["volumes"][0]["source"] = str(old / old_name)
    AppliedServiceModels(owner.manager, applied).record(("auth",))
    path, desired = changed_model(owner, applied)
    desired["services"]["auth"]["volumes"][0]["source"] = str(path / new_name)
    with pytest.raises(ContainerError, match="outside.*explicitly select"):
        validate(owner, desired)


def test_nested_generated_file_identity_is_exact(owner):
    old = stage_files(owner, {"nested/config.yml": "old"})
    applied = model(owner, old / "nested/config.yml")
    AppliedServiceModels(owner.manager, applied).record(("auth", "admin"))
    path = owner.prepare_files({"nested/config.yml": "new"})
    desired = model(owner, path / "nested/config.yml")
    assert validate(owner, desired) == ("auth",)
