#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration consumers retain applied settings and receive only owned inputs."""
from copy import deepcopy

import pytest

from linktools.cntr.artifacts import AppliedServiceModels, bind_consumer_files, bind_prepared_files
from linktools.cntr.errors import ContainerError
from test_prepared_files import context, model, owner


def saved(owner, *, directory=False, mode=0o600):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "same", "acl.yml": "deny"}, mode=mode)
    desired = model(owner)
    spec = desired["services"]["auth"]
    spec.update(environment={"VALUE": "applied"}, ports=[{"target": 80, "published": "8080"}])
    if directory:
        spec["volumes"] = [{"type": "bind", "source": str(owner.get_app_path("generated/current")),
                            "target": "/generated", "read_only": True}]
    bound = bind_prepared_files(ctx, desired)
    AppliedServiceModels(owner.manager, bound).record(("auth", "admin"))
    return ctx, bound


@pytest.mark.parametrize("changed", [False, True])
def test_consumer_directory_rebind_preserves_applied_settings(owner, changed):
    old, applied = saved(owner, directory=True)
    desired = deepcopy(applied)
    desired["services"]["auth"].update(image="auth:pending", environment={"VALUE": "pending"}, ports=[])
    original = deepcopy(desired)
    current = context(owner)
    current.target_services = ("admin",)
    candidate = current.write_files(owner, {"base.yml": "same", "acl.yml": "allow" if changed else "deny"})
    store = AppliedServiceModels(owner.manager, desired)

    result = bind_consumer_files(current, desired, store, ("auth",))

    expected = deepcopy(applied["services"]["auth"])
    expected["volumes"][0]["source"] = str(candidate)
    assert result["services"]["auth"] == expected
    assert result["services"]["admin"] == desired["services"]["admin"]
    assert desired == original
    assert store.previous_model("auth") == applied
    assert current.target_services == ("admin",)
    store.set_model(result)
    assert store.changed_services == ({"auth"} if changed else set())
    assert candidate != old.prepared_dirs["auth"] if changed else candidate == old.prepared_dirs["auth"]


def test_consumer_single_files_keep_identical_inputs(owner):
    _, applied = saved(owner)
    current = context(owner)
    candidate = current.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    desired = model(owner)
    desired["services"]["auth"]["environment"] = {"VALUE": "pending"}
    store = AppliedServiceModels(owner.manager, desired)

    result = bind_consumer_files(current, desired, store, ("auth", "admin"))

    assert result["services"]["admin"] == applied["services"]["admin"]
    mounts = result["services"]["auth"]["volumes"]
    assert mounts[0] == applied["services"]["auth"]["volumes"][0]
    assert mounts[1]["source"] == str(candidate / "acl.yml")
    assert result["services"]["auth"]["environment"] == {"VALUE": "applied"}


def test_consumer_permission_change_uses_new_single_file(owner):
    _, applied = saved(owner)
    current = context(owner)
    candidate = current.write_files(owner, {"base.yml": "same", "acl.yml": "deny"}, mode=0o640)
    store = AppliedServiceModels(owner.manager, model(owner))

    result = bind_consumer_files(current, model(owner), store, ("auth",))

    assert result["services"]["auth"]["volumes"][0]["source"] == str(candidate / "base.yml")
    assert result["services"]["auth"]["volumes"][0] != applied["services"]["auth"]["volumes"][0]


def test_consumer_missing_applied_model_requires_explicit_selection(owner):
    current = context(owner)
    current.write_files(owner, {"base.yml": "same", "acl.yml": "deny"})
    store = AppliedServiceModels(owner.manager, model(owner))
    with pytest.raises(ContainerError, match="No applied Compose model.*auth.*explicitly select"):
        bind_consumer_files(current, model(owner), store, ("auth",))


def test_consumer_legacy_current_binding_and_unowned_mounts(owner):
    applied = model(owner)
    mounts = applied["services"]["auth"]["volumes"]
    mounts.extend([
        {"type": "bind", "source": str(owner.get_app_path("generated/manual/file")), "target": "/manual"},
        {"type": "bind", "source": str(owner.get_app_path("other/current/file")), "target": "/other"},
    ])
    AppliedServiceModels(owner.manager, applied).record(("auth",))
    current = context(owner)
    candidate = current.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    store = AppliedServiceModels(owner.manager, model(owner))

    result = bind_consumer_files(current, model(owner), store, ("auth",))

    rebound = result["services"]["auth"]["volumes"]
    assert rebound[0]["source"] == str(candidate / "base.yml")
    assert rebound[1]["source"] == str(candidate / "acl.yml")
    assert rebound[2:] == mounts[2:]


def support_model(owner):
    _, applied = saved(owner)
    spec = applied["services"]["auth"]
    spec.update(networks={"auth-net": {"aliases": ["applied"]}},
                secrets=[{"source": "auth-secret", "target": "secret"}], configs=["auth-config"])
    spec["volumes"].append({"type": "volume", "source": "auth-data", "target": "/data"})
    applied.update(networks={"auth-net": {"name": "applied-net"}, "unrelated-old": {"name": "old"}},
                   volumes={"auth-data": {"name": "applied-data"}},
                   secrets={"auth-secret": {"file": "/applied/secret"}},
                   configs={"auth-config": {"file": "/applied/config"}}, name="project")
    AppliedServiceModels(owner.manager, applied).record(("auth",))
    return applied


def test_consumer_retains_referenced_support_without_importing_unrelated_old_support(owner):
    applied = support_model(owner)
    desired = model(owner)
    desired["networks"] = {"unrelated-new": {"name": "new"}}
    current = context(owner)
    current.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    store = AppliedServiceModels(owner.manager, desired)

    result = bind_consumer_files(current, desired, store, ("auth",))

    assert result["networks"] == {"auth-net": applied["networks"]["auth-net"],
                                  "unrelated-new": {"name": "new"}}
    for category in ("volumes", "secrets", "configs"):
        assert result[category] == applied[category]
    assert result["name"] == "project"
    assert result["services"]["auth"]["image"] == applied["services"]["auth"]["image"]
    assert result["services"]["auth"]["networks"] == applied["services"]["auth"]["networks"]


@pytest.mark.parametrize("category,name", [("networks", "auth-net"), ("volumes", "auth-data"),
                                           ("secrets", "auth-secret"), ("configs", "auth-config")])
def test_consumer_support_conflict_requires_explicit_selection(owner, category, name):
    applied = support_model(owner)
    desired = deepcopy(applied)
    desired[category][name] = {"external": True}
    current = context(owner)
    store = AppliedServiceModels(owner.manager, desired)
    with pytest.raises(ContainerError, match="conflicts.*auth.*explicitly select"):
        bind_consumer_files(current, desired, store, ("auth",))


def test_consumer_project_support_conflict_requires_explicit_selection(owner):
    applied = support_model(owner)
    desired = dict(applied, name="pending-project")
    current = context(owner)
    store = AppliedServiceModels(owner.manager, desired)
    with pytest.raises(ContainerError, match="Compose 'name' conflicts.*auth.*explicitly select"):
        bind_consumer_files(current, desired, store, ("auth",))


def test_consumer_missing_prepared_relative_input_fails_closed(owner):
    _, applied = saved(owner)
    current = context(owner)
    current.write_files(owner, {"base.yml": "same"})
    store = AppliedServiceModels(owner.manager, applied)
    with pytest.raises(ContainerError, match="Missing prepared input.*acl.yml"):
        bind_consumer_files(current, applied, store, ("auth",))
