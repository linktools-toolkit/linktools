#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision labels leave source paths stable; pointer switching remains reversible."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from linktools.cntr.artifacts import (CONFIG_REVISION_LABEL, apply_config_revisions,
                                     switch_config_revisions, stage_files)
from linktools.cntr.errors import ContainerError
from test_prepared_files import context, model, owner


def test_revision_injection_preserves_mounts_labels_and_input_model(owner):
    ctx = context(owner)
    candidate = ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    desired = model(owner)
    desired["services"]["auth"]["labels"] = {"user.label": "retained"}
    original = deepcopy(desired)
    revised = apply_config_revisions(ctx, desired)
    assert desired == original
    for name in ("auth", "admin"):
        assert revised["services"][name]["volumes"] == desired["services"][name]["volumes"]
        assert revised["services"][name]["labels"][CONFIG_REVISION_LABEL] == candidate.name
    assert revised["services"]["auth"]["labels"]["user.label"] == "retained"
    assert not owner.get_app_path("generated/current").exists()


def test_absent_revision_does_not_add_label_or_infer_from_mounts(owner):
    ctx = context(owner)
    desired = model(owner)
    assert apply_config_revisions(ctx, desired) == desired


def test_explicit_service_revision_scope_overrides_context_selection(owner):
    ctx = context(owner)
    ctx.target_services = ("auth",)
    candidate = ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    desired = model(owner)
    revised = apply_config_revisions(ctx, desired, ("admin",))
    assert revised["services"]["auth"] == desired["services"]["auth"]
    assert revised["services"]["admin"]["labels"][CONFIG_REVISION_LABEL] == candidate.name


def test_shared_service_cannot_hide_conflicting_revision(owner):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    ctx.project_containers.append(SimpleNamespace(name="overlay", services={"auth": {}},
                                                  get_config_revision=lambda name: "different"))
    with pytest.raises(ContainerError, match="Conflicting configuration revisions.*auth"):
        apply_config_revisions(ctx, model(owner))


def test_generated_revision_must_identify_prepared_tree(owner):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "same"})
    owner.get_config_revision = lambda service: "opaque-mismatch"
    with pytest.raises(ContainerError, match="must identify the prepared tree"):
        apply_config_revisions(ctx, model(owner))


def test_external_revision_without_prepared_files_remains_opaque(owner):
    ctx = context(owner)
    owner.get_config_revision = lambda service: "version:external/7"
    desired = model(owner)
    result = apply_config_revisions(ctx, desired)
    assert result["services"]["auth"]["labels"][CONFIG_REVISION_LABEL] == "version:external/7"
    assert result["services"]["auth"]["volumes"] == desired["services"]["auth"]["volumes"]


def test_generated_owner_requires_at_least_one_selected_service_revision(owner):
    ctx = context(owner)
    ctx.write_files(owner, {"base.yml": "same"})
    owner.get_config_revision = lambda service: None
    with pytest.raises(ContainerError, match="must declare get_config_revision"):
        apply_config_revisions(ctx, model(owner))


def test_unconfigured_sidecar_does_not_need_its_own_revision(owner):
    ctx = context(owner)
    candidate = ctx.write_files(owner, {"base.yml": "same"})
    owner.get_config_revision = lambda service: candidate.name if service == "auth" else None
    result = apply_config_revisions(ctx, model(owner))
    assert result["services"]["auth"]["labels"][CONFIG_REVISION_LABEL] == candidate.name
    assert "labels" not in result["services"]["admin"]


def test_switch_can_restore_absent_pointer_without_removing_generations(owner):
    ctx = context(owner)
    candidate = ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    previous = switch_config_revisions(ctx, {owner.name: candidate.name})
    assert previous == {owner.name: None}
    assert owner.get_app_path("generated/current").resolve() == candidate
    switch_config_revisions(ctx, previous)
    assert not owner.get_app_path("generated/current").is_symlink()
    assert candidate.is_dir()


def test_switch_same_revision_is_noop(owner):
    ctx = context(owner)
    candidate = ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    switch_config_revisions(ctx, {owner.name: candidate.name})
    current = owner.get_app_path("generated/current")
    before = current.lstat()
    assert switch_config_revisions(ctx, {owner.name: candidate.name}) == {owner.name: candidate.name}
    assert current.lstat().st_ino == before.st_ino


@pytest.mark.parametrize("revision", ["../escape", "/absolute", "current", "..", "a\\b"])
def test_switch_rejects_escaping_revision_before_changing_pointer(owner, revision):
    ctx = context(owner)
    with pytest.raises(ContainerError, match="Invalid generated configuration revision"):
        switch_config_revisions(ctx, {owner.name: revision})
    assert not owner.get_app_path("generated/current").exists()


def test_switch_does_not_replace_user_owned_directory(owner):
    ctx = context(owner)
    candidate = ctx.write_files(owner, {"base.yml": "same", "acl.yml": "allow"})
    current = owner.get_app_path("generated/current")
    current.mkdir()
    (current / "owned").write_text("user")
    with pytest.raises(ContainerError, match="must be a symbolic link"):
        switch_config_revisions(ctx, {owner.name: candidate.name})
    assert (current / "owned").read_text() == "user"


def test_multiple_switches_restore_prior_pointers_on_io_failure(owner, monkeypatch):
    ctx = context(owner)
    old = stage_files(owner, {"base.yml": "old"})
    candidate = ctx.write_files(owner, {"base.yml": "new"})
    other_root = owner.get_app_path("other")
    other = SimpleNamespace(name="other", get_app_path=lambda *parts: other_root.joinpath(*parts))
    ctx.project_containers.append(other)
    other_root.joinpath("generated", "old").mkdir(parents=True)
    other_root.joinpath("generated", "new").mkdir()
    switch_config_revisions(ctx, {owner.name: old.name, "other": "old"})
    import linktools.cntr.artifacts as artifacts
    replace = artifacts.os.replace

    def fail_other(source, target):
        if str(target) == str(other_root / "generated/current"):
            raise OSError("switch failed")
        return replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_other)
    with pytest.raises(OSError, match="switch failed"):
        switch_config_revisions(ctx, {owner.name: candidate.name, "other": "new"})
    assert owner.get_app_path("generated/current").resolve() == old
    assert other_root.joinpath("generated/current").resolve() == other_root / "generated/old"
    assert not list(other_root.joinpath("generated").glob(".current-*"))
