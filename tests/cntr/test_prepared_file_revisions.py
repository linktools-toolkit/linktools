#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Caller-owned revisions name immutable trees without bypassing validation."""
import os

import pytest

from linktools.cntr import ContainerError
from linktools.cntr.artifacts import (AppliedServiceModels, prepared_files_path, prepared_hooks,
                                     prune_prepared_files, sha256_of, stage_files)
from linktools.cntr.lifecycle.hooks import HookPhase
from test_prepared_files import context, model, owner


@pytest.mark.parametrize("revision", ["release-1", "", "../outside", "/absolute", "a\\b", "发布/一"])
def test_revision_is_opaque_and_does_not_mix_content(owner, revision):
    root = owner.get_app_path("generated")
    path = prepared_files_path(root, {"config": "a"}, revision=revision)
    assert path == root / ("revision-" + sha256_of(revision))
    assert prepared_files_path(root, {"other": "b"}, revision=revision, mode=0o640, group=42) == path
    assert path != prepared_files_path(root, {"config": "a"})
    assert not root.exists()


@pytest.mark.parametrize("revision", [1, False, [], object()])
def test_revision_rejects_non_strings_before_registering(owner, revision):
    with pytest.raises(ContainerError, match="revision must be a string or None"):
        owner.prepare_files({"config": "value"}, revision=revision)
    assert prepared_hooks(owner) == ()
    assert not owner.get_app_path("generated").exists()


@pytest.mark.parametrize("change", ["name", "content", "mode", "group"])
def test_same_revision_conflict_does_not_hide_in_hook_dedup(owner, change):
    files, mode, group = {"config": "first"}, 0o600, None
    path = owner.prepare_files(files, revision="release-1")
    if change == "name":
        files = {"other": "first"}
    elif change == "content":
        files = {"config": "second"}
    elif change == "mode":
        mode = 0o640
    else:
        group = os.getgid()
    with pytest.raises(ContainerError, match="revision conflicts with an existing declaration"):
        owner.prepare_files(files, revision="release-1", mode=mode, group=group)
    assert len(prepared_hooks(owner)) == 1
    assert owner.prepare_files({"config": "first"}, revision="release-1") == path
    ctx = context(owner)
    owner.hooks.call(HookPhase.BEFORE_START, ctx)
    assert ctx.prepared_dirs[owner.name] == path
    assert (path / "config").read_text() == "first"


def test_revision_and_files_are_captured_once(owner):
    files = {"config": "first"}
    revision = "release-1"
    path = owner.prepare_files(files, revision=revision)
    files["config"] = "second"
    revision = "release-2"
    assert not path.exists()
    ctx = context(owner)
    owner.hooks.call(HookPhase.BEFORE_START, ctx)
    assert ctx.file_path(owner, "config").read_text() == "first"
    assert ctx.prepared_dirs[owner.name] == path
    assert not prepared_files_path(owner.get_app_path("generated"), files, revision=revision).exists()


@pytest.mark.parametrize("change", ["name", "content", "mode", "group"])
def test_existing_revision_rejects_different_files_or_permissions(owner, change):
    path = stage_files(owner, {"config": "first"}, revision="release-1")
    files, mode, group = {"config": "first"}, 0o600, None
    if change == "name":
        files = {"other": "first"}
    elif change == "content":
        files = {"config": "second"}
    elif change == "mode":
        mode = 0o640
    else:
        group = (path / "config").stat().st_gid + 1
    with pytest.raises(ContainerError, match="revision conflicts"):
        stage_files(owner, files, revision="release-1", mode=mode, group=group)
    assert stage_files(owner, {"config": "first"}, revision="release-1") == path
    assert (path / "config").read_text() == "first"
    assert (path / "config").stat().st_mode & 0o777 == 0o600


def test_failed_revision_write_can_retry_without_polluting_old_generation(owner, monkeypatch):
    old = stage_files(owner, {"config": "old"}, revision="old")
    index = owner.manager.artifact_index.load()
    new = owner.prepare_files({"config": "new"}, revision="new")
    original = owner.runtime.chmod
    calls = []

    def fail_once(path, mode):
        if not calls:
            calls.append(True)
            raise OSError("test write failure")
        original(path, mode)

    monkeypatch.setattr(owner.runtime, "chmod", fail_once)
    ctx = context(owner)
    with pytest.raises(OSError, match="test write failure"):
        prepared_hooks(owner)[0].callback(ctx)
    assert not new.exists()
    assert not tuple(new.parent.glob(".prepare-*"))
    assert ctx.prepared_dirs == {}
    assert owner.manager.artifact_index.load() == index
    assert (old / "config").read_text() == "old"
    assert owner.prepare_files({"config": "new"}, revision="new") == new
    prepared_hooks(owner)[0].callback(ctx)
    assert (new / "config").read_text() == "new"
    assert (old / "config").read_text() == "old"


def test_revision_pruning_keeps_referenced_and_rollback_generations(owner):
    def apply(revision, value):
        ctx = context(owner)
        path = ctx.write_files(owner, {"config": value}, revision=revision)
        models = AppliedServiceModels(owner.manager, model(owner, path))
        models.record(("auth", "admin"))
        prune_prepared_files(ctx, models)
        return path

    first = apply("first", "old")
    abandoned = stage_files(owner, {"config": "abandoned"}, revision="abandoned")
    later = apply("later", "new")
    assert first.exists() and later.exists()
    assert not abandoned.exists()
    newest = apply("newest", "newest")
    assert later.exists() and newest.exists()
    assert not first.exists()
    assert all(entry.get("tree") != str(first.relative_to(owner.manager.data_path))
               for entry in owner.manager.artifact_index.load().values())
    # Removed generations have no permanent version registry.
    assert stage_files(owner, {"config": "reused"}, revision="first") == first


def test_revision_alone_changes_applied_mounts(owner):
    files = {"config": "unchanged"}
    first = stage_files(owner, files, revision="v1")
    initial = AppliedServiceModels(owner.manager, model(owner, first))
    initial.record(("auth", "admin"))
    same = stage_files(owner, files, revision="v1")
    assert AppliedServiceModels(owner.manager, model(owner, same)).changed_services == frozenset()
    next_path = owner.prepare_files(files, revision="v2")
    assert not next_path.exists()
    changed = AppliedServiceModels(owner.manager, model(owner, next_path))
    assert changed.changed_services == frozenset(("auth", "admin"))


def test_concurrent_revision_publish_cannot_replace_an_empty_tree(owner, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event

    reached_rename, release_rename, second_started = Event(), Event(), Event()
    original = os.rename
    original_lock = owner.manager.environ.locks.file_lock
    destination = prepared_files_path(owner.get_app_path("generated"), {}, revision="same")

    def pause_first(source, target):
        if str(target) == str(destination) and not reached_rename.is_set():
            reached_rename.set()
            assert release_rename.wait(5)
        return original(source, target)

    @contextmanager
    def observe_lock(path):
        if reached_rename.is_set():
            second_started.set()
        with original_lock(path):
            yield

    def publish_second():
        return stage_files(owner, {"config": "different"}, revision="same")

    monkeypatch.setattr(os, "rename", pause_first)
    monkeypatch.setattr(owner.manager.environ.locks, "file_lock", observe_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(stage_files, owner, {}, revision="same")
        try:
            assert reached_rename.wait(5)
            second = executor.submit(publish_second)
            assert second_started.wait(5)
        finally:
            release_rename.set()
        assert first.result(timeout=5) == destination
        with pytest.raises(ContainerError, match="revision conflicts"):
            second.result(timeout=5)
    assert tuple(destination.iterdir()) == ()
    assert not tuple(destination.parent.glob(".prepare-*"))
