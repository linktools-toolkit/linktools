#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated trees keep their established identity and immutable-file guarantees."""
import hashlib
import json
import os

import pytest

from linktools.cntr.artifacts import prepare_files, prepared_hooks, stage_files
from linktools.cntr.errors import ContainerError
from test_prepared_files import context, owner


def test_pure_destination_preserves_existing_stage_hash(owner):
    files = {"nested/file": "value", "config": "secret"}
    payload = json.dumps([0o640, None, 0o755, sorted(files.items())], ensure_ascii=True, separators=(",", ":"))
    expected = owner.get_app_path("generated") / hashlib.sha256(payload.encode("utf-8")).hexdigest()
    declared = prepare_files(owner, files, mode=0o640)
    assert declared == expected
    assert not declared.exists()
    prepared_hooks(owner)[0].callback(context(owner))
    assert declared.is_dir()
    assert stage_files(owner, files, mode=0o640) == declared


@pytest.mark.parametrize("name", ["", "../escape", "/absolute", "nested/../../escape", "a\\b"])
def test_prepare_rejects_escaping_paths_before_registration(owner, name):
    with pytest.raises(ContainerError, match="within its tree"):
        prepare_files(owner, {name: "secret"})
    assert prepared_hooks(owner) == ()
    assert not owner.get_app_path("generated").exists()


@pytest.mark.parametrize("tampering", ["extra", "directory-mode", "file-symlink", "tree-symlink"])
def test_staging_rechecks_existing_tree_integrity(owner, tampering):
    files = {"nested/config": "secret"}
    tree = stage_files(owner, files)
    if tampering == "extra":
        (tree / "extra").write_text("injected")
    elif tampering == "directory-mode":
        (tree / "nested").chmod(0o777)
    elif tampering == "file-symlink":
        target = owner.get_app_path("external")
        target.write_text("secret")
        (tree / "nested/config").unlink()
        (tree / "nested/config").symlink_to(target)
    else:
        renamed = tree.with_name("external")
        tree.rename(renamed)
        tree.symlink_to(renamed)
    with pytest.raises(ContainerError, match="modified"):
        stage_files(owner, files)


def test_stage_group_and_directory_modes_are_preserved(owner):
    group = os.getegid()
    path = stage_files(owner, {"nested/config": "secret"}, mode=0o640, group=group)
    assert (path / "nested/config").stat().st_gid == group
    assert (path / "nested/config").stat().st_mode & 0o777 == 0o640
    assert path.stat().st_mode & 0o777 == 0o755
    assert (path / "nested").stat().st_mode & 0o777 == 0o755
    assert stage_files(owner, {"nested/config": "secret"}, mode=0o640, group=group) == path
