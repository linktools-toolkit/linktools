#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SymlinkPolicy contract for storage/filesystem/_path_security.py, plus its
wiring into FileAssetBackend across all four subdirectories (data/metadata/
whiteouts/idempotency) it protects uniformly."""

import os

import pytest

from linktools.ai.errors import InvalidAssetPathError
from linktools.ai.asset.file import FileAssetBackend, _filename
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset._path_security import (
    SymlinkPolicy,
    open_temp_nofollow,
    resolve_secure_path,
)


# --------------------------------------------------------------------------
# resolve_secure_path unit contract
# --------------------------------------------------------------------------


def test_target_symlink_denied_under_deny(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = root / "leaf"
    link.symlink_to(target)
    with pytest.raises(InvalidAssetPathError):
        resolve_secure_path(root, "leaf", "file.json", policy=SymlinkPolicy.DENY)


def test_parent_directory_symlink_denied_under_deny(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # "sub" (a parent component of the candidate) is itself a symlink.
    (root / "sub").symlink_to(outside)
    with pytest.raises(InvalidAssetPathError):
        resolve_secure_path(root, "sub", "file.json", policy=SymlinkPolicy.DENY)


def test_internal_symlink_allowed_under_allow_internal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    real_dir = root / "real"
    real_dir.mkdir()
    (root / "alias").symlink_to(real_dir)
    # "alias" resolves to "root/real", which is still inside root -- allowed.
    result = resolve_secure_path(
        root, "alias", "file.json", policy=SymlinkPolicy.ALLOW_INTERNAL
    )
    assert result == root / "alias" / "file.json"


def test_external_symlink_denied_under_allow_internal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "alias").symlink_to(outside)
    with pytest.raises(InvalidAssetPathError):
        resolve_secure_path(
            root, "alias", "file.json", policy=SymlinkPolicy.ALLOW_INTERNAL
        )


def test_nonexistent_tail_is_allowed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # Neither "sub" nor "file.json" exist yet -- a create/write target.
    result = resolve_secure_path(root, "sub", "file.json", policy=SymlinkPolicy.DENY)
    assert result == root / "sub" / "file.json"


def test_traversal_and_nul_and_empty_components_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for bad in ("..", ".", "", "a\x00b"):
        with pytest.raises(InvalidAssetPathError):
            resolve_secure_path(root, bad, policy=SymlinkPolicy.DENY)


def test_toctou_race_caught_by_pre_write_recheck(tmp_path):
    """Simulates the race the pre-write recheck defends against: a path that
    was safe at initial resolve, then swapped for a symlink before the
    backend's _atomic_write actually replaces the temp file into place."""
    backend = FileAssetBackend(root=tmp_path / "backend")
    outside = tmp_path / "outside"
    outside.mkdir()

    # Seed so the metadata dir exists, then swap the target metadata path for
    # a symlink pointing outside root -- exactly what a concurrent attacker
    # would do between this backend's initial resolve and its write.
    async def _seed():
        await backend.raw_put(AssetPath("/x.txt"), b"1", content_type=None, metadata={})

    import asyncio

    asyncio.run(_seed())
    meta_name = _filename(AssetPath("/x.txt")) + ".json"
    meta_path = backend._root / ".assets" / "metadata" / meta_name
    meta_path.unlink()
    meta_path.symlink_to(outside / "does-not-exist.json")

    async def _rewrite():
        await backend.raw_put(AssetPath("/x.txt"), b"2", content_type=None, metadata={})

    with pytest.raises(InvalidAssetPathError):
        asyncio.run(_rewrite())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subdir", "make_path"),
    [
        (".assets/metadata", lambda p: _filename(p) + ".json"),
        (".assets/whiteouts", lambda p: _filename(p) + ".json"),
        (".assets/idempotency", lambda p: "some-key.json"),
    ],
)
async def test_symlink_in_each_protected_subdir_is_denied(tmp_path, subdir, make_path):
    backend = FileAssetBackend(root=tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    await backend.raw_put(AssetPath("/seed.txt"), b"x", content_type=None, metadata={})
    filename = make_path(AssetPath("/evil"))
    link = tmp_path / subdir / filename
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside / "does-not-exist.json")
    with pytest.raises(InvalidAssetPathError):
        if subdir.endswith("idempotency"):
            backend._idempotency_path("some-key")
        elif subdir.endswith("whiteouts"):
            backend._whiteout_path(AssetPath("/evil"))
        else:
            backend._meta_path(AssetPath("/evil"))


def test_open_temp_nofollow_creates_unique_fd(tmp_path):
    fd, path = open_temp_nofollow(tmp_path, prefix=".t.", suffix=".tmp")
    try:
        assert path.parent == tmp_path
        os.write(fd, b"hello")
    finally:
        os.close(fd)
        path.unlink()
