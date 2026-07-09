# -*- coding: utf-8 -*-
"""Tests for :class:`linktools.core._paths.EnvironmentPaths` (spec §5.3).

EnvironmentPaths owns the resolved, normalised filesystem layout for an
Environment. Getters return absolute Paths and never create directories;
creation is explicit; deletes verify the target is within an expected root.
"""
import os

import pytest

from linktools.core._paths import EnvironmentPaths
from linktools.errors import EnvironmentError


@pytest.fixture
def paths(tmp_path):
    root = tmp_path / "pkg"
    storage = tmp_path / "storage"
    root.mkdir()
    storage.mkdir()
    return EnvironmentPaths(root=str(root), storage=str(storage))


def test_getters_are_absolute_and_normalized(paths, tmp_path):
    assert os.path.isabs(str(paths.root))
    assert os.path.isabs(str(paths.storage))
    for name in ("data", "temp", "cache", "config", "tools", "downloads", "logs", "assets"):
        value = getattr(paths, name)
        assert os.path.isabs(str(value)), name
        # No residual ".." segments.
        assert ".." not in str(value), name


def test_default_derivation(paths):
    assert paths.data == paths.storage / "data"
    assert paths.temp == paths.storage / "temp"
    assert paths.cache == paths.storage / "cache"
    assert paths.config == paths.storage / "config"
    assert paths.logs == paths.storage / "logs"
    assert paths.downloads == paths.storage / "downloads"
    # tools live under data (current convention), assets under the package root.
    assert paths.tools == paths.data / "tools"
    assert paths.assets == paths.root / "assets"


def test_explicit_overrides_win(tmp_path):
    paths = EnvironmentPaths(
        root=tmp_path / "pkg",
        storage=tmp_path / "storage",
        data=tmp_path / "custom-data",
        cache=tmp_path / "custom-cache",
    )
    assert paths.data == tmp_path / "custom-data"
    assert paths.cache == tmp_path / "custom-cache"
    # non-overridden ones still derive from storage
    assert paths.temp == paths.storage / "temp"


def test_getters_do_not_create_directories(paths):
    assert not paths.data.exists()
    assert not paths.cache.exists()
    _ = paths.data, paths.cache  # accessing must not create


def test_ensure_creates_directory(paths):
    paths.ensure_data()
    assert paths.data.is_dir()
    paths.ensure_cache()
    assert paths.cache.is_dir()
    # ensure is idempotent
    paths.ensure_data()


def test_ensure_respects_readonly(paths):
    paths.readonly = True
    paths.ensure_data()
    assert not paths.data.exists()


def test_safe_remove_within_storage(paths):
    target = paths.data / "f.txt"
    paths.ensure_data()
    target.write_text("x")
    paths.safe_remove(target)
    assert not target.exists()


def test_safe_remove_outside_root_is_rejected(paths, tmp_path):
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x")
    with pytest.raises(EnvironmentError):
        paths.safe_remove(outside)
    assert outside.exists()  # not deleted


def test_safe_remove_accepts_explicit_root(paths, tmp_path):
    # Deleting a path under root (e.g. an asset) must work when root is given.
    target = paths.root / "scratch.txt"
    target.write_text("x")
    paths.safe_remove(target, root=paths.root)
    assert not target.exists()


def test_temporary_factory_is_isolated_and_cleanable():
    import shutil
    paths = EnvironmentPaths.temporary()
    try:
        assert paths.storage.is_dir()
        assert paths.storage != EnvironmentPaths.temporary().storage  # unique per call
    finally:
        paths.cleanup()
    assert not paths.storage.exists()


def test_two_instances_are_isolated(tmp_path):
    a = EnvironmentPaths(root=tmp_path / "ra", storage=tmp_path / "sa")
    b = EnvironmentPaths(root=tmp_path / "rb", storage=tmp_path / "sb")
    assert a.data != b.data
    assert a.cache != b.cache
