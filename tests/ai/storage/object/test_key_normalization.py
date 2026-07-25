#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""storage.object StorageKey contract.

``StorageKey`` is the storage kernel's normalized path: a frozen value type
that enforces the normalization rules (leading ``/``; collapse consecutive
``/``; forbid ``.``/``..``/NUL; root is ``/``; no backend namespace) and
exposes the navigation surface (``parent``/``name``/``join()``/``is_under()``).
These contracts are written against the TARGET ``linktools.ai.storage.object.models``
module, which does not exist yet (it lands when the storage.object core types
are introduced, moving + renaming today's ``asset.path.AssetPath``). Until then
every case here fails on import -- ``xfail(strict=True)`` holds the suite green
and ratchets to enforced the moment the type + its validation make the contract
pass."""

from __future__ import annotations

import pytest


def _storage_key():
    """Lazy import so the module collects before the storage.object type exists."""
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey


class TestStorageKeyNormalization:
    def test_root_is_a_valid_key(self) -> None:
        StorageKey = _storage_key()
        key = StorageKey("/")
        assert key.value == "/"

    def test_missing_leading_slash_is_rejected(self) -> None:
        StorageKey = _storage_key()
        with pytest.raises(ValueError):
            StorageKey("relative/path")

    def test_empty_is_rejected(self) -> None:
        StorageKey = _storage_key()
        with pytest.raises(ValueError):
            StorageKey("")

    def test_consecutive_slashes_collapse(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a//b").value == "/a/b"
        assert StorageKey("///").value == "/"

    def test_trailing_slash_does_not_create_empty_segment(self) -> None:
        # A trailing slash is collapsed; the normalized value has no dangling
        # empty segment (consistent with the consecutive-slash rule).
        StorageKey = _storage_key()
        assert StorageKey("/a/b/").value == "/a/b"

    def test_dot_segment_is_rejected(self) -> None:
        StorageKey = _storage_key()
        with pytest.raises(ValueError):
            StorageKey("/a/./b")

    def test_dotdot_segment_is_rejected(self) -> None:
        StorageKey = _storage_key()
        with pytest.raises(ValueError):
            StorageKey("/a/../b")

    def test_nul_byte_is_rejected(self) -> None:
        StorageKey = _storage_key()
        with pytest.raises(ValueError):
            StorageKey("/a/\x00b")

    def test_parent_walks_up_one_segment(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b").parent == StorageKey("/a")

    def test_parent_of_root_is_root(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/").parent == StorageKey("/")

    def test_name_is_the_last_segment(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b").name == "b"

    def test_join_appends_a_segment(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b").join("c") == StorageKey("/a/b/c")

    def test_is_under_checks_prefix(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b/c").is_under(StorageKey("/a"))
        assert not StorageKey("/a/b/c").is_under(StorageKey("/x"))

    def test_is_under_root_matches_everything(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b").is_under(StorageKey("/"))

    def test_value_is_immutable(self) -> None:
        StorageKey = _storage_key()
        key = StorageKey("/a")
        with pytest.raises((AttributeError, TypeError)):
            key.value = "/b"  # type: ignore[misc]

    def test_equality_is_by_value(self) -> None:
        StorageKey = _storage_key()
        assert StorageKey("/a/b") == StorageKey("/a//b")  # post-normalize
