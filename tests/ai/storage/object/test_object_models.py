#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""storage.object core data-model contract.

Pins the frozen value types the storage kernel exposes (StorageKey/ObjectInfo/
StoredObject) plus the three-state lookup shape (Found/Masked/Missing), the
write-options surface (WriteOptions), and the list-depth enum. These are pure
data contracts -- no backend required."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _models():
    from linktools.ai.storage.object import models

    return models


class TestObjectInfo:
    def test_carries_key_etag_version_commit_revision(self) -> None:
        models = _models()
        key = models.StorageKey("/a/b")
        info = models.ObjectInfo(
            key=key,
            etag="etag-1",
            version=3,
            commit_revision=42,
            content_type="application/json",
            size=10,
            modified_at=datetime.now(timezone.utc),
            metadata={"k": "v"},
        )
        assert info.key == key
        assert info.etag == "etag-1"
        assert info.version == 3
        assert info.commit_revision == 42
        assert info.size == 10

    def test_commit_revision_is_optional(self) -> None:
        # A backend without transactions reports commit_revision=None.
        models = _models()
        info = models.ObjectInfo(
            key=models.StorageKey("/a"),
            etag="e",
            version=1,
            commit_revision=None,
            content_type=None,
            size=0,
            modified_at=datetime.now(timezone.utc),
            metadata={},
        )
        assert info.commit_revision is None

    def test_is_frozen(self) -> None:
        models = _models()
        info = models.ObjectInfo(
            key=models.StorageKey("/a"),
            etag="e",
            version=1,
            commit_revision=None,
            content_type=None,
            size=0,
            modified_at=datetime.now(timezone.utc),
            metadata={},
        )
        with pytest.raises((AttributeError, TypeError)):
            info.version = 2  # type: ignore[misc]


class TestStoredObject:
    def test_pairs_info_and_content(self) -> None:
        models = _models()
        info = models.ObjectInfo(
            key=models.StorageKey("/a"),
            etag="e",
            version=1,
            commit_revision=None,
            content_type=None,
            size=5,
            modified_at=datetime.now(timezone.utc),
            metadata={},
        )
        obj = models.StoredObject(info=info, content=b"hello")
        assert obj.info is info
        assert obj.content == b"hello"


class TestThreeStateShape:
    def test_found_wraps_a_stored_object(self) -> None:
        models = _models()
        obj = models.StoredObject(
            info=models.ObjectInfo(
                key=models.StorageKey("/a"),
                etag="e",
                version=1,
                commit_revision=None,
                content_type=None,
                size=0,
                modified_at=datetime.now(timezone.utc),
                metadata={},
            ),
            content=b"",
        )
        found = models.Found(obj)
        assert found.object is obj

    def test_masked_carries_key_version_and_commit_revision(self) -> None:
        # The masked three-state carries commit_revision (not just key+version)
        # so an overlay consumer can tell whether its cached view is stale.
        models = _models()
        masked = models.Masked(
            key=models.StorageKey("/a"), version=2, commit_revision=9
        )
        assert masked.key == models.StorageKey("/a")
        assert masked.version == 2
        assert masked.commit_revision == 9

    def test_missing_is_a_singleton(self) -> None:
        models = _models()
        assert models.Missing is models.Missing


class TestWriteOptions:
    def test_defaults_are_no_preconditions_no_idempotency(self) -> None:
        models = _models()
        opts = models.WriteOptions()
        assert opts.if_match is None
        assert opts.if_none_match is None
        assert opts.idempotency_key is None

    def test_preconditions_and_idempotency_are_settable(self) -> None:
        models = _models()
        opts = models.WriteOptions(
            if_match="etag-1", if_none_match=True, idempotency_key="req-1"
        )
        assert opts.if_match == "etag-1"
        assert opts.if_none_match is True
        assert opts.idempotency_key == "req-1"

    def test_is_frozen(self) -> None:
        models = _models()
        opts = models.WriteOptions()
        with pytest.raises((AttributeError, TypeError)):
            opts.if_match = "x"  # type: ignore[misc]


class TestDepthEnum:
    def test_one_and_zero_or_infinity_are_distinct_levels(self) -> None:
        # list() takes a Depth (default ONE). The enum must distinguish a
        # single-level listing from a deeper recursive one.
        models = _models()
        levels = {member.name for member in models.Depth}
        assert "ONE" in levels
