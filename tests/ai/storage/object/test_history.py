#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Object-history contract for storage.object (ObjectHistoryReader).

Backends that support history expose per-key versioning: get_version (exact),
list_versions (monotonic), and a tombstone appears as a versioned deletion
rather than the key vanishing. Runs against backends via
``object_store_factory``; backends without history skip. xfail(strict=True)
until the backends exist."""

from __future__ import annotations

import asyncio

import pytest


def _supports_history(store) -> bool:
    return hasattr(store, "get_version") and hasattr(store, "list_versions")


class TestHistory:
    def test_get_version_returns_the_exact_prior_version(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_history(store):
                pytest.skip("backend has no history capability")
            v1 = await store.put(_key("/a"), b"v1")
            await store.put(_key("/a"), b"v2")
            fetched = await store.get_version(_key("/a"), v1.info.version)
            assert fetched is not None and fetched.content == b"v1"

        asyncio.run(_run())

    def test_list_versions_is_monotonically_increasing(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_history(store):
                pytest.skip("backend has no history capability")
            await store.put(_key("/a"), b"v1")
            await store.put(_key("/a"), b"v2")
            await store.put(_key("/a"), b"v3")
            page = await store.list_versions(_key("/a"))
            versions = [v.version for v in page.items]
            assert versions == sorted(versions)
            assert len(set(versions)) == len(versions)

        asyncio.run(_run())

    def test_delete_records_a_tombstone_version(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_history(store):
                pytest.skip("backend has no history capability")
            await store.put(_key("/a"), b"v1")
            await store.delete(_key("/a"))
            # Current state is missing, but history still lists the key.
            assert await store.get(_key("/a")) is None
            page = await store.list_versions(_key("/a"))
            assert len(page.items) >= 2  # the live version + the tombstone

        asyncio.run(_run())

    def test_noop_write_records_no_history_row(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            if not _supports_history(store):
                pytest.skip("backend has no history capability")
            await store.put(_key("/a"), b"v1", options=WriteOptions(idempotency_key="r"))
            before = len((await store.list_versions(_key("/a"))).items)
            await store.put(_key("/a"), b"v1", options=WriteOptions(idempotency_key="r"))
            after = len((await store.list_versions(_key("/a"))).items)
            assert before == after

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
