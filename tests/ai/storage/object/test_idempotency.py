#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent-write contract for ObjectWriter.put (storage.object).

WriteOptions.idempotency_key makes a put safely replayable: the SAME key +
idempotency_key + content returns the SAME StoredObject with NO version bump
on replay, while the SAME idempotency_key with DIFFERENT content is a conflict.
Runs against every backend via ``object_store_factory``. xfail(strict=True)
until the backends exist."""

from __future__ import annotations

import asyncio

import pytest


class TestIdempotency:
    def test_replay_same_key_and_content_returns_same_object(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            opts = WriteOptions(idempotency_key="req-1")
            first = await store.put(_key("/a"), b"v1", options=opts)
            replay = await store.put(_key("/a"), b"v1", options=opts)
            assert replay.info.version == first.info.version
            assert replay.info.etag == first.info.etag

        asyncio.run(_run())

    def test_replay_does_not_bump_version(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            opts = WriteOptions(idempotency_key="req-2")
            first = await store.put(_key("/b"), b"v1", options=opts)
            await store.put(_key("/b"), b"v1", options=opts)  # replay
            await store.put(_key("/b"), b"v1", options=opts)  # replay again
            obj = await store.get(_key("/b"))
            assert obj is not None and obj.info.version == first.info.version

        asyncio.run(_run())

    def test_same_key_different_content_same_idempotency_key_is_conflict(
        self, object_store_factory
    ) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            opts = WriteOptions(idempotency_key="req-3")
            await store.put(_key("/c"), b"v1", options=opts)
            # Reusing the idempotency key with DIFFERENT content is a conflict,
            # not a silent overwrite.
            with pytest.raises(Exception):
                await store.put(_key("/c"), b"DIFFERENT", options=opts)
            obj = await store.get(_key("/c"))
            assert obj is not None and obj.content == b"v1"

        asyncio.run(_run())

    def test_distinct_idempotency_keys_bump_version(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            first = await store.put(
                _key("/d"), b"v1", options=WriteOptions(idempotency_key="req-a")
            )
            second = await store.put(
                _key("/d"), b"v2", options=WriteOptions(idempotency_key="req-b")
            )
            assert second.info.version > first.info.version

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
