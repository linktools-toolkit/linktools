#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAS (compare-and-swap) contract for ObjectWriter.put (storage.object).

WriteOptions.if_match / if_none_match are the optimistic-concurrency gate:
if_none_match=True creates-only; if_match=<etag> updates-only-when-current.
These run against every backend via ``object_store_factory``. xfail(strict=True)
until the backends exist; the state outcomes (a failed put leaves the original
object untouched) are the load-bearing contract -- the exception type is
tightened to the concrete storage-object error when that lands."""

from __future__ import annotations

import asyncio

import pytest


class TestCAS:
    def test_if_none_match_create_succeeds_on_missing_key(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            obj = await store.put(
                _key("/a"), b"v1", options=WriteOptions(if_none_match=True)
            )
            assert obj.content == b"v1"

        asyncio.run(_run())

    def test_if_none_match_rejects_existing_key(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            await store.put(_key("/a"), b"v1")
            # A second create-only put on the existing key must fail...
            with pytest.raises(Exception):
                await store.put(
                    _key("/a"), b"v2", options=WriteOptions(if_none_match=True)
                )
            # ...and leave the original object untouched.
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"v1"

        asyncio.run(_run())

    def test_if_match_succeeds_when_etag_is_current(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            first = await store.put(_key("/a"), b"v1")
            updated = await store.put(
                _key("/a"), b"v2", options=WriteOptions(if_match=first.info.etag)
            )
            assert updated.content == b"v2"

        asyncio.run(_run())

    def test_if_match_rejects_stale_etag(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            await store.put(_key("/a"), b"v1")
            # A concurrent update moved the etag past "stale".
            with pytest.raises(Exception):
                await store.put(
                    _key("/a"), b"v3", options=WriteOptions(if_match="stale-etag")
                )
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"v1"

        asyncio.run(_run())

    def test_concurrent_same_if_match_only_one_wins(self, object_store_factory) -> None:
        # Two puts racing on the same current etag: exactly one wins, the other
        # is rejected. The object ends up as exactly one of the two payloads.
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            first = await store.put(_key("/a"), b"v1")
            etag = first.info.etag

            results: "list[bool]" = []

            async def _contend(payload: bytes) -> None:
                try:
                    await store.put(_key("/a"), payload, options=WriteOptions(if_match=etag))
                    results.append(True)
                except Exception:
                    results.append(False)

            await asyncio.gather(_contend(b"win-a"), _contend(b"win-b"))
            assert results.count(True) == 1, (
                f"exactly one concurrent if_match put must win, got {results}"
            )

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
