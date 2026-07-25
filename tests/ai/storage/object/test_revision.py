#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision-source contract for storage.object.

A revisioned object store exposes a namespace token (RevisionSource.revision)
that changes on real mutations and is stable across no-ops / idempotent
replays / rollbacks. Four revision strategies exist: native (the backend's
own counter), static (an injected release id), scanning (an O(N) digest over
a directory reader), and composite (an overlay's encoded backend-set +
order + per-layer revision). xfail(strict=True) until the types/strategies
land."""

from __future__ import annotations

import asyncio

import pytest


class TestNativeRevision:
    def test_revision_changes_on_put(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not hasattr(store, "revision"):
                pytest.skip("backend has no revision capability")
            before = await store.revision()
            await store.put(_key("/a"), b"v1")
            assert await store.revision() != before

        asyncio.run(_run())

    def test_revision_changes_on_delete(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not hasattr(store, "revision"):
                pytest.skip("backend has no revision capability")
            await store.put(_key("/a"), b"v1")
            mid = await store.revision()
            await store.delete(_key("/a"))
            assert await store.revision() != mid

        asyncio.run(_run())

    def test_identical_noop_does_not_change_revision(self, object_store_factory) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.models import WriteOptions

            store = object_store_factory()
            if not hasattr(store, "revision"):
                pytest.skip("backend has no revision capability")
            # The FIRST use of an idempotency key is a real write (it bumps);
            # capture the revision AFTER it, then assert a replay is a no-op.
            await store.put(_key("/a"), b"v1", options=WriteOptions(idempotency_key="r"))
            at_put = await store.revision()
            await store.put(_key("/a"), b"v1", options=WriteOptions(idempotency_key="r"))
            await store.put(_key("/a"), b"v1", options=WriteOptions(idempotency_key="r"))
            assert await store.revision() == at_put

        asyncio.run(_run())


@pytest.mark.xfail(
    strict=True,
    reason="static/scanning revision readers are not yet implemented",
)
class TestRevisionStrategies:
    def test_static_revision_returns_injected_release_id(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.index import StaticRevisionReader

            reader = StaticRevisionReader("release-42")
            assert await reader.revision() == "release-42"

        asyncio.run(_run())

    def test_static_revision_changes_when_release_id_changes(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.index import StaticRevisionReader

            assert await StaticRevisionReader("r1").revision() != (
                await StaticRevisionReader("r2").revision()
            )

        asyncio.run(_run())

    # Composite revision (backend-set + order + per-layer) is covered by
    # test_overlay.TestCompositeRevision -- it ratcheted green when the
    # RevisionedOverlayObjectStore landed.


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
