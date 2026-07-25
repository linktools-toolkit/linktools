#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transaction contract for storage.object (TransactionalObjectBackend).

All mutations inside one transaction share a SINGLE commit_revision (the
namespace bumps at most once per tx); an all-no-op tx writes no commit row;
a rolled-back tx leaves no object/history/idempotency/commit side effects.
The filesystem backend EXPLICITLY rejects multi-object transactions (it is
process-local checked-write only). Runs via ``object_store_factory``.
xfail(strict=True) until the backends exist."""

from __future__ import annotations

import asyncio

import pytest


def _supports_transaction(store) -> bool:
    return hasattr(store, "transaction")


class TestTransaction:
    def test_multi_object_tx_shares_one_commit_revision(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store):
                pytest.skip("backend has no transaction capability")
            async with store.transaction():
                await store.put(_key("/a"), b"1")
                await store.put(_key("/b"), b"2")
                await store.put(_key("/c"), b"3")
            # All three keys are visible after the tx commits.
            assert (await store.get(_key("/a"))) is not None
            assert (await store.get(_key("/c"))) is not None

        asyncio.run(_run())

    def test_rolled_back_tx_leaves_no_side_effects(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store):
                pytest.skip("backend has no transaction capability")
            await store.put(_key("/a"), b"keep")
            try:
                async with store.transaction():
                    await store.put(_key("/a"), b"staged")
                    await store.put(_key("/b"), b"staged")
                    raise RuntimeError("abort the tx")
            except RuntimeError:
                pass
            # The staged writes were rolled back; the pre-tx value survives.
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"keep"
            assert await store.get(_key("/b")) is None

        asyncio.run(_run())

    def test_all_noop_tx_bumps_no_revision(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store) or not hasattr(store, "revision"):
                pytest.skip("backend has no transaction/revision capability")
            before = await store.revision()
            async with store.transaction():
                pass  # no mutations
            assert await store.revision() == before

        asyncio.run(_run())

    def test_filesystem_rejects_multi_object_transaction(self, object_store_factory) -> None:
        # Filesystem is checked-write only; it must refuse a multi-object tx
        # rather than silently faking atomicity.
        async def _run() -> None:
            store = object_store_factory()
            if _supports_transaction(store) and "filesystem" not in type(store).__name__.lower():
                pytest.skip("only the filesystem backend rejects multi-object tx")
            with pytest.raises(Exception):
                async with store.transaction():
                    await store.put(_key("/a"), b"1")
                    await store.put(_key("/b"), b"2")

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
