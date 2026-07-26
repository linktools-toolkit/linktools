#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transaction contract for storage.object (TransactionalObjectBackend).

The call pattern is fixed: the parent store carries no transaction context,
writes go through the transaction-bound ``tx``.

    async with store.transaction() as tx:
        await tx.put(...)
        item = await tx.get(...)

All mutations inside one transaction share a SINGLE commit_revision (the
namespace bumps at most once per tx); an all-no-op tx writes no commit row;
a rolled-back tx leaves no object/history/idempotency/commit side effects.
The filesystem backend EXPLICITLY rejects multi-object transactions (it is
process-local checked-write only). Runs via ``object_store_factory``."""

from __future__ import annotations

import asyncio

import pytest

from linktools.ai.storage.object.errors import StorageTransactionNotSupportedError


def _supports_transaction(store) -> bool:
    return hasattr(store, "transaction")


class TestTransaction:
    def test_multi_object_tx_commits_atomically(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store):
                pytest.skip("backend has no transaction capability")
            async with store.transaction() as tx:
                await tx.put(_key("/a"), b"1")
                await tx.put(_key("/b"), b"2")
                await tx.put(_key("/c"), b"3")
            # All three keys are visible through the PARENT store after commit.
            assert (await store.get(_key("/a"))) is not None
            assert (await store.get(_key("/c"))) is not None

        asyncio.run(_run())

    def test_transaction_supports_read_your_writes(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store):
                pytest.skip("backend has no transaction capability")
            async with store.transaction() as tx:
                await tx.put(_key("/a"), b"staged")
                # Visible IMMEDIATELY inside the tx, before commit.
                obj = await tx.get(_key("/a"))
                assert obj is not None and obj.content == b"staged"
            # And visible through parent after commit.
            assert (await store.get(_key("/a"))) is not None

        asyncio.run(_run())

    def test_rolled_back_tx_leaves_no_side_effects(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            if not _supports_transaction(store):
                pytest.skip("backend has no transaction capability")
            await store.put(_key("/a"), b"keep")
            try:
                async with store.transaction() as tx:
                    await tx.put(_key("/a"), b"staged")
                    await tx.put(_key("/b"), b"staged")
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
            async with store.transaction() as tx:
                # Read-only: no mutations.
                _ = await tx.get(_key("/a"))
            assert await store.revision() == before

        asyncio.run(_run())

    def test_filesystem_rejects_multi_object_transaction(self, object_store_factory) -> None:
        # Filesystem is checked-write only; it must refuse a multi-object tx
        # rather than silently faking atomicity.
        async def _run() -> None:
            store = object_store_factory()
            if "filesystem" not in type(store).__name__.lower():
                pytest.skip("only the filesystem backend rejects multi-object tx")
            with pytest.raises(StorageTransactionNotSupportedError):
                async with store.transaction():
                    await store.put(_key("/a"), b"1")


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
