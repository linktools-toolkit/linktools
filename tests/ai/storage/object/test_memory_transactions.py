#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-backend transaction behavior (P1 transaction child backend).

Detailed coverage of the in-memory transaction child backend: read-your-
writes, list-your-writes, single-revision-per-tx, idempotency rollback,
move visibility, context-exit safety, nested-tx refusal. The cross-backend
contract lives in ``test_transaction.py``; these are the Memory-specific
behaviors the reference implementation must exhibit."""

from __future__ import annotations

import asyncio

import pytest

from linktools.ai.storage.backends.memory.object import (
    MemoryObjectBackend,
    _MemoryTransactionBackend,
    _MemoryTransactionState,
)
from linktools.ai.storage.object.errors import (
    StorageIdempotencyConflictError,
    StorageTransactionClosedError,
)
from linktools.ai.storage.object.models import (
    Depth,
    Masked,
    Missing,
    StorageKey,
    WriteOptions,
)
from linktools.ai.storage.object.store import ObjectStore


def _key(value: str) -> StorageKey:
    return StorageKey(value)


@pytest.fixture
def store() -> ObjectStore:
    return ObjectStore(primary=MemoryObjectBackend())


# --- read-your-writes --------------------------------------------------------


def test_read_your_writes_inside_tx(store):
    async def _run():
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"v1")
            obj = await tx.get(_key("/a"))
            assert obj is not None and obj.content == b"v1"
            # A second put to the same key updates the visible view.
            await tx.put(_key("/a"), b"v2")
            obj = await tx.get(_key("/a"))
            assert obj is not None and obj.content == b"v2"

    asyncio.run(_run())


def test_list_your_writes_inside_tx(store):
    async def _run():
        await store.put(_key("/parent/committed"), b"c")
        async with store.transaction() as tx:
            await tx.put(_key("/parent/staged1"), b"s1")
            await tx.put(_key("/parent/staged2"), b"s2")
            page = await tx.list(_key("/parent"), depth=Depth.ONE)
            values = sorted(i.key.value for i in page.items)
            # Committed + two staged all visible together.
            assert values == [
                "/parent/committed",
                "/parent/staged1",
                "/parent/staged2",
            ]

    asyncio.run(_run())


# --- delete semantics inside a transaction -----------------------------------


def test_delete_then_get_returns_missing_in_tx(store):
    async def _run():
        await store.put(_key("/a"), b"keep")
        async with store.transaction() as tx:
            await tx.delete(_key("/a"))
            # raw_get is the three-state form; a tombstone inside tx is Masked.
            lookup = await tx._primary.raw_get(_key("/a"))
            assert isinstance(lookup, Masked)
            # The public get() collapses Masked to None.
            assert await tx.get(_key("/a")) is None

    asyncio.run(_run())


def test_delete_missing_key_is_idempotent_in_tx(store):
    async def _run():
        async with store.transaction() as tx:
            # No prior record: delete is a no-op, raw_get returns Missing.
            await tx.delete(_key("/never"))
            lookup = await tx._primary.raw_get(_key("/never"))
            assert lookup is Missing

    asyncio.run(_run())


# --- same-key consecutive put version increments -----------------------------


def test_same_key_consecutive_put_increments_version(store):
    async def _run():
        async with store.transaction() as tx:
            r1 = await tx.put(_key("/a"), b"v1")
            r2 = await tx.put(_key("/a"), b"v2")
            r3 = await tx.put(_key("/a"), b"v3")
            # Three writes to the same key inside one tx: versions 1, 2, 3.
            assert (r1.info.version, r2.info.version, r3.info.version) == (1, 2, 3)
            # All three share ONE commit_revision (the tx bumps at most once).
            assert r1.info.commit_revision == r2.info.commit_revision == r3.info.commit_revision

    asyncio.run(_run())


# --- move visibility inside a transaction ------------------------------------


def test_move_visibility_in_tx(store):
    async def _run():
        await store.put(_key("/src"), b"payload")
        async with store.transaction() as tx:
            result = await tx.move(_key("/src"), _key("/dst"))
            assert result.content == b"payload"
            # Source is gone (tombstoned) inside the tx.
            assert await tx.get(_key("/src")) is None
            # Target is live inside the tx.
            dst = await tx.get(_key("/dst"))
            assert dst is not None and dst.content == b"payload"
        # After commit: source gone, target live, through the parent.
        assert await store.get(_key("/src")) is None
        assert await store.get(_key("/dst")) is not None

    asyncio.run(_run())


# --- rollback: data and idempotency ------------------------------------------


def test_rollback_discards_staged_data(store):
    async def _run():
        await store.put(_key("/keep"), b"orig")
        try:
            async with store.transaction() as tx:
                await tx.put(_key("/keep"), b"staged")
                await tx.put(_key("/new"), b"staged")
                raise RuntimeError("abort")
        except RuntimeError:
            pass
        # Pre-tx value survives.
        kept = await store.get(_key("/keep"))
        assert kept is not None and kept.content == b"orig"
        # Staged new key never landed.
        assert await store.get(_key("/new")) is None

    asyncio.run(_run())


def test_rollback_does_not_poison_parent_idempotency(store):
    """A key replayed in a ROLLED-BACK tx must NOT be poisoned on the parent:
    the parent's idempotency table never sees the staged entry, so the same
    key can be used in a later tx without conflict."""
    async def _run():
        options = WriteOptions(idempotency_key="req-1")
        try:
            async with store.transaction() as tx:
                await tx.put(_key("/a"), b"staged", options=options)
                raise RuntimeError("abort")
        except RuntimeError:
            pass
        # /a was never committed. Replay the same idempotency key in a fresh
        # tx with the SAME content: it must succeed (no conflict), because the
        # rolled-back tx did not durably reserve the key.
        async with store.transaction() as tx:
            result = await tx.put(_key("/a"), b"staged", options=options)
            assert result.content == b"staged"
        assert (await store.get(_key("/a"))).content == b"staged"

    asyncio.run(_run())


def test_committed_tx_idempotency_survives_for_replay(store):
    async def _run():
        options = WriteOptions(idempotency_key="req-1")
        async with store.transaction() as tx:
            r1 = await tx.put(_key("/a"), b"v1", options=options)
        # Replay the SAME key + content after commit: same result, no bump.
        r2 = await store.put(_key("/a"), b"v1", options=options)
        assert r2.info.etag == r1.info.etag
        assert r2.info.version == r1.info.version

    asyncio.run(_run())


def test_conflicting_replay_inside_tx_raises(store):
    async def _run():
        options_a = WriteOptions(idempotency_key="req-1")
        options_b = WriteOptions(idempotency_key="req-1")  # SAME key, different content
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"first", options=options_a)
            with pytest.raises(StorageIdempotencyConflictError):
                await tx.put(_key("/a"), b"second", options=options_b)

    asyncio.run(_run())


# --- revision rules ----------------------------------------------------------


def test_empty_tx_does_not_bump_revision(store):
    async def _run():
        before = await store.revision()
        async with store.transaction() as tx:
            # Read-only; no mutations.
            _ = await tx.get(_key("/a"))
        after = await store.revision()
        assert before == after

    asyncio.run(_run())


def test_one_tx_one_revision(store):
    async def _run():
        before = await store.revision()
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"1")
            await tx.put(_key("/b"), b"2")
            await tx.put(_key("/c"), b"3")
            await tx.delete(_key("/a"))
            await tx.move(_key("/b"), _key("/d"))
        after = await store.revision()
        # FIVE mutations, ONE revision bump.
        assert int(after) == int(before) + 1

    asyncio.run(_run())


def test_historical_revision_view_after_commit(store):
    """After a tx commits, reading AT the pre-tx revision returns the
    pre-tx state -- the staged writes are visible only at the new revision."""
    async def _run():
        await store.put(_key("/a"), b"r0")
        r0_revision = int(await store.revision())
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"r1")
            await tx.put(_key("/b"), b"new")
        r1_revision = int(await store.revision())
        assert r1_revision == r0_revision + 1
        # At r0: /a has its original content; /b does not exist.
        at_r0 = await store._primary.raw_get_at_revision(_key("/a"), r0_revision)
        assert at_r0 is not None and at_r0.content == b"r0"
        at_r0_b = await store._primary.raw_get_at_revision(_key("/b"), r0_revision)
        assert at_r0_b is None

    asyncio.run(_run())


# --- lifecycle safety --------------------------------------------------------


def test_child_unusable_after_context_exit(store):
    """Holding a reference to the tx child AFTER the context exits and then
    calling any operation must raise StorageTransactionClosedError -- the
    child's staged state is no longer valid."""
    async def _run():
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"1")
            child = tx._primary  # the _MemoryTransactionBackend
        # Context exited; using the child raises.
        with pytest.raises(StorageTransactionClosedError):
            await child.raw_get(_key("/a"))
        with pytest.raises(StorageTransactionClosedError):
            await child.raw_put_checked(
                _key("/b"), b"2", options=WriteOptions(), request_hash="x"
            )

    asyncio.run(_run())


def test_nested_transaction_refused(store):
    """A transaction-bound child must not open its own nested transaction."""
    async def _run():
        async with store.transaction() as tx:
            child = tx._primary
            # The child has no .transaction() that yields a usable backend.
            # Asserting the data structures don't allow it: a child is NOT a
            # TransactionalObjectBackend (the parent is).
            from linktools.ai.storage.object.backend import TransactionalObjectBackend

            assert not isinstance(child, TransactionalObjectBackend)

    asyncio.run(_run())


# --- internals --------------------------------------------------------------


def test_parent_has_no_ambient_tx_state():
    """The PARENT backend never carries transaction-local state. This is the
    regression guard for the 'delete self._tx' rule: a leaked attribute
    would silently resurrect the ambient-state bug."""
    backend = MemoryObjectBackend()
    assert not hasattr(backend, "_tx")
    # Open + close one transaction; parent state stays clean.
    async def _run():
        async with backend.transaction() as child:
            assert isinstance(child, _MemoryTransactionBackend)
            assert isinstance(child._state, _MemoryTransactionState)
        # After exit: parent has no ambient tx references.
        assert not hasattr(backend, "_tx")

    asyncio.run(_run())
