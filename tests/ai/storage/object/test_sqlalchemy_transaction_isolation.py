#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLAlchemy transaction isolation (P1, transaction-bound child backend).

The transaction-bound child backend uses a SESSION-LOCAL AsyncSession that no
other coroutine can observe; the reusable parent backend opens its own session
per op and never carries active-session state. These tests fix deterministic
interleavings (via asyncio.Event barriers, not random sleeps) that prove the
two cannot bleed state into each other.

Note: SQLite (the test DB) serializes writers via a single write lock, so the
concurrent-write scenarios use a non-deadlocking pattern: one task holds the
tx open while the other does a PARENT (single-op, blocking) write that waits
for the lock, then the holder releases via rollback/commit and the parent
write proceeds. The structural assertions (different sessions, parent has no
ambient state, child closed after exit) hold on every dialect."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.backends.sqlalchemy.object import (
    SqlAlchemyObjectBackend,
    SqlAlchemyObjectStore,
    _SqlAlchemyTransactionBackend,
)
from linktools.ai.storage.object.errors import StorageTransactionClosedError
from linktools.ai.storage.object.models import StorageKey, WriteOptions
from linktools.ai.storage.sqlalchemy.dialects import SqliteDialect


def _key(value: str) -> StorageKey:
    return StorageKey(value)


@pytest.fixture
def store(tmp_path):
    # busy_timeout -> SQLite block-wait, so a parent write that hits the tx's
    # write lock waits for the holder to release rather than erroring.
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'iso.db'}",
        connect_args={"timeout": 30.0},
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyObjectStore(session_factory=session_factory)
    # Pre-create the schema so concurrent transactions don't race on the
    # lazy _ensure_schema() write path (which itself needs the write lock).
    asyncio.run(store._ensure_schema())
    return store


# --- parent has no ambient transaction state ---------------------------------


def test_parent_backend_has_no_ambient_tx_fields():
    """The PARENT backend never carries active-session or staged-revision
    state. This is the regression guard for the 'delete _tx_session/_tx_revision'
    rule: a leaked attribute would silently resurrect the cross-coroutine
    bleed bug."""
    backend = SqlAlchemyObjectBackend(
        session_factory=async_sessionmaker(create_async_engine("sqlite+aiosqlite://")),
        dialect=SqliteDialect(),
    )
    assert not hasattr(backend, "_tx_session")
    assert not hasattr(backend, "_tx_revision")
    assert not hasattr(backend, "_session")


def test_transaction_yields_session_bound_child(store):
    """ObjectStore.transaction() yields an ObjectStore whose backend is a
    _SqlAlchemyTransactionBackend (the session-bound child), not the parent."""

    async def _run():
        async with store.transaction() as tx:
            assert isinstance(tx._primary, _SqlAlchemyTransactionBackend)
            # The child IS-A SqlAlchemyObjectBackend (inheritance), but is NOT
            # the same instance as the parent backend.
            assert isinstance(tx._primary, SqlAlchemyObjectBackend)
            assert tx._primary is not store.backend

    asyncio.run(_run())


# --- each transaction gets its own session -----------------------------------


def test_each_transaction_gets_a_fresh_session(store):
    """Two sequential transactions get DIFFERENT AsyncSession instances."""

    async def _run():
        sessions = []
        async with store.transaction() as tx:
            sessions.append(tx._primary._session)
        async with store.transaction() as tx:
            sessions.append(tx._primary._session)
        assert sessions[0] is not sessions[1]

    asyncio.run(_run())


def test_concurrent_transactions_use_independent_sessions(store):
    """Two concurrent READ-only transactions get different AsyncSession
    instances (SQLite allows concurrent readers, so this needs no write
    lock)."""

    async def _run():
        captured = {}
        barrier = asyncio.Event()

        async def reader(name):
            async with store.transaction() as tx:
                captured[name] = tx._primary._session
                barrier.set()
                await asyncio.sleep(0.05)  # overlap with the other reader

        await asyncio.gather(reader("a"), reader("b"))
        assert captured["a"] is not captured["b"]

    asyncio.run(_run())


# --- isolation: parent reads cannot observe uncommitted tx state ------------


def test_parent_read_does_not_observe_uncommitted_tx_write(store):
    """A parent (non-tx) read opens its OWN session; it must not observe a
    concurrent transaction's uncommitted writes."""

    async def _run():
        await store.put(_key("/seed"), b"seed")
        barrier = asyncio.Event()
        proceed = asyncio.Event()
        captured = {}

        async def writer():
            async with store.transaction() as tx:
                await tx.put(_key("/x"), b"uncommitted")
                barrier.set()
                await proceed.wait()  # hold the tx open

        async def reader():
            await barrier.wait()
            captured["parent_saw"] = await store.get(_key("/x"))
            proceed.set()

        await asyncio.gather(writer(), reader())
        # The parent read ran while the tx was still open; it must NOT see
        # /x (the tx has not committed).
        assert captured["parent_saw"] is None
        # After commit, /x is visible.
        assert (await store.get(_key("/x"))).content == b"uncommitted"

    asyncio.run(_run())


# --- B's commit survives A's rollback (non-deadlocking) ----------------------


def test_parent_commit_survives_concurrent_tx_rollback(store):
    """Task A opens a tx and writes /a (holds the SQLite write lock); Task B
    signals it is about to do a parent.put(/b) and then does so. When A rolls
    back, B's write proceeds and commits. /b must survive; /a must not exist.

    Synchronization: B signals ``b_started`` BEFORE the put call, so A can
    deterministically roll back to release the lock. The put itself may or may
    not end up blocking on the lock (depending on scheduling), but B's commit
    must survive A's rollback either way."""

    async def _run():
        a_entered = asyncio.Event()
        b_started = asyncio.Event()
        b_outcome = {}

        async def writer_a():
            async with store.transaction() as tx:
                await tx.put(_key("/a"), b"staged-by-a")
                a_entered.set()
                await b_started.wait()
                raise RuntimeError("A aborts")

        async def writer_b():
            await a_entered.wait()
            b_started.set()  # tell A to roll back; release is imminent
            b_outcome["result"] = await store.put(_key("/b"), b"committed-by-b")
            b_outcome["done"] = True

        results = await asyncio.gather(
            writer_a(), writer_b(), return_exceptions=True
        )
        assert any(isinstance(r, RuntimeError) for r in results)
        assert b_outcome.get("done") is True
        # B's commit survived A's rollback.
        assert (await store.get(_key("/b"))).content == b"committed-by-b"
        # A's staged write did NOT land.
        assert await store.get(_key("/a")) is None

    asyncio.run(_run())


# --- child lifecycle ---------------------------------------------------------


def test_child_unusable_after_context_exit(store):
    """Holding a reference to the tx child after the context exits must raise
    StorageTransactionClosedError on any operation."""

    async def _run():
        async with store.transaction() as tx:
            await tx.put(_key("/a"), b"1")
            child = tx._primary
        with pytest.raises(StorageTransactionClosedError):
            await child.raw_get(_key("/a"))
        with pytest.raises(StorageTransactionClosedError):
            await child.raw_put_checked(
                _key("/b"),
                b"2",
                options=WriteOptions(),
                request_hash="x",
            )

    asyncio.run(_run())


def test_child_session_attribute_is_set_at_construction():
    """The child's _session attribute is set at construction (the supplied
    AsyncSession), and is the same session throughout the child's lifetime."""
    from sqlalchemy.ext.asyncio import AsyncSession

    engine = create_async_engine("sqlite+aiosqlite://")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _run():
        async with session_factory() as session:
            child = _SqlAlchemyTransactionBackend(
                session=session,
                dialect=SqliteDialect(),
            )
            assert child._session is session
            assert isinstance(child._session, AsyncSession)
            child.close()
            with pytest.raises(StorageTransactionClosedError):
                await child.raw_get(_key("/a"))

    asyncio.run(_run())


def test_nested_transaction_on_child_refused(store):
    """A transaction-bound child cannot open a nested transaction."""

    async def _run():
        async with store.transaction() as tx:
            child = tx._primary
            # The child has no session_factory; its transaction() refuses.
            from linktools.ai.storage.object.errors import StorageObjectNotFoundError

            with pytest.raises(StorageObjectNotFoundError):
                async with child.transaction():
                    pass

    asyncio.run(_run())
