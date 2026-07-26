#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hash-collision defense (P3 hash-collision defense).

The SQLAlchemy object backend uses SHA-256 key_hash columns as the index for
both storage_objects and storage_object_idempotency. A hash collision (real or
injected) where two plaintext plaintext keys share a digest MUST surface as
StorageHashCollisionError -- never as a silent read of the wrong object.

These tests inject a FIXED hash function so two distinct keys collide
deterministically, then verify every read/write path refuses to serve the
wrong object."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.backends.sqlalchemy.object import (
    SqlAlchemyObjectBackend,
    SqlAlchemyObjectStore,
)
from linktools.ai.storage.object.errors import StorageHashCollisionError
from linktools.ai.storage.object.models import StorageKey, WriteOptions
from linktools.ai.storage.sqlalchemy.dialects import SqliteObjectDialect


def _key(value: str) -> StorageKey:
    return StorageKey(value)




@pytest.fixture
def store(tmp_path, monkeypatch):
    """A SqlAlchemyObjectStore whose key_hash function is INJECTED so two
    distinct keys share a digest. _collider_keys is the pair of keys that map
    to the SAME hash."""
    collider_keys = {"/alpha", "/beta"}

    def fake_hash(value: str) -> bytes:
        if value in collider_keys:
            # Both collider keys share ONE digest, regardless of plaintext.
            return hashlib.sha256(b"__collider__").digest()
        return hashlib.sha256(value.encode("utf-8")).digest()

    # Patch the module-level _key_hash the backend uses. The idempotency hash
    # is NOT patched (different namespace; only the object-key collision is
    # under test here).
    from linktools.ai.storage.backends.sqlalchemy import object as backend_module

    monkeypatch.setattr(backend_module, "_key_hash", lambda k: fake_hash(k.value))

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'collision.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    s = SqlAlchemyObjectStore(session_factory=session_factory)
    asyncio.run(s._ensure_schema())
    return s, collider_keys


# --- the structural property: a collision is surfaced, never silently served -


def test_get_on_colliding_key_raises(store):
    """After /alpha is written, a get(/beta) -- whose hash collides with
    /alpha's -- MUST raise StorageHashCollisionError rather than return /alpha's
    content (the wrong object)."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        with pytest.raises(StorageHashCollisionError) as excinfo:
            await s.get(_key("/beta"))
        # The digest in the error is the colliding digest; the message must
        # NOT include either plaintext key (avoid leaking sensitive keys via
        # the error).
        msg = str(excinfo.value)
        assert "/alpha" not in msg
        assert "/beta" not in msg

    asyncio.run(_run())


def test_put_on_colliding_key_raises(store):
    """A put(/beta) when /alpha (hash-colliding) already exists MUST raise
    rather than overwrite /alpha."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        with pytest.raises(StorageHashCollisionError):
            await s.put(_key("/beta"), b"B")
        # /alpha was NOT overwritten.
        assert (await s.get(_key("/alpha"))).content == b"A"

    asyncio.run(_run())


def test_stat_on_colliding_key_raises(store):
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        with pytest.raises(StorageHashCollisionError):
            await s.stat(_key("/beta"))

    asyncio.run(_run())


def test_delete_on_colliding_key_raises(store):
    """delete(/beta) when /alpha exists MUST raise rather than tombstone the
    wrong object."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        with pytest.raises(StorageHashCollisionError):
            await s.delete(_key("/beta"))
        # /alpha survived.
        assert (await s.get(_key("/alpha"))).content == b"A"

    asyncio.run(_run())


def test_move_target_colliding_raises(store):
    """move(/x, /beta) where /alpha exists at /beta's hash MUST raise rather
    than clobbering /alpha."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        await s.put(_key("/x"), b"X")
        with pytest.raises(StorageHashCollisionError):
            await s.move(_key("/x"), _key("/beta"))
        # /alpha survived.
        assert (await s.get(_key("/alpha"))).content == b"A"

    asyncio.run(_run())


def test_version_read_on_colliding_key_raises(store):
    """raw_get_version(/beta, 1) when /alpha v1 exists at the same hash MUST
    raise rather than return /alpha's content as if it were /beta's."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        with pytest.raises(StorageHashCollisionError):
            await s.get_version(_key("/beta"), 1)

    asyncio.run(_run())


def test_distinct_keys_with_injected_collision_do_not_corrupt_each_other(store):
    """A non-colliding key still works normally -- the collision is scoped to
    just the colliding pair."""
    s, _ = store

    async def _run():
        await s.put(_key("/alpha"), b"A")
        await s.put(_key("/gamma"), b"G")  # distinct hash, no collision
        # /gamma reads back fine.
        assert (await s.get(_key("/gamma"))).content == b"G"
        # /alpha still reads back fine.
        assert (await s.get(_key("/alpha"))).content == b"A"

    asyncio.run(_run())


def test_no_collision_silent_path_for_non_colliding_keys(tmp_path):
    """Without the injected collision, the collision check is a no-op: normal
    distinct-key writes/reads succeed and never raise."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'clean.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    s = SqlAlchemyObjectStore(session_factory=session_factory)
    asyncio.run(s._ensure_schema())

    async def _run():
        await s.put(_key("/alpha"), b"A")
        await s.put(_key("/beta"), b"B")
        assert (await s.get(_key("/alpha"))).content == b"A"
        assert (await s.get(_key("/beta"))).content == b"B"
        assert (await s.get(_key("/gamma"))) is None

    asyncio.run(_run())
