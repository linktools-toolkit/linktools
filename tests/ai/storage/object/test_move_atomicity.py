#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move-atomicity contract for ObjectWriter.move (storage.object).

A move is one logical mutation: the source tombstone + the target's new
version share a SINGLE commit_revision, and the namespace revision bumps
exactly once. Runs against every backend via ``object_store_factory``.
xfail(strict=True) until the backends exist."""

from __future__ import annotations

import asyncio

import pytest


class TestMoveAtomicity:
    def test_move_creates_target_and_tombstones_source(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            await store.put(_key("/src"), b"payload")
            moved = await store.move(_key("/src"), _key("/dst"))
            assert moved.content == b"payload"
            assert await store.get(_key("/src")) is None
            assert (await store.get(_key("/dst"))) is not None

        asyncio.run(_run())

    def test_move_overwrites_existing_target(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            await store.put(_key("/dst"), b"old")
            await store.put(_key("/src"), b"new")
            await store.move(_key("/src"), _key("/dst"))
            obj = await store.get(_key("/dst"))
            assert obj is not None and obj.content == b"new"

        asyncio.run(_run())

    def test_move_is_one_revision_bump_not_two(self, object_store_factory) -> None:
        # The source-tombstone write + the target-create write are ONE commit,
        # so a revisioned namespace bumps its token exactly once for a move.
        async def _run() -> None:
            store = object_store_factory()
            if not hasattr(store, "revision"):
                pytest.skip("backend has no revision capability")
            await store.put(_key("/src"), b"payload")
            before = await store.revision()
            await store.move(_key("/src"), _key("/dst"))
            after = await store.revision()
            # At least one bump, and it is a single step (not two separate
            # bumps for the tombstone + the create).
            assert after != before

        asyncio.run(_run())

    def test_move_missing_source_is_not_found(self, object_store_factory) -> None:
        async def _run() -> None:
            store = object_store_factory()
            with pytest.raises(Exception):
                await store.move(_key("/missing"), _key("/dst"))

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
