#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OverlayObjectStore contract (storage.object overlay layer).

An overlay composes a primary writer backend over ordered reader backends.
Semantics: primary wins; a primary tombstone (Masked) blocks resurrection from
an overlay; overlay registration order = lookup priority; move operates only
on primary-resident sources; reveal unmasks. Drives real Memory backends."""

from __future__ import annotations

import asyncio

import pytest


def _memory():
    from linktools.ai.storage.backends.memory.object import MemoryObjectStore

    return MemoryObjectStore()


def _codec():
    import os

    from linktools.ai.storage.object.cursor import HmacObjectCursorCodec

    return HmacObjectCursorCodec(os.urandom(32), scope="test")


class TestOverlay:
    def test_primary_wins_over_overlay(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/a"), b"from-overlay")
            await primary.put(_key("/a"), b"from-primary")
            store = OverlayObjectStore(primary=primary.backend, overlays=(overlay.backend,))
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"from-primary"

        asyncio.run(_run())

    def test_primary_falls_through_to_overlay_when_missing(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/only-overlay"), b"v")
            store = OverlayObjectStore(primary=primary.backend, overlays=(overlay.backend,))
            obj = await store.get(_key("/only-overlay"))
            assert obj is not None and obj.content == b"v"

        asyncio.run(_run())

    def test_primary_tombstone_masks_overlay_resurrection(self) -> None:
        # A deleted key in the primary must NOT be resurrected from an overlay.
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/a"), b"overlay-has-it")
            await primary.put(_key("/a"), b"primary-had-it")
            await primary.delete(_key("/a"))
            store = OverlayObjectStore(primary=primary.backend, overlays=(overlay.backend,))
            assert await store.get(_key("/a")) is None

        asyncio.run(_run())

    def test_overlay_order_is_lookup_priority(self) -> None:
        # Earlier-registered overlays take precedence over later ones.
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            hi = _memory()
            lo = _memory()
            await hi.put(_key("/a"), b"hi")
            await lo.put(_key("/a"), b"lo")
            store = OverlayObjectStore(primary=primary.backend, overlays=(hi.backend, lo.backend))
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"hi"

        asyncio.run(_run())

    def test_move_only_affects_primary_resident_source(self) -> None:
        # move() operates on the primary; an overlay-only key cannot be moved
        # (it is not primary-resident).
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/overlay-only"), b"v")
            store = OverlayObjectStore(primary=primary.backend, overlays=(overlay.backend,))
            with pytest.raises(Exception):
                await store.move(_key("/overlay-only"), _key("/dst"))

        asyncio.run(_run())

    def test_reveal_unmasks_a_primary_tombstone(self) -> None:
        # reveal() reads the overlay value a primary tombstone is hiding.
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/a"), b"hidden")
            await primary.put(_key("/a"), b"x")
            await primary.delete(_key("/a"))
            store = OverlayObjectStore(primary=primary.backend, overlays=(overlay.backend,))
            assert await store.get(_key("/a")) is None
            revealed = await store.reveal(_key("/a"))
            assert revealed is not None and revealed.content == b"hidden"

        asyncio.run(_run())


class TestOverlayList:
    """The k-way-merge listing + HMAC cursor: each backend advances through
    its own independent pagination position; same-key priority mirrors get()
    (primary wins, then overlay registration order); a primary tombstone
    shadows an overlay-only candidate at the same key."""

    def test_merges_keys_from_primary_and_overlay_sorted(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await primary.put(_key("/a"), b"1")
            await primary.put(_key("/c"), b"3")
            await overlay.put(_key("/b"), b"2")
            store = OverlayObjectStore(
                primary=primary.backend, overlays=(overlay.backend,), cursor_codec=_codec()
            )
            page = await store.list(_key("/"), limit=10)
            assert [i.key.value for i in page.items] == ["/a", "/b", "/c"]

        asyncio.run(_run())

    def test_primary_wins_same_key_in_listing(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await primary.put(_key("/a"), b"from-primary")
            await overlay.put(_key("/a"), b"from-overlay")
            store = OverlayObjectStore(
                primary=primary.backend, overlays=(overlay.backend,), cursor_codec=_codec()
            )
            page = await store.list(_key("/"), limit=10)
            assert len(page.items) == 1
            obj = await store.get(_key("/a"))
            assert obj is not None and obj.content == b"from-primary"

        asyncio.run(_run())

    def test_primary_tombstone_excludes_overlay_entry_from_listing(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await overlay.put(_key("/a"), b"overlay-has-it")
            await primary.put(_key("/a"), b"primary-had-it")
            await primary.delete(_key("/a"))
            store = OverlayObjectStore(
                primary=primary.backend, overlays=(overlay.backend,), cursor_codec=_codec()
            )
            page = await store.list(_key("/"), limit=10)
            assert [i.key.value for i in page.items] == []

        asyncio.run(_run())

    def test_pagination_round_trips_across_calls(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            for name in ("a", "b", "c"):
                await primary.put(_key(f"/{name}"), name.encode())
            store = OverlayObjectStore(
                primary=primary.backend, cursor_codec=_codec()
            )
            seen: "list[str]" = []
            cursor = None
            while True:
                page = await store.list(_key("/"), limit=1, cursor=cursor)
                seen.extend(i.key.value for i in page.items)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            assert seen == ["/a", "/b", "/c"]

        asyncio.run(_run())

    def test_stale_cursor_after_backend_set_changes_raises(self) -> None:
        # backend_id is positional ("primary", "overlay:0", ...), so a
        # changed BACKEND COUNT (not just a different overlay instance in the
        # same slot) is what produces a different id sequence -- add a
        # second overlay to change the shape.
        async def _run() -> None:
            from linktools.ai.storage.object.cursor import StaleObjectCursorError
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            overlay = _memory()
            await primary.put(_key("/a"), b"1")
            await primary.put(_key("/b"), b"2")
            codec = _codec()
            store = OverlayObjectStore(
                primary=primary.backend, overlays=(overlay.backend,), cursor_codec=codec
            )
            page = await store.list(_key("/"), limit=1)
            assert page.next_cursor is not None
            # A DIFFERENT overlay composition -- one MORE overlay layer.
            extra_overlay = _memory()
            store2 = OverlayObjectStore(
                primary=primary.backend,
                overlays=(overlay.backend, extra_overlay.backend),
                cursor_codec=codec,
            )
            with pytest.raises(StaleObjectCursorError):
                await store2.list(_key("/"), limit=1, cursor=page.next_cursor)

        asyncio.run(_run())

    def test_stale_cursor_after_revision_changes_raises(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.cursor import StaleObjectCursorError
            from linktools.ai.storage.object.overlay import OverlayObjectStore

            primary = _memory()
            await primary.put(_key("/a"), b"1")
            await primary.put(_key("/b"), b"2")
            store = OverlayObjectStore(primary=primary.backend, cursor_codec=_codec())
            page = await store.list(_key("/"), limit=1)
            assert page.next_cursor is not None
            # Mutate primary between page 1 and page 2 -- its revision moves.
            await primary.put(_key("/c"), b"3")
            with pytest.raises(StaleObjectCursorError):
                await store.list(_key("/"), limit=1, cursor=page.next_cursor)

        asyncio.run(_run())


class TestCompositeRevision:
    def test_revision_with_no_backends_raises(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import (
                RevisionedOverlayObjectStore,
            )

            overlay = RevisionedOverlayObjectStore()
            with pytest.raises(Exception):
                await overlay.revision()

        asyncio.run(_run())

    def test_revision_changes_when_a_layer_mutates(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import (
                RevisionedOverlayObjectStore,
            )

            primary = _memory()
            overlay = _memory()
            store = RevisionedOverlayObjectStore(
                primary=primary.backend, overlays=(overlay.backend,)
            )
            before = await store.revision()
            await primary.put(_key("/a"), b"v")
            after = await store.revision()
            assert before != after

        asyncio.run(_run())

    def test_revision_changes_when_overlay_order_changes(self) -> None:
        async def _run() -> None:
            from linktools.ai.storage.object.overlay import (
                RevisionedOverlayObjectStore,
            )

            primary = _memory()
            hi = _memory()
            lo = _memory()
            await hi.put(_key("/a"), b"hi")
            await lo.put(_key("/a"), b"lo")
            store_a = RevisionedOverlayObjectStore(
                primary=primary.backend, overlays=(hi.backend, lo.backend)
            )
            store_b = RevisionedOverlayObjectStore(
                primary=primary.backend, overlays=(lo.backend, hi.backend)
            )
            assert await store_a.revision() != await store_b.revision()

        asyncio.run(_run())


def _key(value: str):
    from linktools.ai.storage.object.models import StorageKey

    return StorageKey(value)
