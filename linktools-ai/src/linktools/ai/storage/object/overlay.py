#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The overlay object store: primary + ordered overlays.

Composes a primary ``ObjectWriterBackend`` over ordered ``ObjectReaderBackend``
overlays with the storage kernel's resolution semantics: primary wins; a
primary tombstone (``Masked``) blocks resurrection from an overlay; overlay
registration order is lookup priority; move operates only on primary-resident
sources. ``RevisionedOverlayObjectStore`` adds a composite revision token
encoding the backend set + order + per-layer revision so a cached view is
invalidated when any of those changes.

The k-way-merge listing + HMAC cursor land in a follow-up; this layer covers
the read/write/reveal/composite-revision surface the overlay contracts pin."""

from __future__ import annotations

import hashlib

from .backend import ObjectReaderBackend, ObjectWriterBackend
from .cursor import (
    BackendCursorState,
    BufferedObjectHead,
    ObjectCursorCodecProtocol,
    ObjectListCursor,
    StaleObjectCursorError,
)
from .errors import StorageObjectError, StorageObjectNotFoundError
from .models import (
    Depth,
    Found,
    Masked,
    Missing,
    ObjectInfo,
    ObjectPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)
from .store import (
    _delete_request_hash,
    _move_request_hash,
    _put_request_hash,
    _require_persistable_key,
)


class OverlayObjectStore:
    """Primary + ordered overlays. The primary must be a writer; overlays are
    readers. ``backend_id`` is tagged on each backend so a future listing
    cursor can attribute items to their source."""

    def __init__(
        self,
        *,
        primary: "ObjectWriterBackend | None" = None,
        overlays: "tuple[ObjectReaderBackend, ...]" = (),
        cursor_codec: "ObjectCursorCodecProtocol | None" = None,
    ) -> None:
        if primary is not None and not isinstance(primary, ObjectWriterBackend):
            raise StorageObjectError(
                "the OverlayObjectStore primary must be an ObjectWriterBackend"
            )
        self._primary = primary
        self._overlays = overlays
        # Injected, never auto-generated: list() needs it only when a caller
        # actually pages; a store that never lists need not construct one.
        self._cursor_codec = cursor_codec
        if primary is not None:
            primary.backend_id = "primary"
        for index, overlay in enumerate(overlays):
            overlay.backend_id = f"overlay:{index}"

    @property
    def backends(self) -> "tuple[ObjectReaderBackend, ...]":
        if self._primary is None:
            return self._overlays
        return (self._primary,) + self._overlays

    # --- read ----------------------------------------------------------------

    async def _lookup(self, key: StorageKey, *, include_content: bool = True):
        """Three-state resolution: primary first (Found returns, Masked stops),
        then overlays in registration order (first Found wins)."""
        if self._primary is not None:
            primary_lookup = await self._primary.raw_get(key, include_content=include_content)
            if isinstance(primary_lookup, Found):
                return primary_lookup
            if isinstance(primary_lookup, Masked):
                return Missing
        for overlay in self._overlays:
            overlay_lookup = await overlay.raw_get(key, include_content=include_content)
            if isinstance(overlay_lookup, Found):
                return overlay_lookup
        return Missing

    async def get(self, key: StorageKey) -> "StoredObject | None":
        _require_persistable_key(key)
        lookup = await self._lookup(key)
        return lookup.object if isinstance(lookup, Found) else None

    async def list(
        self,
        prefix: StorageKey,
        *,
        depth: "Depth" = Depth.ONE,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ObjectPage:
        """List objects under ``prefix`` via a k-way merge over each backend's
        OWN independent pagination position.

        Each backend advances through its own raw_list stream at its own pace,
        tracked as a BackendCursorState (its own opaque page cursor, heads
        already fetched but not yet output, an exhausted flag, and the
        revision it was minted against). A single shared "furthest scanned
        key" cursor would let a fast backend's position race ahead of a slow
        backend's, silently skipping the slow backend's own unscanned items on
        the next call -- tracking one independent state per backend makes
        every backend's progress lossless.

        Same-key priority mirrors get(): primary wins over any overlay; among
        overlays, registration order (overlay:0 highest). Tombstone detection
        mirrors get() too -- an overlay-only candidate is confirmed live via a
        point raw_get(include_content=False) against primary rather than
        requiring backends to embed tombstones in their raw_list stream.

        The returned cursor is an opaque HMAC-signed token naming every live
        backend, its own pagination position, and the revision it was read
        at. A resumed cursor's backend set and per-backend revision are
        cross-checked against the LIVE backend set on every call:
        StaleObjectCursorError if either has changed since the cursor was
        minted."""
        if self._cursor_codec is None:
            raise StorageObjectError(
                "list() with pagination requires a cursor_codec; this overlay "
                "was constructed without one"
            )
        live_backends = self.backends
        live_ids = tuple(backend.backend_id for backend in live_backends)

        if cursor is not None:
            decoded = self._cursor_codec.decode(cursor)
            decoded_ids = tuple(state.backend_id for state in decoded.backend_states)
            if decoded_ids != live_ids:
                raise StaleObjectCursorError(
                    f"cursor backend set {decoded_ids!r} no longer matches the "
                    f"live backend set {live_ids!r}"
                )
            states_by_id = {state.backend_id: state for state in decoded.backend_states}
            for backend in live_backends:
                current_revision = await backend.revision()
                if states_by_id[backend.backend_id].revision != current_revision:
                    raise StaleObjectCursorError(
                        f"backend {backend.backend_id!r} revision changed since "
                        f"this cursor was minted"
                    )
        else:
            states_by_id = {
                backend.backend_id: BackendCursorState(
                    backend_id=backend.backend_id,
                    cursor=None,
                    buffered=(),
                    exhausted=False,
                    revision=await backend.revision(),
                )
                for backend in live_backends
            }

        fetch_size = max(limit, 32)
        buffers: "dict[str, list[ObjectInfo]]" = {}
        next_page_cursor: "dict[str, str | None]" = {}
        exhausted: "dict[str, bool]" = {}
        revision_snapshot: "dict[str, str]" = {}

        for backend in live_backends:
            state = states_by_id[backend.backend_id]
            next_page_cursor[backend.backend_id] = state.cursor
            exhausted[backend.backend_id] = state.exhausted
            revision_snapshot[backend.backend_id] = state.revision
            # Rehydrate heads carried over from a previous page: the cursor
            # only stores a slim BufferedObjectHead (no content_type/size/
            # metadata), so a resumed buffered head is re-fetched via
            # raw_stat rather than stored as full ObjectInfo. A head whose
            # key has since been deleted (raw_stat returns None) is dropped.
            rehydrated: "list[ObjectInfo]" = []
            for head in state.buffered:
                info = await backend.raw_stat(StorageKey(head.key))
                if info is not None:
                    rehydrated.append(info)
            buffers[backend.backend_id] = rehydrated

        async def _refill(backend: "ObjectReaderBackend") -> None:
            bid = backend.backend_id
            if buffers[bid] or exhausted[bid]:
                return
            page = await backend.raw_list(
                prefix, depth=depth, limit=fetch_size, cursor=next_page_cursor[bid]
            )
            buffers[bid].extend(page.items)
            next_page_cursor[bid] = page.next_cursor
            if page.next_cursor is None:
                exhausted[bid] = True

        output: "list[ObjectInfo]" = []
        while len(output) < limit:
            # Refill lazily, right before this round needs a candidate -- NOT
            # after every consumed item -- so the loop never fetches a page it
            # will not actually consume this call.
            for backend in live_backends:
                await _refill(backend)
            candidates = {
                backend.backend_id: buffers[backend.backend_id][0]
                for backend in live_backends
                if buffers[backend.backend_id]
            }
            if not candidates:
                break
            min_key = min(info.key.value for info in candidates.values())
            contributing = [
                bid for bid, info in candidates.items() if info.key.value == min_key
            ]
            # Primary wins over any overlay; among overlays, registration
            # order (overlay:0 highest) -- live_ids is ordered primary-first
            # then overlays in registration order.
            contributing.sort(key=live_ids.index)
            winner_bid = contributing[0]
            winner_info = buffers[winner_bid][0]
            # Every contributing backend advances past this key, even the
            # ones that lost priority -- they must not re-offer it next round.
            for bid in contributing:
                buffers[bid].pop(0)
            if winner_bid != "primary" and self._primary is not None:
                # An overlay-only or overlay-winning candidate: primary may
                # still hold a tombstone for this key outside its OWN current
                # buffer window, so a point check is required to avoid
                # resurrecting a deleted object.
                primary_lookup = await self._primary.raw_get(
                    StorageKey(min_key), include_content=False
                )
                if isinstance(primary_lookup, Masked):
                    continue
            output.append(winner_info)

        more_available = any(
            buffers[backend.backend_id] or not exhausted[backend.backend_id]
            for backend in live_backends
        )
        if not more_available:
            return ObjectPage(items=tuple(output), next_cursor=None)

        next_states = tuple(
            BackendCursorState(
                backend_id=backend.backend_id,
                cursor=next_page_cursor[backend.backend_id],
                buffered=tuple(
                    BufferedObjectHead(
                        key=info.key.value,
                        version=info.version,
                        etag=info.etag,
                        tombstone=False,
                    )
                    for info in buffers[backend.backend_id]
                ),
                exhausted=exhausted[backend.backend_id],
                revision=revision_snapshot[backend.backend_id],
            )
            for backend in live_backends
        )
        next_token = self._cursor_codec.encode(
            ObjectListCursor(version=1, backend_states=next_states)
        )
        return ObjectPage(items=tuple(output), next_cursor=next_token)

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        if key.is_root:
            return None
        # stat is metadata-only; route through raw_stat, but still honor a
        # primary tombstone (Masked) so overlays are not resurrected.
        if self._primary is not None:
            info = await self._primary.raw_stat(key)
            if info is not None:
                return info
            if isinstance(await self._primary.raw_get(key, include_content=False), Masked):
                return None
        for overlay in self._overlays:
            overlay_info = await overlay.raw_stat(key)
            if overlay_info is not None:
                return overlay_info
        return None

    async def reveal(self, key: StorageKey) -> "StoredObject | None":
        """Unmask: read the FIRST overlay value even when the primary masks
        the key. Used to inspect what a tombstone is hiding."""
        _require_persistable_key(key)
        for overlay in self._overlays:
            overlay_lookup = await overlay.raw_get(key)
            if isinstance(overlay_lookup, Found):
                return overlay_lookup.object
        return None

    # --- write (primary only) ------------------------------------------------

    async def put(
        self,
        key: StorageKey,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject:
        self._require_primary()
        _require_persistable_key(key)
        return await self._primary.raw_put_checked(
            key, content, options=options, request_hash=_put_request_hash(key, content, options)
        )

    async def delete(
        self, key: StorageKey, *, options: WriteOptions = WriteOptions()
    ) -> None:
        self._require_primary()
        _require_persistable_key(key)
        await self._primary.raw_delete_checked(
            key, options=options, request_hash=_delete_request_hash(key, options)
        )

    async def move(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> StoredObject:
        self._require_primary()
        _require_persistable_key(source)
        _require_persistable_key(target)
        try:
            return await self._primary.raw_move_checked(
                source,
                target,
                options=options,
                request_hash=_move_request_hash(source, target, options),
            )
        except StorageObjectNotFoundError:
            # A missing source at the primary may live only in an overlay --
            # an overlay-only source cannot be moved atomically.
            for overlay in self._overlays:
                if isinstance(await overlay.raw_get(source, include_content=False), Found):
                    raise StorageObjectError(
                        f"cannot move overlay-only source atomically: {source}"
                    ) from None
            raise

    def _require_primary(self) -> None:
        if self._primary is None:
            raise StorageObjectError("this overlay has no writable primary backend")


class RevisionedOverlayObjectStore(OverlayObjectStore):
    """An OverlayObjectStore that exposes a composite revision token: a digest
    over the live backend set, their registration order, and each layer's own
    revision. Any of those changing invalidates a cached view derived from it."""

    async def revision(self) -> str:
        layers = self.backends
        if not layers:
            raise StorageObjectError(
                "cannot compute a composite revision with no backends"
            )
        hasher = hashlib.sha256()
        for backend in layers:
            hasher.update(backend.backend_id.encode())
            hasher.update(b":")
            hasher.update((await backend.revision()).encode())
            hasher.update(b"\n")
        return hasher.hexdigest()
