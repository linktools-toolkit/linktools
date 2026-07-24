#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssetStore: Primary+Overlay composition. Every cross-cutting concern --
idempotency-key comparison, conditional writes, whiteout-aware fallback lookup --
lives here, not in any backend. Backends only implement raw CRUD (see backend.py)."""

import hashlib
import json
import secrets
from datetime import datetime, timezone
from typing import Any

from .backend import AssetReaderBackend, AssetWriterBackend
from .cursor import AssetCursorCodec, AssetListCursor, BackendCursorState, BufferedAssetHead
from .models import (
    AssetKind,
    Depth,
    Found,
    Masked,
    Missing,
    Asset,
    AssetInfo,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath, _require_persistable_path
from ..errors import (
    AssetMoveNotSupportedError,
    IdempotencyConflictError,
    AssetPreconditionFailedError,
    AssetReadOnlyError,
    StaleAssetCursorError,
)


def _synthetic_root_info() -> AssetInfo:
    """The namespace root's fabricated AssetInfo: no backend stores a root
    record (backends only persist real Assets), so the Store synthesizes it
    on every call rather than reading it from anywhere."""
    return AssetInfo(
        path=AssetPath("/"),
        kind=AssetKind.DIRECTORY,
        etag="",
        version=0,
        content_type=None,
        size=0,
        modified_at=datetime.now(timezone.utc),
        synthetic=True,
    )


def _request_hash(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


class AssetStore:
    def __init__(
        self,
        *,
        primary: AssetWriterBackend,
        overlays: "tuple[AssetReaderBackend, ...]" = (),
        metrics: Any = None,
        cursor_secret: "bytes | None" = None,
    ) -> None:
        # primary must be a Writer: a read-only backend (ReadOnlyAssetBackend)
        # lacks the write methods, so it fails this structural check at
        # construction with a clear error rather than silently no-oping writes.
        if not isinstance(primary, AssetWriterBackend):
            raise AssetReadOnlyError(
                "the AssetStore primary must be an AssetWriterBackend; a "
                "read-only backend can only be supplied as an overlay"
            )
        self._primary = primary
        self._overlays = overlays
        # Tag each backend with a stable canonical id so a multi-backend listing
        # cursor can attribute each item to its source backend. The primary is
        # always "primary"; overlays are numbered by their position. A backend
        # carries its own default backend_id; the store overrides it here so the
        # id reflects the store's composition, not the backend's origin.
        primary.backend_id = "primary"
        for index, overlay in enumerate(overlays):
            overlay.backend_id = f"overlay:{index}"
        # Optional ObservabilityMetrics sink. When wired, an idempotency-key
        # CAS conflict (same key reused with a different request body)
        # increments ``asset_cas_conflict_total``. Default None = no-op so
        # existing callers keep their no-metrics behavior.
        self._metrics = metrics
        # List-cursor HMAC secret. A single-process deployment never needs to
        # decode a cursor minted by a DIFFERENT process, so a fresh random
        # secret per process is safe; a multi-process downstream sharing one
        # Storage must pass an explicit shared cursor_secret so every worker
        # can decode tokens the others minted.
        self._cursor_codec = AssetCursorCodec(cursor_secret or secrets.token_bytes(32))

    async def _lookup_chain(self, path: AssetPath):
        """Reader-facing three-state resolution across Primary then Overlays. A
        Masked result at Primary stops the search."""
        primary_lookup = await self._primary.raw_get(path)
        if isinstance(primary_lookup, Found):
            return primary_lookup
        if isinstance(primary_lookup, Masked):
            return Missing()
        for overlay in self._overlays:
            overlay_lookup = await overlay.raw_get(path)
            if isinstance(overlay_lookup, Found):
                return overlay_lookup
        return Missing()

    async def get(self, path: AssetPath) -> "Asset | None":
        _require_persistable_path(path)
        lookup = await self._lookup_chain(path)
        return lookup.asset if isinstance(lookup, Found) else None

    async def stat(self, path: AssetPath) -> "AssetLookupInfo | None":
        """Metadata-only stat: delegate to backend.raw_stat so the content blob
        is never loaded. Every Reader backend implements raw_stat.

        The namespace root is synthesized here directly -- no backend stores a
        root record, so root never reaches raw_stat.

        Three-state resolution mirrors _lookup_chain: a Masked primary result
        stops the overlay search. The masked check uses
        raw_get(include_content=False) -- only invoked when raw_stat returned
        None and overlays exist that might otherwise resurrect a masked path."""
        if path.is_root:
            return _synthetic_root_info()
        info = await self._primary.raw_stat(path)
        if info is not None:
            return info
        # raw_stat returns None for both Missing and Masked. Distinguish them so
        # a primary whiteout hides overlays (no resurrection).
        primary_lookup = await self._primary.raw_get(path, include_content=False)
        if isinstance(primary_lookup, Masked):
            return None
        for overlay in self._overlays:
            overlay_info = await overlay.raw_stat(path)
            if overlay_info is not None:
                return overlay_info
        return None

    async def list(
        self,
        path: AssetPath,
        *,
        depth: Depth = Depth.ONE,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> AssetPage:
        """List assets under `path` via a k-way merge over each backend's OWN
        independent pagination position.

        Each backend advances through its own raw_list stream at its own
        pace, tracked as a BackendCursorState (its own opaque page cursor,
        heads already fetched but not yet output, an exhausted flag, and the
        revision it was minted against). A single shared "furthest scanned
        path" cursor would let a fast backend's position race ahead of a
        slow backend's, silently skipping the slow backend's own unscanned
        items on the next call -- tracking one independent state per backend
        makes every backend's progress lossless.

        Same-path priority mirrors the prior single-cursor merge: primary
        wins over any overlay; among overlays, registration order (overlay:0
        highest). Whiteout/shadow detection also mirrors the prior approach
        -- an overlay-only candidate is confirmed live via a point
        raw_get(include_content=False) against primary rather than requiring
        backends to embed tombstones in their raw_list stream.

        The returned cursor is an opaque HMAC-signed token (see cursor.py)
        naming every live backend, its own pagination position, and the
        revision it was read at. A resumed cursor's backend set and
        per-backend revision are cross-checked against the LIVE backend set
        on every call: StaleAssetCursorError if either has changed since the
        cursor was minted (a resumed listing is never silently continued
        against inconsistent backend state), InvalidAssetCursorError if the
        token fails to decode at all."""
        live_backends: "tuple[AssetReaderBackend, ...]" = (self._primary,) + tuple(self._overlays)
        live_ids = tuple(backend.backend_id for backend in live_backends)

        if cursor is not None:
            decoded = self._cursor_codec.decode(cursor)
            decoded_ids = tuple(state.backend_id for state in decoded.backend_states)
            if decoded_ids != live_ids:
                raise StaleAssetCursorError(
                    f"cursor backend set {decoded_ids!r} no longer matches the "
                    f"live backend set {live_ids!r}"
                )
            states_by_id = {state.backend_id: state for state in decoded.backend_states}
            for backend in live_backends:
                current_revision = await backend.revision()
                if states_by_id[backend.backend_id].revision != current_revision:
                    raise StaleAssetCursorError(
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
        buffers: "dict[str, list[AssetInfo]]" = {}
        next_page_cursor: "dict[str, str | None]" = {}
        exhausted: "dict[str, bool]" = {}
        revision_snapshot: "dict[str, str]" = {}

        for backend in live_backends:
            state = states_by_id[backend.backend_id]
            next_page_cursor[backend.backend_id] = state.cursor
            exhausted[backend.backend_id] = state.exhausted
            revision_snapshot[backend.backend_id] = state.revision
            # Rehydrate heads carried over from a previous page: the cursor
            # only stores a slim BufferedAssetHead (no content_type / size /
            # metadata -- the cursor is kept minimal by design), so a
            # resumed buffered head is re-fetched via raw_stat rather than
            # stored as full AssetInfo. The backend's OWN cursor already
            # advanced past these heads when they were first fetched, so
            # skipping this step would silently lose them. A head whose
            # path has since been deleted (raw_stat returns None) is simply
            # dropped -- it no longer exists to list.
            rehydrated: "list[AssetInfo]" = []
            for head in state.buffered:
                info = await backend.raw_stat(AssetPath(head.path))
                if info is not None:
                    rehydrated.append(info)
            buffers[backend.backend_id] = rehydrated

        async def _refill(backend: "AssetReaderBackend") -> None:
            bid = backend.backend_id
            if buffers[bid] or exhausted[bid]:
                return
            page = await backend.raw_list(
                path, depth=depth, limit=fetch_size, cursor=next_page_cursor[bid]
            )
            buffers[bid].extend(page.items)
            next_page_cursor[bid] = page.cursor
            if page.cursor is None:
                exhausted[bid] = True

        output: "list[AssetInfo]" = []
        if path.is_root and cursor is None:
            # The root itself is always relative-depth 0 from itself, so every
            # Depth (ZERO/ONE/INFINITY) includes it. No backend stores a root
            # record, so it is synthesized here rather than fetched -- and
            # only on the FIRST page (cursor is None): "/" sorts before every
            # real path, so a resumed page never needs to synthesize it again.
            output.append(_synthetic_root_info())

        while len(output) < limit:
            # Refill lazily, right before this round needs a candidate --
            # NOT after every consumed item -- so the loop never fetches a
            # page it will not actually consume this call. Fetching eagerly
            # after the limit-th item would stash a whole unused page into
            # the NEXT cursor's buffered state, wasting a raw_stat rehydrate
            # per leftover head on resume for zero benefit.
            for backend in live_backends:
                await _refill(backend)
            candidates = {
                backend.backend_id: buffers[backend.backend_id][0]
                for backend in live_backends
                if buffers[backend.backend_id]
            }
            if not candidates:
                break
            min_path = min(info.path.value for info in candidates.values())
            contributing = [
                bid for bid, info in candidates.items() if info.path.value == min_path
            ]
            # Primary wins over any overlay; among overlays, registration
            # order (overlay:0 highest) -- live_ids is ordered primary-first
            # then overlays in registration order, so sorting contributors
            # by their position in live_ids reproduces that priority.
            contributing.sort(key=live_ids.index)
            winner_bid = contributing[0]
            winner_info = buffers[winner_bid][0]
            # Every contributing backend advances past this path, even the
            # ones that lost priority -- they must not re-offer it next round.
            for bid in contributing:
                buffers[bid].pop(0)
            if winner_bid != "primary":
                # An overlay-only or overlay-winning candidate: primary may
                # still hold a whiteout for this path outside its OWN current
                # buffer window, so a point check is required to avoid
                # resurrecting a deleted asset.
                primary_lookup = await self._primary.raw_get(
                    AssetPath(min_path), include_content=False
                )
                if isinstance(primary_lookup, Masked):
                    continue
            output.append(winner_info)

        more_available = any(
            buffers[backend.backend_id] or not exhausted[backend.backend_id]
            for backend in live_backends
        )
        if not more_available:
            return AssetPage(items=tuple(output), cursor=None)

        next_states = tuple(
            BackendCursorState(
                backend_id=backend.backend_id,
                cursor=next_page_cursor[backend.backend_id],
                buffered=tuple(
                    BufferedAssetHead(
                        path=info.path.value,
                        kind=info.kind.value,
                        version=info.version,
                        etag=info.etag,
                        whiteout=False,
                    )
                    for info in buffers[backend.backend_id]
                ),
                exhausted=exhausted[backend.backend_id],
                revision=revision_snapshot[backend.backend_id],
            )
            for backend in live_backends
        )
        next_token = self._cursor_codec.encode(
            AssetListCursor(version=1, backend_states=next_states)
        )
        return AssetPage(items=tuple(output), cursor=next_token)

    async def put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> Asset:
        _require_persistable_path(path)
        # the hash must cover every input that changes the operation's
        # meaning, not just its payload -- otherwise two PUTs with the same
        # path/content/metadata but DIFFERENT preconditions (if_match,
        # if_none_match) or actor hash identically, and a replayed idempotency
        # key would incorrectly return the first call's cached result instead
        # of re-evaluating (or conflicting on) the differing preconditions.
        req_hash = _request_hash(
            b"put",
            path.value.encode(),
            content,
            (options.content_type or "").encode(),
            json.dumps(dict(options.metadata), sort_keys=True).encode(),
            (options.if_match or "").encode(),
            str(options.if_none_match).encode(),
            (options.actor or "").encode(),
        )
        # The primary is an AssetWriterBackend: precondition + idempotency +
        # mutate run as ONE atomic call inside the backend so a concurrent
        # writer cannot interleave the three steps.
        try:
            return await self._primary.raw_put_checked(
                path, content, options=options, request_hash=req_hash
            )
        except IdempotencyConflictError:
            if self._metrics is not None:
                self._metrics.counter(
                    "asset_cas_conflict_total",
                    attributes={"operation": "put"},
                )
            raise

    async def delete(
        self, path: AssetPath, *, options: WriteOptions = WriteOptions()
    ) -> None:
        _require_persistable_path(path)
        # same rationale as put() -- if_match/actor must be part of the
        # hash so a replayed key with a different precondition/actor cannot be
        # mistaken for the same request.
        req_hash = _request_hash(
            b"delete",
            path.value.encode(),
            (options.if_match or "").encode(),
            (options.actor or "").encode(),
        )
        # The primary is an AssetWriterBackend: precondition + idempotency +
        # mutate run as ONE atomic call inside the backend.
        try:
            await self._primary.raw_delete_checked(
                path, options=options, request_hash=req_hash
            )
        except IdempotencyConflictError:
            if self._metrics is not None:
                self._metrics.counter(
                    "asset_cas_conflict_total",
                    attributes={"operation": "delete"},
                )
            raise
        return

    async def move(
        self,
        src: AssetPath,
        dst: AssetPath,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> Asset:
        """MOVE: a single atomic domain operation. The primary Writer folds
        load-source + write-target + whiteout-source + bump-revision into ONE
        transaction/locked section, so a concurrent reader never sees the
        intermediate states (target written while source still live, or source
        masked while target missing) that a put+delete decomposition would
        expose, and the revision counter bumps exactly once.

        Move supports only primary-resident sources. A source that lives only
        in an overlay cannot be moved atomically (copying across backends plus a
        whiteout is not an atomic move); it is refused with
        :class:`AssetMoveNotSupportedError` rather than faked.

        Idempotency: folded into the backend's atomic ``raw_move_checked`` (the
        precondition + move + idempotency-record save run in one locked
        section/transaction), mirroring put/delete."""
        _require_persistable_path(src)
        _require_persistable_path(dst)
        req_hash = _request_hash(
            b"move",
            src.value.encode(),
            dst.value.encode(),
            (options.if_match or "").encode(),
            str(options.if_none_match).encode(),
            (options.actor or "").encode(),
        )
        # Delegate to the backend's atomic checked op: it handles idempotency
        # replay, the move itself, and a missing-source precondition. Only when
        # the backend reports the source missing do we classify WHY: an
        # overlay-only source is refused (AssetMoveNotSupportedError), anything
        # else re-raises the backend's precondition failure.
        try:
            return await self._primary.raw_move_checked(
                src, dst, options=options, request_hash=req_hash
            )
        except AssetPreconditionFailedError:
            if await self._primary.raw_stat(src) is None:
                for overlay in self._overlays:
                    if isinstance(
                        await overlay.raw_get(src, include_content=False), Found
                    ):
                        raise AssetMoveNotSupportedError(
                            f"cannot move overlay-only source atomically: {src}"
                        ) from None
            raise
