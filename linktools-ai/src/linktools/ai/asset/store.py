#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AssetStore: Primary+Overlay composition. Every cross-cutting concern --
idempotency-key comparison, conditional writes, whiteout-aware fallback lookup --
lives here, not in any backend. Backends only implement raw CRUD (see backend.py)."""

import hashlib
import json
from typing import Any

from .backend import AssetReaderBackend, AssetWriterBackend
from .models import (
    Depth,
    Found,
    IdempotencyRecord,
    Masked,
    Missing,
    Asset,
    AssetInfo,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath
from ..errors import (
    AssetMoveNotSupportedError,
    IdempotencyConflictError,
    AssetPreconditionFailedError,
    AssetReadOnlyError,
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
    ) -> None:
        # A readonly backend is a Reader, not a Writer -- it cannot serve as a
        # primary (writes would silently no-op or raise mid-flight). Reject it
        # at construction so the misconfiguration surfaces immediately, not on
        # the first write.
        if getattr(primary, "readonly", False):
            raise AssetReadOnlyError(
                "a readonly backend cannot be the AssetStore primary; supply a "
                "Writer as primary and the readonly backend as an overlay"
            )
        self._primary = primary
        self._overlays = overlays
        # Optional ObservabilityMetrics sink. When wired, an idempotency-key
        # CAS conflict (same key reused with a different request body)
        # increments ``asset_cas_conflict_total``. Default None = no-op so
        # existing callers keep their no-metrics behavior.
        self._metrics = metrics

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
        lookup = await self._lookup_chain(path)
        return lookup.asset if isinstance(lookup, Found) else None

    async def stat(self, path: AssetPath) -> "AssetLookupInfo | None":
        """Metadata-only stat: delegate to backend.raw_stat so the content blob
        is never loaded. Every Reader backend implements raw_stat.

        Three-state resolution mirrors _lookup_chain: a Masked primary result
        stops the overlay search. The masked check uses
        raw_get(include_content=False) -- only invoked when raw_stat returned
        None and overlays exist that might otherwise resurrect a masked path."""
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
        """List assets under `path`, merging Primary and Overlay results.

        Cursor pagination: each backend is asked for limit+1 items
        past the cursor; the (limit+1)th item from any backend signals "more
        available". After merge + whiteout filter, if the result exceeds limit,
        the limit-th path becomes next_cursor and the caller passes it back to
        continue. The cursor is the literal normalized path string, stable
        because every backend sorts by path.

        Multi-backend merge caveat: a backend may return items past the cursor
        that are dropped by whiteout or shadowed by a higher-priority backend.
        The wasted fetch is correctness-neutral -- the next call's cursor
        strictly advances, so progress is guaranteed and termination is
        reached when no backend has more items past the cursor.

        whiteout/shadow filtering can shrink the merged page BELOW
        `limit` even though a backend still has unfetched items beyond its own
        fetch window (e.g. a whiteout removes an item from the middle of this
        page's candidate set, and the page that's left is short even though
        more real items exist further out). Naively returning cursor=None
        whenever ``len(items) <= limit`` would silently truncate the listing
        in that case. So track (a) whether ANY backend's raw page reported
        more available (a non-None page.cursor) and (b) the max path SCANNED
        this round (returned or filtered) across every backend -- when (a) is
        true, the next cursor is that max-scanned path (not just the last
        RETURNED path), so the next call resumes strictly after everything
        already examined. Filtering decisions (shadow/whiteout) are re-derived
        from primary state at any cursor position, not from the cursor value
        itself, so resuming past already-scanned-but-filtered paths never
        skips a still-live item."""
        merged: "dict[str, AssetInfo]" = {}
        # Fetch limit+1 from each backend so we can detect "more available"
        # without a second count query.
        fetch_limit = limit + 1
        max_scanned: "str | None" = cursor
        backend_has_more = False

        def _note_page(page: AssetPage) -> None:
            nonlocal max_scanned, backend_has_more
            if page.cursor is not None:
                backend_has_more = True
            for info in page.items:
                if max_scanned is None or info.path.value > max_scanned:
                    max_scanned = info.path.value

        for overlay in reversed(self._overlays):
            page = await overlay.raw_list(
                path, depth=depth, limit=fetch_limit, cursor=cursor
            )
            _note_page(page)
            for info in page.items:
                merged[info.path.value] = info
        primary_page = await self._primary.raw_list(
            path, depth=depth, limit=fetch_limit, cursor=cursor
        )
        _note_page(primary_page)
        primary_paths = {info.path.value for info in primary_page.items}
        for info in primary_page.items:
            merged[info.path.value] = info
        for overlay_only_path in list(merged):
            if overlay_only_path in primary_paths:
                continue
            primary_lookup = await self._primary.raw_get(
                AssetPath(overlay_only_path), include_content=False
            )
            if isinstance(primary_lookup, Masked):
                del merged[overlay_only_path]
        items = tuple(merged[key] for key in sorted(merged))
        if len(items) > limit:
            # More available: next page resumes strictly after the last path
            # we're returning this call. items[limit-1] is the limit-th item.
            return AssetPage(items=items[:limit], cursor=items[limit - 1].path.value)
        if backend_has_more:
            # Under-full page (post-filter) but a backend still has unfetched
            # candidates -- return everything we have and point the cursor
            # past every path scanned this round, not just past what we
            # returned, so no unscanned item is skipped and no scanned-but-
            # filtered item is re-examined.
            return AssetPage(items=items, cursor=max_scanned)
        return AssetPage(items=items, cursor=None)

    def _require_writable_primary(self) -> None:
        if self._primary.readonly:
            raise AssetReadOnlyError("primary backend is read-only")

    async def _check_idempotency(
        self, operation: str, key: "str | None", request_hash: str
    ) -> "IdempotencyRecord | None":
        if key is None:
            return None
        record = await self._primary.get_idempotency(f"{operation}:{key}")
        if record is not None and record.request_hash != request_hash:
            if self._metrics is not None:
                self._metrics.counter(
                    "asset_cas_conflict_total",
                    attributes={"operation": operation},
                )
            raise IdempotencyConflictError(
                f"idempotency key {key!r} reused with a different request"
            )
        return record

    async def _save_idempotency(
        self,
        operation: str,
        key: "str | None",
        request_hash: str,
        result: "AssetInfo | None",
    ) -> None:
        if key is None:
            return
        await self._primary.put_idempotency(
            IdempotencyRecord(
                key=f"{operation}:{key}", request_hash=request_hash, result=result
            )
        )

    async def put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> Asset:
        self._require_writable_primary()
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
        self._require_writable_primary()
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
        self._require_writable_primary()
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
