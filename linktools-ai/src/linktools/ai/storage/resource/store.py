#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceStore: Primary+Overlay composition. Every cross-cutting concern --
idempotency-key comparison, conditional writes, whiteout-aware fallback lookup --
lives here, not in any backend. Backends only implement raw CRUD (see backend.py)."""

import hashlib
import json

from .backend import ResourceBackend
from .models import (
    Depth,
    Found,
    IdempotencyRecord,
    Masked,
    Missing,
    Resource,
    ResourceInfo,
    ResourceLookupInfo,
    ResourcePage,
    WriteOptions,
)
from .path import ResourcePath
from ...errors import (
    IdempotencyConflictError,
    ResourcePreconditionFailedError,
    ResourceReadOnlyError,
)


def _request_hash(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


class ResourceStore:
    def __init__(
        self, *, primary: ResourceBackend, overlays: "tuple[ResourceBackend, ...]" = ()
    ) -> None:
        self._primary = primary
        self._overlays = overlays

    async def _lookup_chain(self, path: ResourcePath):
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

    async def get(self, path: ResourcePath) -> "Resource | None":
        lookup = await self._lookup_chain(path)
        return lookup.resource if isinstance(lookup, Found) else None

    async def stat(self, path: ResourcePath) -> "ResourceLookupInfo | None":
        """Metadata-only stat: delegate to backend.raw_stat when
        available so the content blob is never loaded. Falls back to get()+info
        only for backends that don't implement raw_stat (Memory).

        Three-state resolution mirrors _lookup_chain: a Masked primary result
        stops the overlay search. The masked check
        uses raw_get(include_content=False) -- only invoked in the rare case
        where raw_stat returned None and overlays exist that might otherwise
        resurrect a masked path."""
        if hasattr(self._primary, "raw_stat"):
            info = await self._primary.raw_stat(path)
            if info is not None:
                return info
            # raw_stat returns None for both Missing and Masked. Distinguish
            # them so a primary whiteout hides overlays (no resurrection).
            primary_lookup = await self._primary.raw_get(path, include_content=False)
            if isinstance(primary_lookup, Masked):
                return None
            for overlay in self._overlays:
                if hasattr(overlay, "raw_stat"):
                    overlay_info = await overlay.raw_stat(path)
                    if overlay_info is not None:
                        return overlay_info
                else:
                    overlay_lookup = await overlay.raw_get(path, include_content=False)
                    if isinstance(overlay_lookup, Found):
                        return overlay_lookup.resource.info
            return None
        resource = await self.get(path)
        return resource.info if resource is not None else None

    async def propfind(
        self,
        path: ResourcePath,
        *,
        depth: Depth = Depth.ONE,
        limit: int = 100,
        cursor: "str | None" = None,
    ) -> ResourcePage:
        """List resources under `path`, merging Primary and Overlay results.

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
        merged: "dict[str, ResourceInfo]" = {}
        # Fetch limit+1 from each backend so we can detect "more available"
        # without a second count query.
        fetch_limit = limit + 1
        max_scanned: "str | None" = cursor
        backend_has_more = False

        def _note_page(page: ResourcePage) -> None:
            nonlocal max_scanned, backend_has_more
            if page.cursor is not None:
                backend_has_more = True
            for info in page.items:
                if max_scanned is None or info.path.value > max_scanned:
                    max_scanned = info.path.value

        for overlay in reversed(self._overlays):
            page = await overlay.raw_propfind(
                path, depth=depth, limit=fetch_limit, cursor=cursor
            )
            _note_page(page)
            for info in page.items:
                merged[info.path.value] = info
        primary_page = await self._primary.raw_propfind(
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
                ResourcePath(overlay_only_path), include_content=False
            )
            if isinstance(primary_lookup, Masked):
                del merged[overlay_only_path]
        items = tuple(merged[key] for key in sorted(merged))
        if len(items) > limit:
            # More available: next page resumes strictly after the last path
            # we're returning this call. items[limit-1] is the limit-th item.
            return ResourcePage(items=items[:limit], cursor=items[limit - 1].path.value)
        if backend_has_more:
            # Under-full page (post-filter) but a backend still has unfetched
            # candidates -- return everything we have and point the cursor
            # past every path scanned this round, not just past what we
            # returned, so no unscanned item is skipped and no scanned-but-
            # filtered item is re-examined.
            return ResourcePage(items=items, cursor=max_scanned)
        return ResourcePage(items=items, cursor=None)

    def _require_writable_primary(self) -> None:
        if self._primary.readonly:
            raise ResourceReadOnlyError("primary backend is read-only")

    async def _check_idempotency(
        self, operation: str, key: "str | None", request_hash: str
    ) -> "IdempotencyRecord | None":
        if key is None:
            return None
        record = await self._primary.get_idempotency(f"{operation}:{key}")
        if record is not None and record.request_hash != request_hash:
            raise IdempotencyConflictError(
                f"idempotency key {key!r} reused with a different request"
            )
        return record

    async def _save_idempotency(
        self,
        operation: str,
        key: "str | None",
        request_hash: str,
        result: "ResourceInfo | None",
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
        path: ResourcePath,
        content: bytes,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> Resource:
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
        # TOCTOU fix: when the primary backend implements the
        # atomic checked operation, delegate precondition + idempotency + mutate
        # to it as a single atomic call so a concurrent writer cannot interleave
        # the three steps. The Memory backend does not implement it, so the
        # prior 3-step orchestration below remains as the fallback.
        if hasattr(self._primary, "raw_put_checked"):
            return await self._primary.raw_put_checked(
                path, content, options=options, request_hash=req_hash
            )

        existing_record = await self._check_idempotency(
            "put", options.idempotency_key, req_hash
        )
        if existing_record is not None:
            info = existing_record.result
            current_lookup = await self._primary.raw_get(path)
            content_bytes = (
                current_lookup.resource.content
                if isinstance(current_lookup, Found)
                else content
            )
            return Resource(info=info, content=content_bytes)

        current = await self._lookup_chain(path)
        if options.if_none_match and isinstance(current, Found):
            raise ResourcePreconditionFailedError(f"resource already exists: {path}")
        if options.if_match is not None:
            if (
                not isinstance(current, Found)
                or current.resource.info.etag != options.if_match
            ):
                raise ResourcePreconditionFailedError(
                    f"if-match precondition failed: {path}"
                )

        primary_state = await self._primary.raw_get(path)
        if (
            isinstance(primary_state, Found)
            and primary_state.resource.content == content
            and dict(primary_state.resource.info.metadata) == dict(options.metadata)
            and primary_state.resource.info.content_type == options.content_type
        ):
            info = primary_state.resource.info
        else:
            info = await self._primary.raw_put(
                path,
                content,
                content_type=options.content_type,
                metadata=options.metadata,
            )

        await self._save_idempotency("put", options.idempotency_key, req_hash, info)
        return Resource(info=info, content=content)

    async def delete(
        self, path: ResourcePath, *, options: WriteOptions = WriteOptions()
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
        # TOCTOU fix: delegate to the atomic checked op when available (see put).
        if hasattr(self._primary, "raw_delete_checked"):
            await self._primary.raw_delete_checked(
                path, options=options, request_hash=req_hash
            )
            return

        existing_record = await self._check_idempotency(
            "delete", options.idempotency_key, req_hash
        )
        if existing_record is not None:
            return

        reader_lookup = await self._lookup_chain(path)
        primary_raw = await self._primary.raw_get(path)
        if options.if_match is not None:
            if (
                not isinstance(reader_lookup, Found)
                or reader_lookup.resource.info.etag != options.if_match
            ):
                raise ResourcePreconditionFailedError(
                    f"if-match precondition failed: {path}"
                )

        result_info = None
        already_masked = isinstance(reader_lookup, Missing) and isinstance(
            primary_raw, Masked
        )
        never_existed = isinstance(reader_lookup, Missing) and not already_masked
        if not (already_masked or never_existed):
            result_info = await self._primary.raw_delete(path)

        await self._save_idempotency(
            "delete", options.idempotency_key, req_hash, result_info
        )

    async def move(
        self,
        src: ResourcePath,
        dst: ResourcePath,
        *,
        options: WriteOptions = WriteOptions(),
    ) -> Resource:
        """MOVE: a single domain operation. When the primary
        backend implements raw_move AND the source lives in primary, delegate
        to it -- the backend folds load-source + write-target + whiteout-source
        + bump-revision into ONE transaction, so a concurrent reader never sees
        the intermediate states (target written while source still live, or
        source masked while target missing) that a put+delete decomposition
        would expose. The revision counter bumps exactly once for the whole
        move.

        Two cases keep the prior put+delete orchestration: (1) the Memory
        backend has no transaction primitive, so it never implements raw_move;
        (2) an OVERLAY-only source must be copied across backends, which
        cannot be made fully atomic -- the prior path copies the overlay
        resource into primary and writes a primary whiteout to mask the
        overlay source.

        Idempotency: the atomic path keys the idempotency record under
        ``move:{key}``. The
        move path inherits put's ``put:{key}`` keying via the delegated put
        call."""
        self._require_writable_primary()
        if hasattr(self._primary, "raw_move"):
            # Atomic raw_move handles only primary-resident sources. An
            # overlay-only source falls through to the prior cross-backend
            # copy path.
            source_in_primary = (
                await self._primary.raw_stat(src) is not None
                if hasattr(self._primary, "raw_stat")
                else isinstance(
                    await self._primary.raw_get(src, include_content=False), Found
                )
            )
            if source_in_primary:
                req_hash = _request_hash(
                    b"move",
                    src.value.encode(),
                    dst.value.encode(),
                    (options.if_match or "").encode(),
                    str(options.if_none_match).encode(),
                    (options.actor or "").encode(),
                )
                existing = await self._check_idempotency(
                    "move", options.idempotency_key, req_hash
                )
                if existing is not None:
                    # Replay: re-fetch the target content so the caller gets a
                    # complete Resource, not just the cached info.
                    current = await self._primary.raw_get(dst)
                    content = (
                        current.resource.content if isinstance(current, Found) else b""
                    )
                    return Resource(info=existing.result, content=content)
                result = await self._primary.raw_move(src, dst, options=options)
                await self._save_idempotency(
                    "move", options.idempotency_key, req_hash, result.info
                )
                return result
        # Non-atomic fallback (Memory backend, or overlay-source move): put+delete.
        source = await self.get(src)
        if source is None:
            raise ResourcePreconditionFailedError(
                f"cannot move missing resource: {src}"
            )
        result = await self.put(
            dst,
            source.content,
            options=WriteOptions(
                content_type=source.info.content_type,
                metadata=source.info.metadata,
                idempotency_key=options.idempotency_key,
                actor=options.actor,
                if_match=options.if_match,
                if_none_match=options.if_none_match,
            ),
        )
        await self.delete(src)
        return result
