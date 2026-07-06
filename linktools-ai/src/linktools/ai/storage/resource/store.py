#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResourceStore: Primary+Overlay composition. Every cross-cutting concern --
idempotency-key comparison, conditional writes, whiteout-aware fallback lookup --
lives here, not in any backend. Backends only implement raw CRUD (see backend.py)."""

import hashlib
import json

from .backend import ResourceBackend
from .models import Depth, Found, IdempotencyRecord, Masked, Missing, Resource, ResourceInfo, ResourcePage, WriteOptions
from .path import ResourcePath
from ...errors import IdempotencyConflictError, ResourcePreconditionFailedError, ResourceReadOnlyError


def _request_hash(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


class ResourceStore:
    def __init__(self, *, primary: ResourceBackend, overlays: "tuple[ResourceBackend, ...]" = ()) -> None:
        self._primary = primary
        self._overlays = overlays

    async def _lookup_chain(self, path: ResourcePath):
        """Reader-facing three-state resolution across Primary then Overlays. A
        Masked result at Primary stops the search (spec section 14.6 rule 3)."""
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

    async def stat(self, path: ResourcePath) -> "ResourceInfo | None":
        resource = await self.get(path)
        return resource.info if resource is not None else None

    async def propfind(self, path: ResourcePath, *, depth: Depth = Depth.ONE, limit: int = 100, cursor: "str | None" = None) -> ResourcePage:
        """List resources under `path`, merging Primary and Overlay results.

        NOTE: cursor-based continuation is not yet implemented in Phase 1 --
        `cursor` is accepted for forward API compatibility but ignored; callers
        must not rely on being able to page past `limit` items. Real pagination
        is deferred to a later phase.
        """
        merged: "dict[str, ResourceInfo]" = {}
        for overlay in reversed(self._overlays):
            page = await overlay.raw_propfind(path, depth=depth, limit=limit, cursor=cursor)
            for info in page.items:
                merged[info.path.value] = info
        primary_page = await self._primary.raw_propfind(path, depth=depth, limit=limit, cursor=cursor)
        primary_paths = {info.path.value for info in primary_page.items}
        for info in primary_page.items:
            merged[info.path.value] = info
        for overlay_only_path in list(merged):
            if overlay_only_path in primary_paths:
                continue
            primary_lookup = await self._primary.raw_get(ResourcePath(overlay_only_path), include_content=False)
            if isinstance(primary_lookup, Masked):
                del merged[overlay_only_path]
        items = tuple(merged[key] for key in sorted(merged))
        return ResourcePage(items=items[:limit], cursor=None)

    def _require_writable_primary(self) -> None:
        if self._primary.readonly:
            raise ResourceReadOnlyError("primary backend is read-only")

    async def _check_idempotency(self, operation: str, key: "str | None", request_hash: str) -> "IdempotencyRecord | None":
        if key is None:
            return None
        record = await self._primary.get_idempotency(f"{operation}:{key}")
        if record is not None and record.request_hash != request_hash:
            raise IdempotencyConflictError(f"idempotency key {key!r} reused with a different request")
        return record

    async def _save_idempotency(self, operation: str, key: "str | None", request_hash: str, result: "ResourceInfo | None") -> None:
        if key is None:
            return
        await self._primary.put_idempotency(
            IdempotencyRecord(key=f"{operation}:{key}", request_hash=request_hash, result=result)
        )

    async def put(self, path: ResourcePath, content: bytes, *, options: WriteOptions = WriteOptions()) -> Resource:
        self._require_writable_primary()
        req_hash = _request_hash(path.value.encode(), content, json.dumps(dict(options.metadata), sort_keys=True).encode())
        # TOCTOU fix (spec section 16): when the primary backend implements the
        # atomic checked operation, delegate precondition + idempotency + mutate
        # to it as a single atomic call so a concurrent writer cannot interleave
        # the three steps. The Memory backend does not implement it, so the
        # legacy 3-step orchestration below remains as the fallback.
        if hasattr(self._primary, "raw_put_checked"):
            return await self._primary.raw_put_checked(path, content, options=options, request_hash=req_hash)

        existing_record = await self._check_idempotency("put", options.idempotency_key, req_hash)
        if existing_record is not None:
            info = existing_record.result
            current_lookup = await self._primary.raw_get(path)
            content_bytes = current_lookup.resource.content if isinstance(current_lookup, Found) else content
            return Resource(info=info, content=content_bytes)

        current = await self._lookup_chain(path)
        if options.if_none_match and isinstance(current, Found):
            raise ResourcePreconditionFailedError(f"resource already exists: {path}")
        if options.if_match is not None:
            if not isinstance(current, Found) or current.resource.info.etag != options.if_match:
                raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")

        primary_state = await self._primary.raw_get(path)
        if isinstance(primary_state, Found) and primary_state.resource.content == content and dict(primary_state.resource.info.metadata) == dict(options.metadata):
            info = primary_state.resource.info
        else:
            info = await self._primary.raw_put(path, content, content_type=options.content_type, metadata=options.metadata)

        await self._save_idempotency("put", options.idempotency_key, req_hash, info)
        return Resource(info=info, content=content)

    async def delete(self, path: ResourcePath, *, options: WriteOptions = WriteOptions()) -> None:
        self._require_writable_primary()
        req_hash = _request_hash(path.value.encode())
        # TOCTOU fix: delegate to the atomic checked op when available (see put).
        if hasattr(self._primary, "raw_delete_checked"):
            await self._primary.raw_delete_checked(path, options=options, request_hash=req_hash)
            return

        existing_record = await self._check_idempotency("delete", options.idempotency_key, req_hash)
        if existing_record is not None:
            return

        reader_lookup = await self._lookup_chain(path)
        primary_raw = await self._primary.raw_get(path)
        if options.if_match is not None:
            if not isinstance(reader_lookup, Found) or reader_lookup.resource.info.etag != options.if_match:
                raise ResourcePreconditionFailedError(f"if-match precondition failed: {path}")

        result_info = None
        already_masked = isinstance(reader_lookup, Missing) and isinstance(primary_raw, Masked)
        never_existed = isinstance(reader_lookup, Missing) and not already_masked
        if not (already_masked or never_existed):
            result_info = await self._primary.raw_delete(path)

        await self._save_idempotency("delete", options.idempotency_key, req_hash, result_info)

    async def move(self, src: ResourcePath, dst: ResourcePath, *, options: WriteOptions = WriteOptions()) -> Resource:
        self._require_writable_primary()
        source = await self.get(src)
        if source is None:
            raise ResourcePreconditionFailedError(f"cannot move missing resource: {src}")
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
