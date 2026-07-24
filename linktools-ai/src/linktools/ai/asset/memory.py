#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryAssetBackend: a dependency-free AssetWriterBackend used to exercise
the Protocol contract and as one of the parametrized backends in the AssetStore
contract-test suite. Every checked operation runs under a single
``asyncio.Lock`` so precondition + idempotency + mutate are serialized
process-local (the best atomicity an in-memory store can offer)."""

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Mapping

from ..errors import (
    IdempotencyConflictError,
    AssetPreconditionFailedError,
)
from .models import (
    Depth,
    Found,
    IdempotencyRecord,
    Masked,
    Missing,
    Asset,
    AssetInfo,
    AssetKind,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath, _require_persistable_path, matches_asset_depth


class MemoryAssetBackend:
    def __init__(self) -> None:
        self._entries: "dict[str, tuple[bytes, AssetInfo]]" = {}
        self._whiteouts: "dict[str, int]" = {}
        self._idempotency: "dict[str, IdempotencyRecord]" = {}
        self._revision = 0
        # Stable id the AssetStore overrides with the canonical primary/overlay
        # tag; defaults to the backend's origin so it is never blank.
        self.backend_id = "memory"
        # One lock guards every checked op so the precondition + idempotency +
        # mutate steps cannot interleave across concurrent coroutines.
        self._lock = asyncio.Lock()

    async def raw_get(self, path: AssetPath, *, include_content: bool = True):
        _require_persistable_path(path)
        key = path.value
        if key in self._entries:
            content, info = self._entries[key]
            if not include_content:
                return Found(asset=Asset(info=info, content=b""))
            return Found(asset=Asset(info=info, content=content))
        if key in self._whiteouts:
            return Masked(path=path, version=self._whiteouts[key])
        return Missing()

    async def raw_stat(self, path: AssetPath) -> "AssetInfo | None":
        entry = self._entries.get(path.value)
        return entry[1] if entry is not None else None

    async def raw_list(
        self, path: AssetPath, *, depth: Depth, limit: int, cursor: "str | None"
    ) -> AssetPage:
        items = []
        for key, (_content, info) in sorted(self._entries.items()):
            if not matches_asset_depth(path, AssetPath(key), depth):
                continue
            if cursor is not None and key <= cursor:
                continue
            items.append(info)
            if len(items) > limit:
                break
        next_cursor = items[limit - 1].path.value if len(items) > limit else None
        return AssetPage(items=tuple(items[:limit]), cursor=next_cursor)

    async def revision(self) -> str:
        return str(self._revision)

    async def raw_put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ) -> AssetInfo:
        # Unconditional convenience put (not part of the Writer Protocol; tests
        # and direct callers use it). The atomic checked path is
        # raw_put_checked, which the AssetStore uses.
        version = self._next_version(path.value)
        info = self._build_info(
            path, content, content_type=content_type, metadata=metadata, version=version
        )
        self._entries[path.value] = (content, info)
        self._whiteouts.pop(path.value, None)
        self._revision += 1
        return info

    async def raw_delete(self, path: AssetPath) -> "AssetInfo | None":
        removed = self._entries.pop(path.value, None)
        prior_version = removed[1].version if removed else self._whiteouts.get(path.value, 0)
        self._whiteouts[path.value] = prior_version + 1
        self._revision += 1
        return removed[1] if removed else None

    # -- Atomic checked operations (serialized under self._lock) -------------

    def _next_version(self, key: str) -> int:
        prior_entry = self._entries.get(key)
        prior_version = prior_entry[1].version if prior_entry else 0
        return max(prior_version, self._whiteouts.get(key, 0)) + 1

    def _build_info(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
        version: int,
    ) -> AssetInfo:
        return AssetInfo(
            path=path,
            kind=AssetKind.FILE,
            etag=hashlib.sha256(content).hexdigest(),
            version=version,
            content_type=content_type,
            size=len(content),
            modified_at=datetime.now(timezone.utc),
            metadata=dict(metadata),
        )

    async def raw_put_checked(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset:
        _require_persistable_path(path)
        async with self._lock:
            idem_key = options.idempotency_key
            if idem_key is not None:
                cached = self._idempotency.get(f"put:{idem_key}")
                if cached is not None:
                    if cached.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {idem_key!r} reused with a different request"
                        )
                    entry = self._entries.get(path.value)
                    bytes_ = entry[0] if entry is not None else content
                    return Asset(info=cached.result or self._build_info(
                        path, content, content_type=options.content_type,
                        metadata=options.metadata, version=1), content=bytes_)
            lookup = await self.raw_get(path)
            if options.if_none_match and isinstance(lookup, Found):
                raise AssetPreconditionFailedError(f"asset already exists: {path}")
            if options.if_match is not None:
                if not isinstance(lookup, Found) or lookup.asset.info.etag != options.if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
            # Idempotent no-op: identical content/metadata/type on a live entry.
            if (
                isinstance(lookup, Found)
                and lookup.asset.content == content
                and dict(lookup.asset.info.metadata) == dict(options.metadata)
                and lookup.asset.info.content_type == options.content_type
            ):
                info = lookup.asset.info
            else:
                version = self._next_version(path.value)
                info = self._build_info(
                    path, content, content_type=options.content_type,
                    metadata=options.metadata, version=version,
                )
                self._entries[path.value] = (content, info)
                self._whiteouts.pop(path.value, None)
                self._revision += 1
            if idem_key is not None:
                self._idempotency[f"put:{idem_key}"] = IdempotencyRecord(
                    key=f"put:{idem_key}", request_hash=request_hash, result=info
                )
            return Asset(info=info, content=content)

    async def raw_delete_checked(
        self,
        path: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        _require_persistable_path(path)
        async with self._lock:
            idem_key = options.idempotency_key
            if idem_key is not None:
                cached = self._idempotency.get(f"delete:{idem_key}")
                if cached is not None:
                    if cached.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {idem_key!r} reused with a different request"
                        )
                    return
            lookup = await self.raw_get(path)
            if options.if_match is not None:
                if not isinstance(lookup, Found) or lookup.asset.info.etag != options.if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
            removed = self._entries.pop(path.value, None)
            prior_version = removed[1].version if removed else self._whiteouts.get(path.value, 0)
            self._whiteouts[path.value] = prior_version + 1
            self._revision += 1
            if idem_key is not None:
                self._idempotency[f"delete:{idem_key}"] = IdempotencyRecord(
                    key=f"delete:{idem_key}",
                    request_hash=request_hash,
                    result=removed[1] if removed is not None else None,
                )

    async def raw_move_checked(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset:
        _require_persistable_path(source)
        _require_persistable_path(target)
        async with self._lock:
            idem_key = options.idempotency_key
            if idem_key is not None:
                cached = self._idempotency.get(f"move:{idem_key}")
                if cached is not None:
                    if cached.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {idem_key!r} reused with a different request"
                        )
                    entry = self._entries.get(target.value)
                    bytes_ = entry[0] if entry is not None else b""
                    return Asset(
                        info=cached.result
                        or self._build_info(target, b"", content_type=None, metadata={}, version=1),
                        content=bytes_,
                    )
            src_entry = self._entries.get(source.value)
            if src_entry is None:
                raise AssetPreconditionFailedError(
                    f"cannot move missing asset: {source}"
                )
            src_content, src_info = src_entry
            target_lookup = await self.raw_get(target)
            if options.if_none_match and isinstance(target_lookup, Found):
                raise AssetPreconditionFailedError(f"asset already exists: {target}")
            if options.if_match is not None:
                if (
                    not isinstance(target_lookup, Found)
                    or target_lookup.asset.info.etag != options.if_match
                ):
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {target}"
                    )
            target_version = self._next_version(target.value)
            target_info = AssetInfo(
                path=target,
                kind=src_info.kind,
                etag=hashlib.sha256(src_content).hexdigest(),
                version=target_version,
                content_type=src_info.content_type,
                size=len(src_content),
                modified_at=datetime.now(timezone.utc),
                metadata=dict(src_info.metadata),
            )
            self._entries[target.value] = (src_content, target_info)
            self._entries.pop(source.value, None)
            self._whiteouts[source.value] = src_info.version + 1
            self._revision += 1
            if idem_key is not None:
                self._idempotency[f"move:{idem_key}"] = IdempotencyRecord(
                    key=f"move:{idem_key}", request_hash=request_hash, result=target_info
                )
            return Asset(info=target_info, content=src_content)
