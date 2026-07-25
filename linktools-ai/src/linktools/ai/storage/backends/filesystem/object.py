#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemObjectBackend: filesystem-backed ObjectWriterBackend + history.

Every version of a key is an immutable pair of sidecar files under
``.storage/history/<encoded-key>/<version>.{json,bin}`` -- a delete appends a
tombstone version (json only, no ``.bin``) rather than removing anything, so
per-key history is a free byproduct of "never overwrite a version file, only
append the next one." The current live state is always the LATEST version
(tombstone or not); there is no separate "current" data file to keep in sync.

Atomic writes via temp-file-then-``os.replace``; every path is resolved
through :func:`resolve_secure_path` (a per-component lstat walk from the
trusted root) so a symlink planted anywhere in the chain is caught before a
read or write follows it.

This backend does NOT implement ``TransactionalObjectBackend`` --
multi-object transactions are refused (``StorageTransactionNotSupportedError``)
rather than faked; only the single-key checked operations are atomic (via an
in-process lock -- cross-process races on the same root remain a documented
limitation)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ...object.errors import (
    StorageIdempotencyConflictError,
    StorageObjectNotFoundError,
    StoragePreconditionFailedError,
)
from ...object.models import (
    Depth,
    Found,
    Masked,
    Missing,
    ObjectInfo,
    ObjectPage,
    ObjectVersionPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)
from ._path_security import SymlinkPolicy, open_temp_nofollow, resolve_secure_path


def _encoded(key: StorageKey) -> str:
    # Percent-encode so the mapping from StorageKey -> directory name is
    # reversible: "/" and "%" are escaped, so distinct keys can never
    # collide on one directory.
    return urllib.parse.quote(key.value.strip("/") or "__root__", safe="")


def _matches_depth(prefix: StorageKey, candidate: StorageKey, depth: "Depth") -> bool:
    if not candidate.is_under(prefix):
        return False
    if depth is Depth.INFINITY:
        return True
    if prefix.is_root:
        rel_depth = len(candidate._segments)
    else:
        if candidate.value == prefix.value:
            rel_depth = 0
        else:
            rel_depth = len(candidate._segments) - len(prefix._segments)
    if depth is Depth.ZERO:
        return rel_depth == 0
    return rel_depth <= 1


@dataclass
class _IdempotencyRecord:
    request_hash: str
    result_version: "int | None"  # None means "no live object" (a delete)


class FilesystemObjectBackend:
    """Not a ``TransactionalObjectBackend`` (no multi-object transaction);
    IS a ``VersionedObjectBackend`` (native per-key history is intrinsic to
    the append-only version-file layout)."""

    backend_id: str = "primary"

    def __init__(
        self,
        *,
        root: Path,
        symlink_policy: SymlinkPolicy = SymlinkPolicy.DENY,
    ) -> None:
        self._root = Path(root)
        self._symlink_policy = symlink_policy
        self._history_dir = self._root / ".storage" / "history"
        self._idempotency_dir = self._root / ".storage" / "idempotency"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._idempotency_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._revision_cache: "int | None" = None

    # --- path helpers ----------------------------------------------------

    def _key_history_dir(self, key: StorageKey) -> Path:
        return resolve_secure_path(
            self._root, ".storage", "history", _encoded(key), policy=self._symlink_policy
        )

    def _version_json_path(self, key: StorageKey, version: int) -> Path:
        return self._key_history_dir(key) / f"{version}.json"

    def _version_bin_path(self, key: StorageKey, version: int) -> Path:
        return self._key_history_dir(key) / f"{version}.bin"

    def _revision_path(self) -> Path:
        return resolve_secure_path(
            self._root, ".storage", "revision", policy=self._symlink_policy
        )

    def _idempotency_path(self, op_key: str) -> Path:
        return resolve_secure_path(
            self._root,
            ".storage",
            "idempotency",
            urllib.parse.quote(op_key, safe="") + ".json",
            policy=self._symlink_policy,
        )

    def _atomic_write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        resolve_secure_path(
            self._root, *path.relative_to(self._root).parts, policy=self._symlink_policy
        )
        fd, tmp_path = open_temp_nofollow(path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # --- revision ----------------------------------------------------------

    def _read_revision(self) -> int:
        p = self._revision_path()
        if not p.exists():
            return 0
        return int(p.read_text().strip() or "0")

    def _bump_revision(self) -> int:
        value = self._read_revision() + 1
        self._atomic_write(self._revision_path(), str(value).encode("utf-8"))
        self._revision_cache = value
        return value

    async def revision(self) -> str:
        if self._revision_cache is None:
            self._revision_cache = await asyncio.to_thread(self._read_revision)
        return str(self._revision_cache)

    # --- version read/write helpers -----------------------------------------

    def _list_version_numbers(self, key: StorageKey) -> "list[int]":
        d = self._key_history_dir(key)
        if not d.exists():
            return []
        out = []
        for p in d.glob("*.json"):
            try:
                out.append(int(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def _latest_version_number(self, key: StorageKey) -> "int | None":
        versions = self._list_version_numbers(key)
        return versions[-1] if versions else None

    def _read_version_json(self, key: StorageKey, version: int) -> dict:
        return json.loads(self._version_json_path(key, version).read_text())

    def _info_from_json(self, key: StorageKey, raw: dict, revision: "int | None") -> ObjectInfo:
        return ObjectInfo(
            key=key,
            etag=raw["etag"],
            version=raw["version"],
            commit_revision=revision,
            content_type=raw["content_type"],
            size=raw["size"],
            modified_at=datetime.fromisoformat(raw["modified_at"]),
            metadata=raw.get("metadata") or {},
        )

    def _write_version(
        self,
        key: StorageKey,
        *,
        version: int,
        content: "bytes | None",
        content_type: "str | None",
        metadata: "dict",
        tombstone: bool,
    ) -> ObjectInfo:
        revision = self._bump_revision()
        raw = {
            "version": version,
            "etag": sha256(content).hexdigest() if content is not None else "",
            "content_type": content_type,
            "size": len(content) if content is not None else 0,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata),
            "tombstone": tombstone,
            "commit_revision": revision,
        }
        self._atomic_write(
            self._version_json_path(key, version), json.dumps(raw).encode("utf-8")
        )
        if content is not None:
            self._atomic_write(self._version_bin_path(key, version), content)
        return self._info_from_json(key, raw, revision)

    def _live_version(self, key: StorageKey) -> "tuple[int, dict] | None":
        """The latest version's (number, raw json), or None if no versions
        exist at all. Caller checks ``raw["tombstone"]`` to distinguish a
        live object from a masked (deleted) one."""
        latest = self._latest_version_number(key)
        if latest is None:
            return None
        return latest, self._read_version_json(key, latest)

    # --- ObjectReaderBackend -------------------------------------------------

    def _raw_get_sync(self, key: StorageKey, *, include_content: bool):
        live = self._live_version(key)
        if live is None:
            return Missing
        version, raw = live
        if raw["tombstone"]:
            return Masked(key=key, version=version, commit_revision=raw["commit_revision"])
        content = b""
        if include_content:
            content = self._version_bin_path(key, version).read_bytes()
        return Found(StoredObject(info=self._info_from_json(key, raw, raw["commit_revision"]), content=content))

    async def raw_get(self, key: StorageKey, *, include_content: bool = True):
        return await asyncio.to_thread(self._raw_get_sync, key, include_content=include_content)

    def _raw_stat_sync(self, key: StorageKey) -> "ObjectInfo | None":
        live = self._live_version(key)
        if live is None or live[1]["tombstone"]:
            return None
        version, raw = live
        return self._info_from_json(key, raw, raw["commit_revision"])

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None":
        return await asyncio.to_thread(self._raw_stat_sync, key)

    def _raw_list_sync(
        self, prefix: StorageKey, *, depth: "Depth", limit: int, cursor: "str | None"
    ) -> ObjectPage:
        if not self._history_dir.exists():
            return ObjectPage(items=(), next_cursor=None)
        candidates: "list[StorageKey]" = []
        for entry in self._history_dir.iterdir():
            if not entry.is_dir():
                continue
            key = StorageKey("/" + urllib.parse.unquote(entry.name))
            if not _matches_depth(prefix, key, depth):
                continue
            live = self._live_version(key)
            if live is None or live[1]["tombstone"]:
                continue
            if cursor is not None and key.value <= cursor:
                continue
            candidates.append(key)
        candidates.sort(key=lambda k: k.value)
        page_keys = candidates[: limit + 1]
        items = []
        for key in page_keys[:limit]:
            version, raw = self._live_version(key)
            items.append(self._info_from_json(key, raw, raw["commit_revision"]))
        next_cursor = page_keys[limit - 1].value if len(page_keys) > limit else None
        return ObjectPage(items=tuple(items), next_cursor=next_cursor)

    async def raw_list(
        self, prefix: StorageKey, *, depth: "Depth", limit: int, cursor: "str | None"
    ) -> ObjectPage:
        return await asyncio.to_thread(
            self._raw_list_sync, prefix, depth=depth, limit=limit, cursor=cursor
        )

    # --- idempotency ---------------------------------------------------------

    def _read_idempotency(self, op_key: str) -> "_IdempotencyRecord | None":
        p = self._idempotency_path(op_key)
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        return _IdempotencyRecord(
            request_hash=raw["request_hash"], result_version=raw["result_version"]
        )

    def _write_idempotency(self, op_key: str, record: _IdempotencyRecord) -> None:
        raw = {"request_hash": record.request_hash, "result_version": record.result_version}
        self._atomic_write(self._idempotency_path(op_key), json.dumps(raw).encode("utf-8"))

    # --- ObjectWriterBackend -------------------------------------------------

    def _raw_put_checked_sync(
        self, key: StorageKey, content: bytes, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        with self._lock:
            idem_key = f"put:{key.value}:{options.idempotency_key}" if options.idempotency_key else None
            if idem_key is not None:
                record = self._read_idempotency(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    if record.result_version is None:
                        raise StorageObjectNotFoundError(key.value)
                    raw = self._read_version_json(key, record.result_version)
                    content_bytes = self._version_bin_path(key, record.result_version).read_bytes()
                    return StoredObject(
                        info=self._info_from_json(key, raw, raw["commit_revision"]), content=content_bytes
                    )
            live = self._live_version(key)
            live_info = None
            if live is not None and not live[1]["tombstone"]:
                live_info = live[1]
            if options.if_none_match and live_info is not None:
                raise StoragePreconditionFailedError(f"if_none_match failed: {key.value!r} already exists")
            if options.if_match is not None:
                if live_info is None or live_info["etag"] != options.if_match:
                    raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} etag mismatch")
            next_version = (live[0] + 1) if live is not None else 1
            info = self._write_version(
                key,
                version=next_version,
                content=content,
                content_type=options.content_type,
                metadata=dict(options.metadata or {}),
                tombstone=False,
            )
            if idem_key is not None:
                self._write_idempotency(idem_key, _IdempotencyRecord(request_hash=request_hash, result_version=info.version))
            return StoredObject(info=info, content=content)

    async def raw_put_checked(
        self, key: StorageKey, content: bytes, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        return await asyncio.to_thread(
            self._raw_put_checked_sync, key, content, options=options, request_hash=request_hash
        )

    def _raw_delete_checked_sync(
        self, key: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> None:
        with self._lock:
            idem_key = f"delete:{key.value}:{options.idempotency_key}" if options.idempotency_key else None
            if idem_key is not None:
                record = self._read_idempotency(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    return None
            live = self._live_version(key)
            if live is None or live[1]["tombstone"]:
                # Deleting a missing key is a no-op (no tombstone, no bump).
                if idem_key is not None:
                    self._write_idempotency(idem_key, _IdempotencyRecord(request_hash=request_hash, result_version=None))
                return None
            if options.if_match is not None and live[1]["etag"] != options.if_match:
                raise StoragePreconditionFailedError(f"if_match failed: {key.value!r} etag mismatch")
            next_version = live[0] + 1
            self._write_version(
                key, version=next_version, content=None, content_type=None, metadata={}, tombstone=True
            )
            if idem_key is not None:
                self._write_idempotency(idem_key, _IdempotencyRecord(request_hash=request_hash, result_version=None))
            return None

    async def raw_delete_checked(
        self, key: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> None:
        return await asyncio.to_thread(
            self._raw_delete_checked_sync, key, options=options, request_hash=request_hash
        )

    def _raw_move_checked_sync(
        self, source: StorageKey, target: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        with self._lock:
            idem_key = f"move:{source.value}:{target.value}:{options.idempotency_key}" if options.idempotency_key else None
            if idem_key is not None:
                record = self._read_idempotency(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise StorageIdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} replayed with a different request"
                        )
                    if record.result_version is None:
                        raise StorageObjectNotFoundError(source.value)
                    raw = self._read_version_json(target, record.result_version)
                    content_bytes = self._version_bin_path(target, record.result_version).read_bytes()
                    return StoredObject(
                        info=self._info_from_json(target, raw, raw["commit_revision"]), content=content_bytes
                    )
            src_live = self._live_version(source)
            if src_live is None or src_live[1]["tombstone"]:
                raise StorageObjectNotFoundError(source.value)
            src_version, src_raw = src_live
            content = self._version_bin_path(source, src_version).read_bytes()

            tgt_live = self._live_version(target)
            tgt_info = None
            if tgt_live is not None and not tgt_live[1]["tombstone"]:
                tgt_info = tgt_live[1]
            if options.if_none_match and tgt_info is not None:
                raise StoragePreconditionFailedError(f"if_none_match failed: {target.value!r} already exists")
            if options.if_match is not None:
                if tgt_info is None or tgt_info["etag"] != options.if_match:
                    raise StoragePreconditionFailedError(f"if_match failed: {target.value!r} etag mismatch")

            next_target_version = (tgt_live[0] + 1) if tgt_live is not None else 1
            # ONE revision bump for the whole move: write the source
            # tombstone WITHOUT its own bump, then the target write bumps.
            next_source_version = src_version + 1
            revision = self._bump_revision()
            source_raw = {
                "version": next_source_version,
                "etag": "",
                "content_type": None,
                "size": 0,
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
                "tombstone": True,
                "commit_revision": revision,
            }
            self._atomic_write(
                self._version_json_path(source, next_source_version),
                json.dumps(source_raw).encode("utf-8"),
            )
            target_raw = {
                "version": next_target_version,
                "etag": src_raw["etag"],
                "content_type": src_raw["content_type"],
                "size": src_raw["size"],
                "modified_at": datetime.now(timezone.utc).isoformat(),
                "metadata": dict(src_raw.get("metadata") or {}),
                "tombstone": False,
                "commit_revision": revision,
            }
            self._atomic_write(
                self._version_json_path(target, next_target_version),
                json.dumps(target_raw).encode("utf-8"),
            )
            self._atomic_write(self._version_bin_path(target, next_target_version), content)
            info = self._info_from_json(target, target_raw, revision)
            if idem_key is not None:
                self._write_idempotency(idem_key, _IdempotencyRecord(request_hash=request_hash, result_version=info.version))
            return StoredObject(info=info, content=content)

    async def raw_move_checked(
        self, source: StorageKey, target: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        return await asyncio.to_thread(
            self._raw_move_checked_sync, source, target, options=options, request_hash=request_hash
        )

    # --- VersionedObjectBackend (native, free from the version-file layout) --

    def _raw_get_version_sync(self, key: StorageKey, version: int) -> "StoredObject | None":
        p = self._version_json_path(key, version)
        if not p.exists():
            return None
        raw = json.loads(p.read_text())
        if raw["tombstone"]:
            return None
        content = self._version_bin_path(key, version).read_bytes()
        return StoredObject(info=self._info_from_json(key, raw, raw["commit_revision"]), content=content)

    async def raw_get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        return await asyncio.to_thread(self._raw_get_version_sync, key, version)

    def _raw_get_at_revision_sync(self, key: StorageKey, revision: int) -> "StoredObject | None":
        best: "tuple[int, dict] | None" = None
        for v in self._list_version_numbers(key):
            raw = self._read_version_json(key, v)
            if raw["commit_revision"] is not None and raw["commit_revision"] <= revision:
                best = (v, raw)
            else:
                break
        if best is None or best[1]["tombstone"]:
            return None
        version, raw = best
        content = self._version_bin_path(key, version).read_bytes()
        return StoredObject(info=self._info_from_json(key, raw, raw["commit_revision"]), content=content)

    async def raw_get_at_revision(self, key: StorageKey, revision: int) -> "StoredObject | None":
        return await asyncio.to_thread(self._raw_get_at_revision_sync, key, revision)

    def _raw_list_versions_sync(
        self, key: StorageKey, *, limit: int, cursor: "str | None"
    ) -> ObjectVersionPage:
        versions = self._list_version_numbers(key)
        start = 0 if cursor is None else int(cursor)
        page_versions = versions[start : start + limit]
        items = []
        for v in page_versions:
            raw = self._read_version_json(key, v)
            items.append(self._info_from_json(key, raw, raw["commit_revision"]))
        next_start = start + len(page_versions)
        next_cursor = None if next_start >= len(versions) else str(next_start)
        return ObjectVersionPage(items=tuple(items), next_cursor=next_cursor)

    async def raw_list_versions(
        self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None
    ) -> ObjectVersionPage:
        return await asyncio.to_thread(self._raw_list_versions_sync, key, limit=limit, cursor=cursor)

    def _raw_list_at_revision_sync(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        if not self._history_dir.exists():
            return ()
        out: "list[ObjectInfo]" = []
        for entry in sorted(self._history_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            key = StorageKey("/" + urllib.parse.unquote(entry.name))
            if not key.is_under(prefix):
                continue
            best: "tuple[int, dict] | None" = None
            for v in self._list_version_numbers(key):
                raw = self._read_version_json(key, v)
                if raw["commit_revision"] is not None and raw["commit_revision"] <= revision:
                    best = (v, raw)
                else:
                    break
            if best is not None and not best[1]["tombstone"]:
                out.append(self._info_from_json(key, best[1], best[1]["commit_revision"]))
        return tuple(out)

    async def raw_list_at_revision(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        return await asyncio.to_thread(self._raw_list_at_revision_sync, prefix, revision)


class FilesystemObjectStore:
    """Convenience: an ObjectStore pre-wired to a fresh FilesystemObjectBackend."""

    def __init__(self, *, root: Path, symlink_policy: SymlinkPolicy = SymlinkPolicy.DENY) -> None:
        from ...object.store import ObjectStore

        self._backend = FilesystemObjectBackend(root=root, symlink_policy=symlink_policy)
        self._store = ObjectStore(primary=self._backend)

    @property
    def backend(self) -> FilesystemObjectBackend:
        return self._backend

    async def get(self, key: StorageKey) -> "StoredObject | None":
        return await self._store.get(key)

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        return await self._store.stat(key)

    async def list(self, prefix: StorageKey, **kwargs) -> ObjectPage:
        return await self._store.list(prefix, **kwargs)

    async def revision(self) -> str:
        return await self._store.revision()

    async def put(self, key: StorageKey, content: bytes, **kwargs) -> StoredObject:
        return await self._store.put(key, content, **kwargs)

    async def delete(self, key: StorageKey, **kwargs) -> None:
        await self._store.delete(key, **kwargs)

    async def move(self, source: StorageKey, target: StorageKey, **kwargs) -> StoredObject:
        return await self._store.move(source, target, **kwargs)

    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        return await self._backend.raw_get_version(key, version)

    async def list_versions(self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None) -> ObjectVersionPage:
        return await self._backend.raw_list_versions(key, limit=limit, cursor=cursor)
