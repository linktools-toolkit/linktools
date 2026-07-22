#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileAssetBackend: filesystem-backed AssetBackend. Flat filename mapping
(path segments joined with "__") avoids managing nested directories; atomic writes
via temp-file-then-os.replace; whiteouts/idempotency/revision are separate small
JSON files under .asset/ so a crash mid-write cannot corrupt unrelated assets.

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The in-process ``threading.Lock`` is held inside the sync core (running in a
worker thread), so it still serializes the checked operations within one
process while the event loop continues running other coroutines."""

import asyncio
import json
import os
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .models import (
    Depth,
    Found,
    IdempotencyRecord,
    Masked,
    Missing,
    MoveResult,
    Asset,
    AssetInfo,
    AssetKind,
    AssetLookupInfo,
    AssetPage,
    WriteOptions,
)
from .path import AssetPath
from ..errors import (
    IdempotencyConflictError,
    InvalidAssetPathError,
    AssetPreconditionFailedError,
)


class SymlinkPolicy(str, Enum):
    DENY = "deny"
    ALLOW_INTERNAL = "allow_internal"


def _filename(path: AssetPath) -> str:
    # Percent-encode so the mapping from AssetPath -> filename is reversible:
    # "/" and "%" are escaped, so distinct paths can never collide on one filename.
    return urllib.parse.quote(path.value.strip("/"), safe="")


def _path_from_filename(stem: str) -> AssetPath:
    return AssetPath("/" + urllib.parse.unquote(stem))


class FileAssetBackend:
    def __init__(
        self,
        *,
        root: Path,
        readonly: bool = False,
        symlink_policy: SymlinkPolicy = SymlinkPolicy.DENY,
    ) -> None:
        self.readonly = readonly
        self._symlink_policy = symlink_policy
        self._root = Path(root)
        self._data_dir = self._root / "data"
        self._meta_dir = self._root / ".assets" / "metadata"
        self._whiteout_dir = self._root / ".assets" / "whiteouts"
        self._idempotency_dir = self._root / ".assets" / "idempotency"
        self._revision_file = self._root / ".assets" / "revision"
        # In-process lock serializing the checked (precondition+idempotency+
        # mutate) operations. The filesystem itself is not transactional, so
        # this lock is the best atomicity FileAssetBackend can offer: it
        # prevents two concurrent checked-puts within ONE process from
        # interleaving the three steps. Cross-process races remain (a different
        # process writing the same backend root can still interleave) -- that is
        # a documented limitation; true cross-process atomicity requires the
        # SqlAlchemy backend. The lock is held inside the ``_*_sync`` cores
        # (which run in a worker thread via ``asyncio.to_thread``), so the
        # event loop is never blocked either by the I/O or by another worker
        # waiting on this lock.
        self._lock = threading.Lock()
        for d in (
            self._data_dir,
            self._meta_dir,
            self._whiteout_dir,
            self._idempotency_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _resolve(self, directory: Path, filename: str) -> Path:
        resolved = (directory / filename).resolve()
        if self._symlink_policy == SymlinkPolicy.DENY and resolved.is_symlink():
            raise InvalidAssetPathError(f"symlink not allowed: {resolved}")
        if (
            directory.resolve() not in resolved.parents
            and resolved != directory.resolve()
        ):
            raise InvalidAssetPathError(f"path escapes backend root: {resolved}")
        return resolved

    def _atomic_write(self, path: Path, content: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_name, path)
        finally:
            # finally so cancellation propagates naturally while still removing
            # the temp file on any failure (no-op on success once replace ran).
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _read_revision(self) -> int:
        if not self._revision_file.exists():
            return 0
        return int(self._revision_file.read_text().strip() or "0")

    def _bump_revision(self) -> int:
        value = self._read_revision() + 1
        self._atomic_write(self._revision_file, str(value).encode("utf-8"))
        return value

    def _meta_path(self, path: AssetPath) -> Path:
        return self._resolve(self._meta_dir, _filename(path) + ".json")

    def _data_path(self, path: AssetPath) -> Path:
        return self._resolve(self._data_dir, _filename(path))

    def _whiteout_path(self, path: AssetPath) -> Path:
        return self._resolve(self._whiteout_dir, _filename(path) + ".json")

    def _idempotency_path(self, key: str) -> Path:
        return self._resolve(self._idempotency_dir, key.replace("/", "__") + ".json")

    def _load_info(self, path: AssetPath) -> "AssetInfo | None":
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return None
        raw = json.loads(meta_path.read_text())
        return AssetInfo(
            path=path,
            kind=AssetKind(raw["kind"]),
            etag=raw["etag"],
            version=raw["version"],
            content_type=raw["content_type"],
            size=raw["size"],
            modified_at=datetime.fromisoformat(raw["modified_at"]),
            metadata=raw["metadata"],
        )

    def _save_info(self, info: AssetInfo) -> None:
        raw = {
            "kind": info.kind.value,
            "etag": info.etag,
            "version": info.version,
            "content_type": info.content_type,
            "size": info.size,
            "modified_at": info.modified_at.isoformat(),
            "metadata": dict(info.metadata),
        }
        self._atomic_write(self._meta_path(info.path), json.dumps(raw).encode("utf-8"))

    def _raw_get_sync(self, path: AssetPath, *, include_content: bool):
        info = self._load_info(path)
        if info is not None:
            content = b""
            if include_content:
                content = self._data_path(path).read_bytes()
            return Found(asset=Asset(info=info, content=content))
        whiteout_path = self._whiteout_path(path)
        if whiteout_path.exists():
            version = json.loads(whiteout_path.read_text())["version"]
            return Masked(path=path, version=version)
        return Missing()

    async def raw_get(self, path: AssetPath, *, include_content: bool = True):
        return await asyncio.to_thread(
            self._raw_get_sync, path, include_content=include_content
        )

    async def raw_stat(self, path: AssetPath) -> "AssetLookupInfo | None":
        """Metadata-only stat: read only the metadata sidecar,
        never the data file. The sidecar carries path/kind/etag/version/
        content_type/size/modified_at/metadata -- everything stat() needs
        without touching the (potentially large) content blob."""
        return await asyncio.to_thread(self._load_info, path)

    def _raw_list_sync(
        self, path: AssetPath, *, depth: Depth, limit: int, cursor: "str | None"
    ) -> AssetPage:
        """Keyset pagination: metadata files are iterated in sorted
        order (so the global path order is stable), filtered by prefix, by
        ``path > cursor`` (resume point), and by depth, collecting limit+1 items
        so the (limit+1)th path becomes next_cursor. The cursor is the literal
        normalized path string of the last item returned."""
        prefix = path.value.rstrip("/") + "/"
        items = []
        for meta_file in sorted(self._meta_dir.glob("*.json")):
            candidate = _path_from_filename(meta_file.stem)
            if not candidate.value.startswith(prefix):
                continue
            if cursor is not None and candidate.value <= cursor:
                continue
            rest = candidate.value[len(prefix) :]
            if depth == Depth.ONE and "/" in rest:
                continue
            items.append(self._load_info(candidate))
            if len(items) > limit:
                break  # collected limit+1; enough to signal "more available"
        next_cursor = items[limit].path.value if len(items) > limit else None
        return AssetPage(items=tuple(items[:limit]), cursor=next_cursor)

    async def raw_list(
        self, path: AssetPath, *, depth: Depth, limit: int, cursor: "str | None"
    ) -> AssetPage:
        return await asyncio.to_thread(
            self._raw_list_sync,
            path,
            depth=depth,
            limit=limit,
            cursor=cursor,
        )

    def _raw_put_sync(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ):
        prior = self._load_info(path)
        whiteout_path = self._whiteout_path(path)
        prior_whiteout_version = 0
        if whiteout_path.exists():
            prior_whiteout_version = json.loads(whiteout_path.read_text())["version"]
        version = max(prior.version if prior else 0, prior_whiteout_version) + 1
        info = AssetInfo(
            path=path,
            kind=AssetKind.FILE,
            etag=sha256(content).hexdigest(),
            version=version,
            content_type=content_type,
            size=len(content),
            modified_at=datetime.now(timezone.utc),
            metadata=dict(metadata),
        )
        self._atomic_write(self._data_path(path), content)
        self._save_info(info)
        whiteout_path = self._whiteout_path(path)
        if whiteout_path.exists():
            whiteout_path.unlink()
        self._bump_revision()
        return info

    async def raw_put(
        self,
        path: AssetPath,
        content: bytes,
        *,
        content_type: "str | None",
        metadata: "Mapping[str, object]",
    ):
        return await asyncio.to_thread(
            self._raw_put_sync,
            path,
            content,
            content_type=content_type,
            metadata=metadata,
        )

    def _raw_delete_sync(self, path: AssetPath) -> "AssetInfo | None":
        info = self._load_info(path)
        prior_version = info.version if info else 0
        if info is not None:
            self._data_path(path).unlink(missing_ok=True)
            self._meta_path(path).unlink(missing_ok=True)
        whiteout_path = self._whiteout_path(path)
        existing_whiteout_version = 0
        if whiteout_path.exists():
            existing_whiteout_version = json.loads(whiteout_path.read_text())["version"]
        new_version = max(prior_version, existing_whiteout_version) + 1
        self._atomic_write(
            whiteout_path, json.dumps({"version": new_version}).encode("utf-8")
        )
        self._bump_revision()
        return info

    async def raw_delete(self, path: AssetPath) -> "AssetInfo | None":
        return await asyncio.to_thread(self._raw_delete_sync, path)

    async def revision(self) -> int:
        return await asyncio.to_thread(self._read_revision)

    def _read_idempotency_sync(self, key: str) -> "IdempotencyRecord | None":
        path = self._idempotency_path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        result = None
        if raw["result"] is not None:
            r = raw["result"]
            result = AssetInfo(
                path=AssetPath(r["path"]),
                kind=AssetKind(r["kind"]),
                etag=r["etag"],
                version=r["version"],
                content_type=r["content_type"],
                size=r["size"],
                modified_at=datetime.fromisoformat(r["modified_at"]),
                metadata=r["metadata"],
            )
        return IdempotencyRecord(
            key=raw["key"], request_hash=raw["request_hash"], result=result
        )

    def _write_idempotency_sync(self, record: IdempotencyRecord) -> None:
        result = None
        if record.result is not None:
            result = {
                "path": record.result.path.value,
                "kind": record.result.kind.value,
                "etag": record.result.etag,
                "version": record.result.version,
                "content_type": record.result.content_type,
                "size": record.result.size,
                "modified_at": record.result.modified_at.isoformat(),
                "metadata": dict(record.result.metadata),
            }
        raw = {"key": record.key, "request_hash": record.request_hash, "result": result}
        self._atomic_write(
            self._idempotency_path(record.key), json.dumps(raw).encode("utf-8")
        )

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        return await asyncio.to_thread(self._read_idempotency_sync, key)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
        await asyncio.to_thread(self._write_idempotency_sync, record)

    def _raw_put_checked_sync(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset:
        """Precondition + idempotency + put under self._lock (best-effort
        in-process atomicity; see self._lock docstring). Mirrors what
        AssetStore.put does in the 3-step path, but the three steps
        run without interleaving from another in-process caller."""
        with self._lock:
            idem_key = (
                f"put:{options.idempotency_key}" if options.idempotency_key else None
            )
            if idem_key is not None:
                record = self._read_idempotency_sync(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    cached_info = record.result
                    current = self._load_info(path)
                    content_bytes = (
                        self._data_path(path).read_bytes()
                        if current is not None
                        else content
                    )
                    return Asset(info=cached_info, content=content_bytes)
            info = self._load_info(path)
            if options.if_none_match and info is not None:
                raise AssetPreconditionFailedError(
                    f"asset already exists: {path}"
                )
            if options.if_match is not None:
                if info is None or info.etag != options.if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
            if info is not None:
                existing_content = self._data_path(path).read_bytes()
                # content_type must be part of the no-op comparison --
                # a PUT that only changes content_type (same bytes, same
                # metadata) is still a real change and must bump version/etag,
                # not be silently dropped as a no-op.
                if (
                    existing_content == content
                    and dict(info.metadata) == dict(options.metadata)
                    and info.content_type == options.content_type
                ):
                    new_info = info
                else:
                    new_info = self._raw_put_sync(
                        path,
                        content,
                        content_type=options.content_type,
                        metadata=options.metadata,
                    )
            else:
                new_info = self._raw_put_sync(
                    path,
                    content,
                    content_type=options.content_type,
                    metadata=options.metadata,
                )
            if idem_key is not None:
                self._write_idempotency_sync(
                    IdempotencyRecord(
                        key=idem_key, request_hash=request_hash, result=new_info
                    )
                )
            return Asset(info=new_info, content=content)

    async def raw_put_checked(
        self,
        path: AssetPath,
        content: bytes,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> Asset:
        return await asyncio.to_thread(
            self._raw_put_checked_sync,
            path,
            content,
            options=options,
            request_hash=request_hash,
        )

    def _raw_delete_checked_sync(
        self,
        path: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        """Precondition + idempotency + delete under self._lock."""
        with self._lock:
            idem_key = (
                f"delete:{options.idempotency_key}" if options.idempotency_key else None
            )
            if idem_key is not None:
                record = self._read_idempotency_sync(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    return  # idempotent replay: delete returns None
            info = self._load_info(path)
            if options.if_match is not None:
                if info is None or info.etag != options.if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {path}"
                    )
            self._raw_delete_sync(path)

    async def raw_delete_checked(
        self,
        path: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> None:
        return await asyncio.to_thread(
            self._raw_delete_checked_sync,
            path,
            options=options,
            request_hash=request_hash,
        )

    def _raw_move_sync(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> MoveResult:
        """Atomic MOVE on the same filesystem via ``os.replace`` for the data
        file. The whole operation -- idempotency check, precondition, move,
        idempotency-record save -- runs under self._lock so two in-process
        callers cannot interleave the steps; cross-process races remain (a
        documented limitation -- true cross-process atomicity requires the
        SqlAlchemy backend).

        Cross-device source/target would raise OSError(EXDEV) from os.replace;
        we do not catch it. Within one filesystem os.replace is atomic: at no
        point in time does a reader see the target data file half-written.

        The non-data steps (metadata write, source whiteout, target-whiteout
        clear, revision bump) are themselves each atomic via _atomic_write, but
        the SEQUENCE is not crash-atomic -- a crash between the data replace
        and the source whiteout leaves source masked-but-data-gone. That is
        the documented file-mode limitation; file mode requires idempotent
        recoverability rather than strict crash-atomicity.

        Revision bumps exactly once (observable proof the move was not
        decomposed into a put+delete pair, which would bump twice)."""
        with self._lock:
            idem_key = (
                f"move:{options.idempotency_key}" if options.idempotency_key else None
            )
            if idem_key is not None:
                record = self._read_idempotency_sync(idem_key)
                if record is not None:
                    if record.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"idempotency key {options.idempotency_key!r} reused with a different request"
                        )
                    cached_info = record.result
                    if cached_info is not None:
                        content = self._data_path(target)
                        bytes_ = content.read_bytes() if content.exists() else b""
                        return Asset(info=cached_info, content=bytes_)
            source_info = self._load_info(source)
            if source_info is None:
                raise AssetPreconditionFailedError(
                    f"cannot move missing asset: {source}"
                )
            source_data_path = self._data_path(source)
            source_content = source_data_path.read_bytes()

            # Target precondition check (mirror raw_put_checked's logic).
            target_info = self._load_info(target)
            if options.if_none_match and target_info is not None:
                raise AssetPreconditionFailedError(
                    f"asset already exists: {target}"
                )
            if options.if_match is not None:
                if target_info is None or target_info.etag != options.if_match:
                    raise AssetPreconditionFailedError(
                        f"if-match precondition failed: {target}"
                    )

            target_whiteout_path = self._whiteout_path(target)
            prior_target_whiteout_version = 0
            if target_whiteout_path.exists():
                prior_target_whiteout_version = json.loads(
                    target_whiteout_path.read_text()
                )["version"]
            new_target_version = (
                max(
                    target_info.version if target_info else 0,
                    prior_target_whiteout_version,
                )
                + 1
            )
            new_info = AssetInfo(
                path=target,
                kind=source_info.kind,
                etag=source_info.etag,  # same content => same etag
                version=new_target_version,
                content_type=source_info.content_type,
                size=source_info.size,
                modified_at=datetime.now(timezone.utc),
                metadata=dict(source_info.metadata),
            )

            # Move the data file atomically (same-filesystem os.replace).
            # Cross-device would raise OSError(EXDEV); the message documents
            # the limitation rather than silently falling back to copy+unlink.
            target_data_path = self._data_path(target)
            os.replace(source_data_path, target_data_path)

            # Write target metadata, clear any target whiteout.
            self._save_info(new_info)
            if target_whiteout_path.exists():
                target_whiteout_path.unlink()

            # Drop source metadata and write its whiteout.
            self._meta_path(source).unlink(missing_ok=True)
            source_whiteout_path = self._whiteout_path(source)
            existing_sw_version = 0
            if source_whiteout_path.exists():
                existing_sw_version = json.loads(source_whiteout_path.read_text())[
                    "version"
                ]
            sw_version = max(source_info.version, existing_sw_version) + 1
            self._atomic_write(
                source_whiteout_path,
                json.dumps({"version": sw_version}).encode("utf-8"),
            )

            # One revision bump for the whole move.
            self._bump_revision()
            if idem_key is not None:
                self._write_idempotency_sync(
                    IdempotencyRecord(
                        key=idem_key, request_hash=request_hash, result=new_info
                    )
                )
            return Asset(info=new_info, content=source_content)

    async def raw_move_checked(
        self,
        source: AssetPath,
        target: AssetPath,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> MoveResult:
        return await asyncio.to_thread(
            self._raw_move_sync,
            source,
            target,
            options=options,
            request_hash=request_hash,
        )
