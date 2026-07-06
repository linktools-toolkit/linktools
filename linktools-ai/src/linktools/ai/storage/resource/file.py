#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileResourceBackend: filesystem-backed ResourceBackend. Flat filename mapping
(path segments joined with "__") avoids managing nested directories; atomic writes
via temp-file-then-os.replace; whiteouts/idempotency/revision are separate small
JSON files under .resource/ so a crash mid-write cannot corrupt unrelated resources."""

import json
import os
import tempfile
import urllib.parse
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from .models import Depth, Found, IdempotencyRecord, Masked, Missing, Resource, ResourceInfo, ResourceKind, ResourcePage
from .path import ResourcePath
from ...errors import InvalidResourcePathError


class SymlinkPolicy(str, Enum):
    DENY = "deny"
    ALLOW_INTERNAL = "allow_internal"


def _filename(path: ResourcePath) -> str:
    # Percent-encode so the mapping from ResourcePath -> filename is reversible:
    # "/" and "%" are escaped, so distinct paths can never collide on one filename
    # (unlike the previous "__"-joining scheme, where "/a/b" and "/a__b" collided).
    return urllib.parse.quote(path.value.strip("/"), safe="")


def _path_from_filename(stem: str) -> ResourcePath:
    return ResourcePath("/" + urllib.parse.unquote(stem))


class FileResourceBackend:
    def __init__(self, *, root: Path, readonly: bool = False, symlink_policy: SymlinkPolicy = SymlinkPolicy.DENY) -> None:
        self.readonly = readonly
        self._symlink_policy = symlink_policy
        self._root = Path(root)
        self._data_dir = self._root / "data"
        self._meta_dir = self._root / ".resource" / "metadata"
        self._whiteout_dir = self._root / ".resource" / "whiteouts"
        self._idempotency_dir = self._root / ".resource" / "idempotency"
        self._revision_file = self._root / ".resource" / "revision"
        for d in (self._data_dir, self._meta_dir, self._whiteout_dir, self._idempotency_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _resolve(self, directory: Path, filename: str) -> Path:
        resolved = (directory / filename).resolve()
        if self._symlink_policy == SymlinkPolicy.DENY and resolved.is_symlink():
            raise InvalidResourcePathError(f"symlink not allowed: {resolved}")
        if directory.resolve() not in resolved.parents and resolved != directory.resolve():
            raise InvalidResourcePathError(f"path escapes backend root: {resolved}")
        return resolved

    def _atomic_write(self, path: Path, content: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_name, path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    def _read_revision(self) -> int:
        if not self._revision_file.exists():
            return 0
        return int(self._revision_file.read_text().strip() or "0")

    def _bump_revision(self) -> int:
        value = self._read_revision() + 1
        self._atomic_write(self._revision_file, str(value).encode("utf-8"))
        return value

    def _meta_path(self, path: ResourcePath) -> Path:
        return self._resolve(self._meta_dir, _filename(path) + ".json")

    def _data_path(self, path: ResourcePath) -> Path:
        return self._resolve(self._data_dir, _filename(path))

    def _whiteout_path(self, path: ResourcePath) -> Path:
        return self._resolve(self._whiteout_dir, _filename(path) + ".json")

    def _idempotency_path(self, key: str) -> Path:
        return self._resolve(self._idempotency_dir, key.replace("/", "__") + ".json")

    def _load_info(self, path: ResourcePath) -> "ResourceInfo | None":
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return None
        raw = json.loads(meta_path.read_text())
        return ResourceInfo(
            path=path,
            kind=ResourceKind(raw["kind"]),
            etag=raw["etag"],
            version=raw["version"],
            content_type=raw["content_type"],
            size=raw["size"],
            modified_at=datetime.fromisoformat(raw["modified_at"]),
            metadata=raw["metadata"],
        )

    def _save_info(self, info: ResourceInfo) -> None:
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

    async def raw_get(self, path: ResourcePath, *, include_content: bool = True):
        info = self._load_info(path)
        if info is not None:
            content = b""
            if include_content:
                content = self._data_path(path).read_bytes()
            return Found(resource=Resource(info=info, content=content))
        whiteout_path = self._whiteout_path(path)
        if whiteout_path.exists():
            version = json.loads(whiteout_path.read_text())["version"]
            return Masked(path=path, version=version)
        return Missing()

    async def raw_propfind(self, path: ResourcePath, *, depth: Depth, limit: int, cursor: "str | None") -> ResourcePage:
        # NOTE: cursor-based continuation is not yet implemented in Phase 1 -- `cursor`
        # is accepted for forward API compatibility but ignored; results are simply
        # truncated to `limit`. Real pagination is deferred to a later phase.
        prefix = path.value.rstrip("/") + "/"
        items = []
        for meta_file in sorted(self._meta_dir.glob("*.json")):
            candidate = _path_from_filename(meta_file.stem)
            if not candidate.value.startswith(prefix):
                continue
            rest = candidate.value[len(prefix):]
            if depth == Depth.ONE and "/" in rest:
                continue
            items.append(self._load_info(candidate))
        return ResourcePage(items=tuple(items[:limit]), cursor=None)

    async def raw_put(self, path: ResourcePath, content: bytes, *, content_type: "str | None", metadata: "Mapping[str, object]"):
        prior = self._load_info(path)
        whiteout_path = self._whiteout_path(path)
        prior_whiteout_version = 0
        if whiteout_path.exists():
            prior_whiteout_version = json.loads(whiteout_path.read_text())["version"]
        version = max(prior.version if prior else 0, prior_whiteout_version) + 1
        info = ResourceInfo(
            path=path,
            kind=ResourceKind.FILE,
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

    async def raw_delete(self, path: ResourcePath) -> "ResourceInfo | None":
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
        self._atomic_write(whiteout_path, json.dumps({"version": new_version}).encode("utf-8"))
        self._bump_revision()
        return info

    async def revision(self) -> int:
        return self._read_revision()

    async def get_idempotency(self, key: str) -> "IdempotencyRecord | None":
        path = self._idempotency_path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        result = None
        if raw["result"] is not None:
            r = raw["result"]
            result = ResourceInfo(
                path=ResourcePath(r["path"]),
                kind=ResourceKind(r["kind"]),
                etag=r["etag"],
                version=r["version"],
                content_type=r["content_type"],
                size=r["size"],
                modified_at=datetime.fromisoformat(r["modified_at"]),
                metadata=r["metadata"],
            )
        return IdempotencyRecord(key=raw["key"], request_hash=raw["request_hash"], result=result)

    async def put_idempotency(self, record: IdempotencyRecord) -> None:
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
        self._atomic_write(self._idempotency_path(record.key), json.dumps(raw).encode("utf-8"))
