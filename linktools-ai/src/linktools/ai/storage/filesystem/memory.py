#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemMemoryStore: single-process file backend for MemoryStore (the Protocol in
memory/store.py). One JSON file per record, partitioned by tenant:

    root/{tenant_id}/{memory_id}.json   (tenant-scoped records)
    root/{memory_id}.json               (legacy, pre-tenant records)

``search`` is the tenant isolation boundary: it scans ONLY the requesting
scope's tenant subdir, so one tenant can never enumerate another's records.
Legacy flat records are read with a synthesized ``LEGACY_TENANT_ID`` tenant and
are visible ONLY to an explicit legacy scope -- a real tenant's search never
touches the flat layout, so old data is never silently exposed (the migration
quarantine). ``get``/``update``/``forget`` look up by memory_id across both
layouts (the id is the capability); they are not the isolation boundary.

Mirrors FilesystemSwarmStore/FilesystemRunStore's atomic-write + path-traversal-guard
patterns (see storage/filesystem/run.py). The tenant and memory id segments are both
validated via ``_validate_id_segment`` so a caller-controlled value can never
escape the store root. The ``asyncio.Lock`` serializes ``update``/``forget`` so
that optimistic-concurrency invariants hold within one process.

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ...errors import MemoryConflictError, MemoryNotFoundError
from ...memory.models import MemoryMatch, MemoryRecord
from ...memory.scope import LEGACY_TENANT_ID, MemoryScope, is_legacy_tenant
from ...memory.store import _UNSET
from .run import _atomic_write, _validate_id_segment


def _record_to_json(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "tenant_id": record.tenant_id,
        "owner_id": record.owner_id,
        "content": record.content,
        "category": record.category,
        "confidence": record.confidence,
        "version": record.version,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "metadata": dict(record.metadata),
        "user_id": record.user_id,
        "workspace_id": record.workspace_id,
        "session_id": record.session_id,
    }


def _record_from_json(raw: dict) -> MemoryRecord:
    # Reader tolerates the pre-tenant format: an older record has no tenant_id
    # field at all. Such a record is migrated in-memory to the legacy tenant --
    # it is NOT shown to any real tenant's scope (search skips the flat layout
    # unless the scope is itself the legacy tenant).
    return MemoryRecord(
        id=raw["id"],
        tenant_id=raw.get("tenant_id") or LEGACY_TENANT_ID,
        owner_id=raw["owner_id"],
        content=raw["content"],
        category=raw["category"],
        confidence=raw["confidence"],
        version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        updated_at=datetime.fromisoformat(raw["updated_at"]),
        metadata=raw["metadata"],
        user_id=raw.get("user_id"),
        workspace_id=raw.get("workspace_id"),
        session_id=raw.get("session_id"),
    )


def _subscope_matches(record: MemoryRecord, scope: MemoryScope) -> bool:
    # A NULL sub-scope field on the record means "shared at the tenant level":
    # visible to any user / workspace / session of that tenant. A non-NULL
    # record field is a hard match against the corresponding scope value, and
    # a None scope value means "do not narrow on this axis".
    if scope.user_id is not None and record.user_id is not None and record.user_id != scope.user_id:
        return False
    if (
        scope.workspace_id is not None
        and record.workspace_id is not None
        and record.workspace_id != scope.workspace_id
    ):
        return False
    if (
        scope.session_id is not None
        and record.session_id is not None
        and record.session_id != scope.session_id
    ):
        return False
    return True


class FilesystemMemoryStore:
    """Single-process MemoryStore backed by per-record JSON files, partitioned
    by tenant (``root/{tenant_id}/{memory_id}.json``). Writes are atomic
    (temp-file + ``os.replace``) and both the tenant and memory id segments are
    validated to prevent path traversal. An ``asyncio.Lock`` serializes
    ``update``/``forget`` so that optimistic-concurrency invariants hold within
    one process."""

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- paths ---------------------------------------------------------

    def _tenant_subdir(self, tenant_id: str) -> Path:
        # Validate the tenant segment so a caller-controlled tenant_id can never
        # escape the store root via "../".
        segment = _validate_id_segment(tenant_id, kind="tenant_id")
        path = self._root / segment
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _partitioned_path(self, tenant_id: str, memory_id: str) -> Path:
        memory_id = _validate_id_segment(memory_id, kind="memory_id")
        return self._tenant_subdir(tenant_id) / f"{memory_id}.json"

    def _legacy_path(self, memory_id: str) -> Path:
        # Pre-tenant flat layout: root/{memory_id}.json (no tenant subdir).
        memory_id = _validate_id_segment(memory_id, kind="memory_id")
        return self._root / f"{memory_id}.json"

    def _locate_sync(self, memory_id: str) -> "Path | None":
        # get/update/forget key off memory_id alone (the id is the capability),
        # so the record may live in any tenant subdir or the flat legacy layout.
        # Search the flat legacy path first, then the one-level tenant subdirs.
        legacy = self._legacy_path(memory_id)
        if legacy.exists():
            return legacy
        safe_id = _validate_id_segment(memory_id, kind="memory_id")
        for path in self._root.glob(f"*/{safe_id}.json"):
            return path
        return None

    # -- read ----------------------------------------------------------

    def _get_sync(self, memory_id: str) -> "MemoryRecord | None":
        path = self._locate_sync(memory_id)
        if path is None:
            return None
        return _record_from_json(json.loads(path.read_text()))

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        return await asyncio.to_thread(self._get_sync, memory_id)

    def _search_sync(
        self,
        query: str,
        *,
        scope: MemoryScope,
        category: "str | None",
        limit: int,
    ) -> "tuple[MemoryMatch, ...]":
        # The tenant isolation boundary: scan ONLY this scope's tenant subdir.
        # A real tenant never sees the flat legacy layout; only an explicit
        # legacy scope does (migration quarantine). Keyword search carries no
        # ranking signal, so every hit is returned with score=None.
        needle = query.lower()
        paths: list = list(self._tenant_subdir(scope.tenant_id).glob("*.json"))
        if is_legacy_tenant(scope.tenant_id):
            paths += list(self._root.glob("*.json"))
        out: list = []
        for path in paths:
            record = _record_from_json(json.loads(path.read_text()))
            if not _subscope_matches(record, scope):
                continue
            if category is not None and record.category != category:
                continue
            if needle and needle not in str(record.content).lower():
                continue
            out.append(record)
        out.sort(key=lambda r: r.created_at)
        return tuple(MemoryMatch(record=r, score=None) for r in out[:limit])

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryMatch, ...]":
        return await asyncio.to_thread(
            self._search_sync,
            query,
            scope=scope,
            category=category,
            limit=limit,
        )

    # -- write ---------------------------------------------------------

    def _remember_sync(self, record: MemoryRecord) -> MemoryRecord:
        # Partitioned by tenant; both segments validated against traversal.
        path = self._partitioned_path(record.tenant_id, record.id)
        if path.exists():
            raise MemoryConflictError(f"memory already exists: {record.id}")
        _atomic_write(path, json.dumps(_record_to_json(record)).encode("utf-8"))
        return record

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        return await asyncio.to_thread(self._remember_sync, record)

    def _update_sync(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object,
        category: object,
        confidence: object,
        metadata: object,
    ) -> MemoryRecord:
        path = self._locate_sync(memory_id)
        if path is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        current = _record_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise MemoryConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        # Apply ONLY fields explicitly passed (i.e. `is not _UNSET`); a
        # None value means "clear this field" (e.g. category=None). The
        # tenant / sub-scope identity fields are immutable here.
        new_content = current.content if content is _UNSET else content
        new_category = current.category if category is _UNSET else category
        new_confidence = current.confidence if confidence is _UNSET else confidence
        new_metadata = current.metadata if metadata is _UNSET else metadata
        updated = MemoryRecord(
            id=current.id,
            tenant_id=current.tenant_id,
            owner_id=current.owner_id,
            content=new_content,
            category=new_category,
            confidence=new_confidence,
            version=current.version + 1,
            created_at=current.created_at,
            updated_at=datetime.now(current.created_at.tzinfo or timezone.utc),
            metadata=new_metadata,
            user_id=current.user_id,
            workspace_id=current.workspace_id,
            session_id=current.session_id,
        )
        _atomic_write(path, json.dumps(_record_to_json(updated)).encode("utf-8"))
        return updated

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = _UNSET,
        category: object = _UNSET,
        confidence: object = _UNSET,
        metadata: object = _UNSET,
    ) -> MemoryRecord:
        async with self._lock:
            updated = await asyncio.to_thread(
                self._update_sync,
                memory_id,
                expected_version=expected_version,
                content=content,
                category=category,
                confidence=confidence,
                metadata=metadata,
            )
        return updated

    def _forget_sync(self, memory_id: str, *, expected_version: int) -> None:
        path = self._locate_sync(memory_id)
        if path is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        current = _record_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise MemoryConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        path.unlink()

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._forget_sync, memory_id, expected_version=expected_version
            )
