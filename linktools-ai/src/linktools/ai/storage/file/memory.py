#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileMemoryStore: single-process file backend for MemoryStore (the Protocol in
memory/store.py). One JSON file per record at root/{memory_id}.json.
Mirrors FileSwarmStore/FileRunStore's atomic-write + path-traversal-guard
patterns (see storage/file/run.py). The `_UNSET` sentinel distinguishes
"omit this field" from `category=None` meaning "explicitly clear".

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The ``asyncio.Lock`` is held in the async wrapper and spans the
``to_thread`` call (not the other way around)."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ...errors import MemoryConflictError, MemoryNotFoundError
from ...memory.models import MemoryRecord
from ...memory.store import _UNSET
from .run import _atomic_write, _validate_id_segment


def _record_to_json(record: MemoryRecord) -> dict:
    return {
        "id": record.id,
        "owner_id": record.owner_id,
        "content": record.content,
        "category": record.category,
        "confidence": record.confidence,
        "version": record.version,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "metadata": dict(record.metadata),
    }


def _record_from_json(raw: dict) -> MemoryRecord:
    return MemoryRecord(
        id=raw["id"],
        owner_id=raw["owner_id"],
        content=raw["content"],
        category=raw["category"],
        confidence=raw["confidence"],
        version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        updated_at=datetime.fromisoformat(raw["updated_at"]),
        metadata=raw["metadata"],
    )


class FileMemoryStore:
    """Single-process MemoryStore backed by per-record JSON files.

    Records live at ``root/{memory_id}.json``. Writes are atomic
    (temp-file + ``os.replace``) and ids are validated to prevent path
    traversal. An ``asyncio.Lock`` serializes ``update``/``forget`` so that
    optimistic-concurrency invariants hold within one process.
    """

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- paths ---------------------------------------------------------

    def _path(self, memory_id: str) -> Path:
        return self._root / f"{_validate_id_segment(memory_id, kind='memory_id')}.json"

    # -- read ----------------------------------------------------------

    def _get_sync(self, memory_id: str) -> "MemoryRecord | None":
        path = self._path(memory_id)
        if not path.exists():
            return None
        return _record_from_json(json.loads(path.read_text()))

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        return await asyncio.to_thread(self._get_sync, memory_id)

    def _search_sync(
        self,
        query: str,
        *,
        owner_id: "str | None",
        category: "str | None",
        limit: int,
    ) -> "tuple[MemoryRecord, ...]":
        needle = query.lower()
        out: list = []
        for path in self._root.glob("*.json"):
            raw = json.loads(path.read_text())
            if owner_id is not None and raw["owner_id"] != owner_id:
                continue
            if category is not None and raw["category"] != category:
                continue
            if needle not in str(raw["content"]).lower():
                continue
            out.append(_record_from_json(raw))
        out.sort(key=lambda r: r.created_at)
        return tuple(out[:limit])

    async def search(
        self,
        query: str,
        *,
        owner_id: "str | None" = None,
        category: "str | None" = None,
        limit: int = 10,
    ) -> "tuple[MemoryRecord, ...]":
        return await asyncio.to_thread(
            self._search_sync,
            query,
            owner_id=owner_id,
            category=category,
            limit=limit,
        )

    # -- write ---------------------------------------------------------

    def _remember_sync(self, record: MemoryRecord) -> MemoryRecord:
        # _path validates the id segment, guarding against path traversal.
        path = self._path(record.id)
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
        current = self._get_sync(memory_id)
        if current is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        if current.version != expected_version:
            raise MemoryConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        # Apply ONLY fields explicitly passed (i.e. `is not _UNSET`); a
        # None value means "clear this field" (e.g. category=None).
        new_content = current.content if content is _UNSET else content
        new_category = current.category if category is _UNSET else category
        new_confidence = current.confidence if confidence is _UNSET else confidence
        new_metadata = current.metadata if metadata is _UNSET else metadata
        updated = MemoryRecord(
            id=current.id,
            owner_id=current.owner_id,
            content=new_content,
            category=new_category,
            confidence=new_confidence,
            version=current.version + 1,
            created_at=current.created_at,
            updated_at=datetime.now(current.created_at.tzinfo or timezone.utc),
            metadata=new_metadata,
        )
        _atomic_write(
            self._path(memory_id), json.dumps(_record_to_json(updated)).encode("utf-8")
        )
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
            return await asyncio.to_thread(
                self._update_sync,
                memory_id,
                expected_version=expected_version,
                content=content,
                category=category,
                confidence=confidence,
                metadata=metadata,
            )

    def _forget_sync(self, memory_id: str, *, expected_version: int) -> None:
        current = self._get_sync(memory_id)
        if current is None:
            raise MemoryNotFoundError(f"memory not found: {memory_id}")
        if current.version != expected_version:
            raise MemoryConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        self._path(memory_id).unlink()

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._forget_sync, memory_id, expected_version=expected_version
            )
