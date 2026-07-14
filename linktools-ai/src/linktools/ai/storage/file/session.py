#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileSessionStore: root/{session_id}/record.json + root/{session_id}/messages/
{sequence:010d}.json (one file per message, append-only).

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ...errors import SessionError
from ...session.models import (
    MessageRole,
    NewSessionMessage,
    SessionMessage,
    SessionRecord,
    SessionStatus,
)


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _record_to_json(record: SessionRecord) -> dict:
    return {
        "id": record.id,
        "parent_id": record.parent_id,
        "user_id": record.user_id,
        "tenant_id": record.tenant_id,
        "status": record.status.value,
        "version": record.version,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "metadata": dict(record.metadata),
    }


def _record_from_json(raw: dict) -> SessionRecord:
    return SessionRecord(
        id=raw["id"],
        parent_id=raw["parent_id"],
        # Older on-disk records predate ownership columns -> unowned (None).
        user_id=raw.get("user_id"),
        tenant_id=raw.get("tenant_id"),
        status=SessionStatus(raw["status"]),
        version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        updated_at=datetime.fromisoformat(raw["updated_at"]),
        metadata=raw["metadata"],
    )


def _message_to_json(message: SessionMessage) -> dict:
    return {
        "id": message.id,
        "session_id": message.session_id,
        "sequence": message.sequence,
        "role": message.role.value,
        "content": message.content,
        "run_id": message.run_id,
        "created_at": message.created_at.isoformat(),
        "metadata": dict(message.metadata),
    }


def _message_from_json(raw: dict) -> SessionMessage:
    return SessionMessage(
        id=raw["id"],
        session_id=raw["session_id"],
        sequence=raw["sequence"],
        role=MessageRole(raw["role"]),
        content=raw["content"],
        run_id=raw["run_id"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        metadata=raw["metadata"],
    )


class FileSessionStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # this store is the SOLE sequence authority --
        # append_messages() reads the current max sequence and assigns fresh
        # ones itself (mirroring FileEventStore), so the caller no longer
        # computes `len(prior_messages) + 1`. The per-session lock still
        # serializes the read-max-then-write so two concurrent coroutines
        # appending to the same session cannot compute the same sequence.
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._locks_guard = asyncio.Lock()

    async def _session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    def _session_dir(self, session_id: str) -> Path:
        d = self._root / _validate_id_segment(session_id, kind="session_id")
        d.mkdir(parents=True, exist_ok=True)
        (d / "messages").mkdir(parents=True, exist_ok=True)
        return d

    def _record_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "record.json"

    def _create_sync(self, session: SessionRecord) -> SessionRecord:
        self._record_path(session.id).write_text(json.dumps(_record_to_json(session)))
        return session

    async def create(self, session: SessionRecord) -> SessionRecord:
        return await asyncio.to_thread(self._create_sync, session)

    def _get_sync(self, session_id: str) -> "SessionRecord | None":
        path = (
            self._root
            / _validate_id_segment(session_id, kind="session_id")
            / "record.json"
        )
        if not path.exists():
            return None
        return _record_from_json(json.loads(path.read_text()))

    async def get(self, session_id: str) -> "SessionRecord | None":
        return await asyncio.to_thread(self._get_sync, session_id)

    def _next_sequence_sync(self, session_id: str) -> int:
        messages_dir = self._session_dir(session_id) / "messages"
        existing = list(messages_dir.glob("*.json"))
        return max((int(p.stem) for p in existing), default=0) + 1

    def _append_messages_sync(
        self,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        messages_dir = self._session_dir(session_id) / "messages"
        next_seq = self._next_sequence_sync(session_id)
        persisted = []
        for offset, message in enumerate(messages):
            sequence = next_seq + offset
            full = SessionMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                sequence=sequence,
                role=message.role,
                content=message.content,
                run_id=message.run_id,
                created_at=datetime.now(timezone.utc),
                metadata=message.metadata,
            )
            (messages_dir / f"{sequence:010d}.json").write_text(
                json.dumps(_message_to_json(full))
            )
            persisted.append(full)
        return tuple(persisted)

    async def append_messages(
        self,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        lock = await self._session_lock(session_id)
        async with lock:
            return await asyncio.to_thread(
                self._append_messages_sync, session_id, messages
            )

    def _list_messages_sync(
        self, session_id: str, *, after_sequence: int, limit: int
    ) -> "tuple[SessionMessage, ...]":
        messages_dir = (
            self._root
            / _validate_id_segment(session_id, kind="session_id")
            / "messages"
        )
        if not messages_dir.exists():
            return ()
        result = []
        for path in sorted(messages_dir.glob("*.json")):
            message = _message_from_json(json.loads(path.read_text()))
            if message.sequence <= after_sequence:
                continue
            result.append(message)
        return tuple(result[:limit])

    async def list_messages(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> "tuple[SessionMessage, ...]":
        return await asyncio.to_thread(
            self._list_messages_sync,
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    def _update_sync(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None",
        metadata: "Mapping[str, Any] | None",
    ) -> SessionRecord:
        current = self._get_sync(session_id)
        if current is None:
            raise SessionError(f"session not found: {session_id}")
        updated = SessionRecord(
            id=current.id,
            parent_id=current.parent_id,
            user_id=current.user_id,
            tenant_id=current.tenant_id,
            status=status if status is not None else current.status,
            version=current.version + 1,
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=metadata if metadata is not None else current.metadata,
        )
        self._record_path(session_id).write_text(json.dumps(_record_to_json(updated)))
        return updated

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord:
        lock = await self._session_lock(session_id)
        async with lock:
            return await asyncio.to_thread(
                self._update_sync, session_id, status=status, metadata=metadata
            )
