#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileSessionStore: root/{session_id}/record.json + root/{session_id}/messages/
{sequence:010d}.json (one file per message, append-only)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ...errors import SessionError
from ...session.models import MessageRole, SessionMessage, SessionRecord, SessionStatus


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _record_to_json(record: SessionRecord) -> dict:
    return {
        "id": record.id, "parent_id": record.parent_id, "status": record.status.value,
        "version": record.version, "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(), "metadata": dict(record.metadata),
    }


def _record_from_json(raw: dict) -> SessionRecord:
    return SessionRecord(
        id=raw["id"], parent_id=raw["parent_id"], status=SessionStatus(raw["status"]), version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]), updated_at=datetime.fromisoformat(raw["updated_at"]),
        metadata=raw["metadata"],
    )


def _message_to_json(message: SessionMessage) -> dict:
    return {
        "id": message.id, "session_id": message.session_id, "sequence": message.sequence,
        "role": message.role.value, "content": message.content, "run_id": message.run_id,
        "created_at": message.created_at.isoformat(), "metadata": dict(message.metadata),
    }


def _message_from_json(raw: dict) -> SessionMessage:
    return SessionMessage(
        id=raw["id"], session_id=raw["session_id"], sequence=raw["sequence"], role=MessageRole(raw["role"]),
        content=raw["content"], run_id=raw["run_id"], created_at=datetime.fromisoformat(raw["created_at"]),
        metadata=raw["metadata"],
    )


class FileSessionStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        d = self._root / _validate_id_segment(session_id, kind="session_id")
        d.mkdir(parents=True, exist_ok=True)
        (d / "messages").mkdir(parents=True, exist_ok=True)
        return d

    def _record_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "record.json"

    async def create(self, session: SessionRecord) -> SessionRecord:
        self._record_path(session.id).write_text(json.dumps(_record_to_json(session)))
        return session

    async def get(self, session_id: str) -> "SessionRecord | None":
        path = self._root / _validate_id_segment(session_id, kind="session_id") / "record.json"
        if not path.exists():
            return None
        return _record_from_json(json.loads(path.read_text()))

    async def append_messages(self, session_id: str, messages: "tuple[SessionMessage, ...]") -> None:
        messages_dir = self._session_dir(session_id) / "messages"
        for message in messages:
            (messages_dir / f"{message.sequence:010d}.json").write_text(json.dumps(_message_to_json(message)))

    async def list_messages(self, session_id: str, *, after_sequence: int = 0, limit: int = 1000) -> "tuple[SessionMessage, ...]":
        messages_dir = self._root / _validate_id_segment(session_id, kind="session_id") / "messages"
        if not messages_dir.exists():
            return ()
        result = []
        for path in sorted(messages_dir.glob("*.json")):
            message = _message_from_json(json.loads(path.read_text()))
            if message.sequence <= after_sequence:
                continue
            result.append(message)
        return tuple(result[:limit])

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord:
        current = await self.get(session_id)
        if current is None:
            raise SessionError(f"session not found: {session_id}")
        updated = SessionRecord(
            id=current.id, parent_id=current.parent_id,
            status=status if status is not None else current.status,
            version=current.version + 1, created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
            metadata=metadata if metadata is not None else current.metadata,
        )
        self._record_path(session_id).write_text(json.dumps(_record_to_json(updated)))
        return updated
