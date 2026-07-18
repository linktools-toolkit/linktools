#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileSessionStore: root/{session_id}/record.json + root/{session_id}/messages/
{sequence:010d}.json (one file per message, append-only).

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop."""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ...errors import SessionCorruptionError, SessionError
from ...session.models import (
    MessageRole,
    NewSessionMessage,
    SessionMessage,
    SessionRecord,
    SessionStatus,
)
from ._util import _atomic_write


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _load_json(path: Path) -> "Any":
    """Read + parse ``path`` as JSON. A present-but-unparseable file is
    corruption (not a missing session): raise SessionCorruptionError naming the
    path so a repair tool can target it, rather than silently masking it.

    Scope: this catches JSON *parse* failures (the "file unreadable as JSON"
    case). A file that parses but carries a bad schema (unknown enum, missing
    field, bad datetime) surfaces as the raw ValueError/KeyError from the
    record constructor -- not masked, and out of scope for this
    reader-tolerance layer."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        raise SessionCorruptionError(
            f"corrupt session file (not valid JSON, left in place for repair): {path}"
        ) from exc


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
        self.recover_incomplete_batches()

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
        _atomic_write(
            self._record_path(session.id),
            json.dumps(_record_to_json(session)).encode("utf-8"),
        )
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
        return _record_from_json(_load_json(path))

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
        # A SessionRecord may have been materialized after an earlier staged
        # batch. Recover and validate that batch before advancing the marker.
        self.recover_incomplete_batches()
        # Message files are staged first; the session record's committed
        # sequence is the batch commit marker. Readers ignore files above it,
        # so a crash cannot expose a partial batch. Orphaned files are retained
        # for recovery/audit and are never treated as committed.
        messages_dir = self._session_dir(session_id) / "messages"
        next_seq = self._next_sequence_sync(session_id)
        batch_path = self._session_dir(session_id) / f"batch-{next_seq:010d}.journal"
        staged = []
        for offset, message in enumerate(messages):
            sequence = next_seq + offset
            full = SessionMessage(id=str(uuid.uuid4()), session_id=session_id,
                sequence=sequence, role=message.role, content=message.content,
                run_id=message.run_id, created_at=datetime.now(timezone.utc),
                metadata=message.metadata)
            payload = _message_to_json(full)
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            staged.append((full, payload, hashlib.sha256(encoded).hexdigest()))
        _atomic_write(
            batch_path,
            json.dumps({"schema_version": 1, "session_id": session_id,
                        "messages": [{"sequence": m.sequence, "message_id": m.id,
                                      "sha256": digest, "payload": payload}
                                     for m, payload, digest in staged]},
                       sort_keys=True).encode("utf-8"),
        )
        persisted = []
        for full, payload, _digest in staged:
            sequence = full.sequence
            _atomic_write(
                messages_dir / f"{sequence:010d}.json",
                json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
            persisted.append(full)
        current = self._get_sync(session_id)
        if current is None:
            # Legacy callers may append to an externally-created session id
            # before materializing SessionRecord metadata. Keep those writes
            # visible; subsequent appends can establish the commit marker.
            return tuple(persisted)
        committed = next_seq + len(persisted) - 1 if persisted else int(
            current.metadata.get("_committed_sequence", next_seq - 1)
        )
        metadata = dict(current.metadata)
        metadata["_committed_sequence"] = committed
        updated = SessionRecord(
            id=current.id, parent_id=current.parent_id, status=current.status,
            version=current.version + 1, created_at=current.created_at,
            updated_at=datetime.now(timezone.utc), user_id=current.user_id,
            tenant_id=current.tenant_id, metadata=metadata,
        )
        _atomic_write(self._record_path(session_id), json.dumps(_record_to_json(updated)).encode("utf-8"))
        batch_path.unlink(missing_ok=True)
        return tuple(persisted)

    def recover_incomplete_batches(self) -> None:
        """Finalize batch markers whose message files were fully staged."""
        for session_dir in self._root.iterdir():
            if not session_dir.is_dir():
                continue
            for marker in session_dir.glob("batch-*.journal"):
                try:
                    raw = _load_json(marker)
                    current = self._get_sync(raw["session_id"])
                    if current is not None:
                        entries = raw["messages"]
                        for entry in entries:
                            seq = int(entry["sequence"])
                            path = session_dir / "messages" / f"{seq:010d}.json"
                            if not path.exists():
                                _atomic_write(path, json.dumps(entry["payload"], sort_keys=True).encode("utf-8"))
                            payload = path.read_bytes()
                            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                                raise SessionCorruptionError("session batch message hash mismatch")
                            decoded = json.loads(payload)
                            if decoded.get("id") != entry["message_id"]:
                                raise SessionCorruptionError("session batch message id mismatch")
                            if decoded.get("session_id") not in (None, raw["session_id"]):
                                raise SessionCorruptionError("session batch session_id mismatch")
                            if int(decoded.get("sequence", -1)) != seq:
                                raise SessionCorruptionError("session batch sequence mismatch")
                        end = max((int(entry["sequence"]) for entry in entries), default=0)
                        if end >= int(current.metadata.get("_committed_sequence", 0)):
                            metadata = dict(current.metadata)
                            metadata["_committed_sequence"] = end
                            self._update_sync(raw["session_id"], status=None, metadata=metadata)
                        marker.unlink(missing_ok=True)
                except Exception:
                    # Leave corrupt markers for operator repair; readers remain
                    # isolated by the last committed sequence.
                    continue
            record_path = session_dir / "record.json"
            if record_path.exists():
                current = self._get_sync(session_dir.name)
                if current is not None and "_committed_sequence" not in current.metadata:
                    metadata = dict(current.metadata)
                    committed = 0
                    for seq in range(1, 10**9):
                        p = session_dir / "messages" / f"{seq:010d}.json"
                        if not p.exists():
                            break
                        payload = _load_json(p)
                        if payload.get("session_id") not in (None, current.id):
                            raise SessionCorruptionError("session id mismatch")
                        if int(payload.get("sequence", -1)) != seq:
                            raise SessionCorruptionError("session sequence mismatch")
                        committed = seq
                    metadata["_committed_sequence"] = committed
                    metadata["_legacy_missing_committed_sequence"] = True
                    self._update_sync(current.id, status=None, metadata=metadata)

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
        session = self._get_sync(session_id)
        # Legacy records without an explicit commit marker are fail-closed;
        # recovery must validate a batch before making messages visible.
        committed_sequence = int(session.metadata.get("_committed_sequence", 0)) if session else 0
        for path in sorted(messages_dir.glob("*.json")):
            message = _message_from_json(_load_json(path))
            if message.sequence > committed_sequence:
                continue
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
        _atomic_write(
            self._record_path(session_id),
            json.dumps(_record_to_json(updated)).encode("utf-8"),
        )
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
