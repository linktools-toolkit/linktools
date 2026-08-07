#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crash-durable FILE implementation of the Harness StepStore contract."""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.files import read_json, write_json_atomic
from ..storage.lock import FileWriterLock


class DurableFileStepStore:
    """Persist Harness records below a namespace-digested directory."""

    def __init__(self, root: str | Path, namespace: str, *, writer_lock: FileWriterLock | None = None) -> None:
        if not namespace.strip():
            raise ValueError("StepStore namespace is required")
        self._root = Path(root).expanduser().resolve() / "steps" / _digest(namespace)
        self._lock = asyncio.Lock()
        self._writer_lock = writer_lock or FileWriterLock(self._root.parent.parent / ".linktools" / "runtime" / _digest(namespace) / "step.lock")
        self._owns_writer_lock = writer_lock is None
        self._closed = False

    async def initialize(self) -> None:
        async with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            if self._owns_writer_lock:
                await self._writer_lock.acquire()
            self._closed = False

    async def close(self) -> None:
        if self._owns_writer_lock:
            await self._writer_lock.release()
        self._closed = True

    async def register_run(self, record: RunRecord) -> None:
        async with self._lock:
            self._ensure_open()
            path = self._run_path(record.run_id)
            if path.exists():
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT)
            directory = path.parent
            directory.mkdir(parents=True, exist_ok=False)
            write_json_atomic(path, _run_json(record), fsync=True)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        async with self._lock:
            self._ensure_open()
            path = self._run_path(run_id)
            if not path.exists():
                return None
            return _run_from_json(read_json(path))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        async with self._lock:
            self._ensure_open()
            records: list[RunRecord] = []
            if self._root.exists():
                for path in self._root.glob("runs/*/run.json"):
                    record = _run_from_json(read_json(path))
                    if parent_run_id is not None and record.parent_run_id != parent_run_id:
                        continue
                    if conversation_id is not None and record.conversation_id != conversation_id:
                        continue
                    records.append(record)
            return sorted(records, key=lambda item: (item.started_at, item.run_id))

    async def append_event(self, event: StepEvent) -> None:
        async with self._lock:
            self._ensure_open()
            path = self._run_path(event.run_id).with_name("events.jsonl")
            _append_json(path, _event_json(event))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        async with self._lock:
            self._ensure_open()
            return [_event_from_json(item) for item in _read_jsonl(self._run_path(run_id).with_name("events.jsonl"))]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        async with self._lock:
            self._ensure_open()
            run_directory = self._run_path(snapshot.run_id).parent
            run_directory.mkdir(parents=True, exist_ok=True)
            index = len(tuple(run_directory.glob("snapshot-*.json")))
            payload = {
                "run_id": snapshot.run_id,
                "step_index": snapshot.step_index,
                "conversation_id": snapshot.conversation_id,
                "parent_run_id": snapshot.parent_run_id,
                "agent_name": snapshot.agent_name,
                "timestamp": _time_json(snapshot.timestamp),
                "messages": json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages)),
            }
            write_json_atomic(run_directory / f"snapshot-{index:020d}.json", payload, fsync=True)

    async def latest_snapshot(self, *, run_id: str) -> ContinuableSnapshot | None:
        async with self._lock:
            self._ensure_open()
            paths = sorted(self._run_path(run_id).parent.glob("snapshot-*.json"))
            return None if not paths else _snapshot_from_json(read_json(paths[-1]))

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        async with self._lock:
            self._ensure_open()
            _append_json(self._run_path(record.run_id).with_name("tool-effects.jsonl"), _effect_json(record))

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        async with self._lock:
            self._ensure_open()
            latest = _latest_effect(_read_jsonl(self._run_path(run_id).with_name("tool-effects.jsonl")), tool_call_id)
            return None if latest is None else _effect_from_json(latest)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        async with self._lock:
            self._ensure_open()
            latest: dict[str, dict[str, object]] = {}
            for item in _read_jsonl(self._run_path(run_id).with_name("tool-effects.jsonl")):
                tool_call_id = item.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    latest[tool_call_id] = item
            return [_effect_from_json(item) for item in latest.values() if item.get("status") == "started"]

    def _run_path(self, run_id: str) -> Path:
        _validate_id(run_id)
        return self._root / "runs" / _digest(run_id) / "run.json"

    def _ensure_open(self) -> None:
        if self._closed:
            raise LinktoolsAIError(ErrorCode.STORAGE_CLOSED)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_id(value: str) -> None:
    if not value or len(value) > 200 or value in {".", ".."} or any(char in value for char in "/\\\x00"):
        raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)


def _time_json(value: datetime) -> str:
    if value.tzinfo is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.astimezone(timezone.utc).isoformat()


def _append_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _run_json(record: RunRecord) -> dict[str, object]:
    return {"run_id": record.run_id, "conversation_id": record.conversation_id, "parent_run_id": record.parent_run_id, "agent_name": record.agent_name, "metadata": dict(record.metadata), "started_at": _time_json(record.started_at)}


def _run_from_json(value: dict[str, object]) -> RunRecord:
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return RunRecord(run_id=str(value["run_id"]), conversation_id=_optional_string(value.get("conversation_id")), parent_run_id=_optional_string(value.get("parent_run_id")), agent_name=_optional_string(value.get("agent_name")), metadata=metadata, started_at=_datetime(value["started_at"]))


def _event_json(event: StepEvent) -> dict[str, object]:
    return {**asdict(event), "timestamp": _time_json(event.timestamp), "metadata": dict(event.metadata)}


def _event_from_json(value: dict[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["run_id"]), kind=value["kind"], step_index=int(value["step_index"]), timestamp=_datetime(value["timestamp"]), conversation_id=_optional_string(value.get("conversation_id")), parent_run_id=_optional_string(value.get("parent_run_id")), agent_name=_optional_string(value.get("agent_name")), tool_call_id=_optional_string(value.get("tool_call_id")), tool_name=_optional_string(value.get("tool_name")), error=_optional_string(value.get("error")), metadata=_string_map(value.get("metadata", {})))


def _effect_json(record: ToolEffectRecord) -> dict[str, object]:
    return {**asdict(record), "started_at": _time_json(record.started_at), "ended_at": None if record.ended_at is None else _time_json(record.ended_at)}


def _effect_from_json(value: dict[str, object]) -> ToolEffectRecord:
    return ToolEffectRecord(tool_call_id=str(value["tool_call_id"]), tool_name=str(value["tool_name"]), run_id=str(value["run_id"]), status=value["status"], started_at=_datetime(value["started_at"]), ended_at=None if value.get("ended_at") is None else _datetime(value["ended_at"]), idempotency_key=_optional_string(value.get("idempotency_key")), effect_summary=_optional_string(value.get("effect_summary")))


def _snapshot_from_json(value: dict[str, object]) -> ContinuableSnapshot:
    messages = ModelMessagesTypeAdapter.validate_python(value.get("messages", []))
    return ContinuableSnapshot(run_id=str(value["run_id"]), step_index=int(value["step_index"]), messages=messages, conversation_id=_optional_string(value.get("conversation_id")), parent_run_id=_optional_string(value.get("parent_run_id")), agent_name=_optional_string(value.get("agent_name")), timestamp=_datetime(value["timestamp"]))


def _latest_effect(values: list[dict[str, object]], tool_call_id: str) -> dict[str, object] | None:
    return next((value for value in reversed(values) if value.get("tool_call_id") == tool_call_id), None)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return parsed.astimezone(timezone.utc)


__all__ = ["DurableFileStepStore"]
