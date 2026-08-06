#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomically persisted local execution records."""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ..core.errors import ErrorCode, LinktoolsAIError
from ..storage.files import read_json, write_json_atomic


@dataclass(frozen=True, slots=True)
class LocalExecutionRecord:
    project_id: str
    session_id: str
    session_revision: int
    cwd: str
    execution_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    stop_reason: "str | None"

    def __post_init__(self) -> None:
        if (
            not self.project_id.strip()
            or not self.session_id.strip()
            or self.session_revision < 0
            or not self.cwd
            or not self.execution_id.strip()
            or self.status not in {
                "PENDING_START",
                "STARTED",
                "START_UNKNOWN",
                "FAILED_START",
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "PROCESS_RESTARTED",
                "PROCESS_SHUTDOWN",
            }
            or self.created_at.tzinfo is None
            or self.updated_at.tzinfo is None
        ):
            raise ValueError("local execution record is invalid")


class LocalRecordStore:
    def __init__(self, storage_root: 'str | Path', project_id: str, *, work_root: 'str | Path | None' = None) -> None:
        self._storage_root = Path(storage_root).expanduser().resolve()
        self._work_root = self._storage_root if work_root is None else Path(work_root).expanduser().resolve()
        self._project_id = project_id
        self._directory = self._storage_root / ".linktools" / "records"
        self._records: dict[str, LocalExecutionRecord] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            self._directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(self._directory.glob("*.json")):
                record = _record_from_json(read_json(path), self._project_id, self._work_root)
                if record.execution_id != path.stem:
                    raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._records[record.execution_id] = record
            now = datetime.now(timezone.utc)
            for execution_id, record in tuple(self._records.items()):
                if record.status in {"PENDING_START", "STARTED", "START_UNKNOWN"}:
                    updated = replace(record, status="CANCELLED", updated_at=now, stop_reason="PROCESS_RESTARTED")
                    self._records[execution_id] = updated
                    self._write(updated)
            self._initialized = True

    async def save(self, record: LocalExecutionRecord) -> LocalExecutionRecord:
        await self.initialize()
        if record.project_id != self._project_id:
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_execution_id(record.execution_id)
        _validate_cwd(Path(record.cwd), self._work_root)
        async with self._lock:
            self._records[record.execution_id] = record
            self._write(record)
        return record

    async def get(self, execution_id: str) -> 'LocalExecutionRecord | None':
        await self.initialize()
        return self._records.get(execution_id)

    def _write(self, record: LocalExecutionRecord) -> None:
        write_json_atomic(self._directory / f"{record.execution_id}.json", _record_json(record), fsync=True)


def _record_json(record: LocalExecutionRecord) -> 'dict[str, str | int | bool | None]':
    return {
        "project_id": record.project_id,
        "session_id": record.session_id,
        "session_revision": record.session_revision,
        "cwd": record.cwd,
        "execution_id": record.execution_id,
        "status": record.status,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "stop_reason": record.stop_reason,
    }


def _record_from_json(
    value: 'dict[str, str | int | bool | None]',
    project_id: str,
    project_root: Path,
) -> LocalExecutionRecord:
    required = (
        "project_id",
        "session_id",
        "session_revision",
        "cwd",
        "execution_id",
        "status",
        "created_at",
        "updated_at",
    )
    if any(name not in value for name in required) or value.get("project_id") != project_id:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    cwd = _validate_cwd(Path(str(value["cwd"])), project_root)
    try:
        record = LocalExecutionRecord(
            str(value["project_id"]),
            str(value["session_id"]),
            int(str(value["session_revision"])),
            str(cwd),
            str(value["execution_id"]),
            str(value["status"]),
            datetime.fromisoformat(str(value["created_at"])),
            datetime.fromisoformat(str(value["updated_at"])),
            None if value.get("stop_reason") is None else str(value["stop_reason"]),
        )
        if (
            not record.session_id
            or record.session_revision < 0
            or not record.status
            or record.created_at.tzinfo is None
            or record.updated_at.tzinfo is None
        ):
            raise ValueError("local execution record is invalid")
        return record
    except (TypeError, ValueError) as error:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _validate_execution_id(execution_id: str) -> None:
    if not execution_id or execution_id in {".", ".."} or "/" in execution_id or "\\" in execution_id:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_cwd(cwd: Path, project_root: Path) -> Path:
    resolved = cwd.expanduser().resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise LinktoolsAIError(ErrorCode.ASSET_PATH_OUTSIDE_ROOT) from error
    return resolved


__all__ = ["LocalExecutionRecord", "LocalRecordStore"]
