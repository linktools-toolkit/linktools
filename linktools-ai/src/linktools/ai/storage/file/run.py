#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileRunStore: one JSON file per run under root/{run_id}.json. Atomic writes via
temp-file-then-os.replace, matching FileResourceBackend's pattern from Phase 1."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from ...errors import InvalidRunTransitionError, RunConflictError, RunNotFoundError
from ...run.models import ALLOWED_RUN_TRANSITIONS, RunErrorInfo, RunInput, RunnableType, RunRecord, RunResult, RunStatus


def _validate_id_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _to_json(record: RunRecord) -> dict:
    return {
        "id": record.id,
        "root_run_id": record.root_run_id,
        "parent_run_id": record.parent_run_id,
        "session_id": record.session_id,
        "runnable_id": record.runnable_id,
        "runnable_type": record.runnable_type.value,
        "status": record.status.value,
        "input": {"prompt": record.input.prompt, "metadata": dict(record.input.metadata)},
        "result": None if record.result is None else {
            "output": record.result.output, "token_usage": dict(record.result.token_usage), "metadata": dict(record.result.metadata),
        },
        "error": None if record.error is None else {
            "error_type": record.error.error_type, "message": record.error.message, "detail": dict(record.error.detail),
        },
        "version": record.version,
        "created_at": record.created_at.isoformat(),
        "started_at": None if record.started_at is None else record.started_at.isoformat(),
        "finished_at": None if record.finished_at is None else record.finished_at.isoformat(),
        "metadata": dict(record.metadata),
    }


def _from_json(raw: dict) -> RunRecord:
    return RunRecord(
        id=raw["id"], root_run_id=raw["root_run_id"], parent_run_id=raw["parent_run_id"],
        session_id=raw["session_id"], runnable_id=raw["runnable_id"], runnable_type=RunnableType(raw["runnable_type"]),
        status=RunStatus(raw["status"]), input=RunInput(prompt=raw["input"]["prompt"], metadata=raw["input"]["metadata"]),
        result=None if raw["result"] is None else RunResult(**raw["result"]),
        error=None if raw["error"] is None else RunErrorInfo(**raw["error"]),
        version=raw["version"], created_at=datetime.fromisoformat(raw["created_at"]),
        started_at=None if raw["started_at"] is None else datetime.fromisoformat(raw["started_at"]),
        finished_at=None if raw["finished_at"] is None else datetime.fromisoformat(raw["finished_at"]),
        metadata=raw["metadata"],
    )


class FileRunStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._root / f"{_validate_id_segment(run_id, kind='run_id')}.json"

    async def create(self, run: RunRecord) -> RunRecord:
        _atomic_write(self._path(run.id), json.dumps(_to_json(run)).encode("utf-8"))
        return run

    async def get(self, run_id: str) -> "RunRecord | None":
        path = self._path(run_id)
        if not path.exists():
            return None
        return _from_json(json.loads(path.read_text()))

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
    ) -> RunRecord:
        current = await self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if current.version != expected_version:
            raise RunConflictError(f"expected version {expected_version}, found {current.version}")
        if target not in ALLOWED_RUN_TRANSITIONS.get(current.status, frozenset()):
            raise InvalidRunTransitionError(f"cannot transition {current.status} -> {target}")
        started_at = current.started_at
        finished_at = current.finished_at
        if target == RunStatus.RUNNING and started_at is None:
            started_at = datetime.now(current.created_at.tzinfo)
        if target in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
            finished_at = datetime.now(current.created_at.tzinfo)
        updated = RunRecord(
            id=current.id, root_run_id=current.root_run_id, parent_run_id=current.parent_run_id,
            session_id=current.session_id, runnable_id=current.runnable_id, runnable_type=current.runnable_type,
            status=target, input=current.input, result=result if result is not None else current.result,
            error=error if error is not None else current.error, version=current.version + 1,
            created_at=current.created_at, started_at=started_at, finished_at=finished_at,
            metadata=current.metadata,
        )
        _atomic_write(self._path(run_id), json.dumps(_to_json(updated)).encode("utf-8"))
        return updated

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]":
        children = []
        for path in self._root.glob("*.json"):
            raw = json.loads(path.read_text())
            if raw["parent_run_id"] == run_id:
                children.append(_from_json(raw))
        return tuple(children)
