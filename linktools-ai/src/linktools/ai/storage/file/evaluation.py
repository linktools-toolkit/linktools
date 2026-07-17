#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileEvaluationStore: a persistent, local EvalStore.

Mirrors FileTaskStore's pattern (an asyncio.Lock across asyncio.to_thread, and
per-record JSON files written atomically with fsync) so the default-local
Storage (FileStorage) provides real cross-process evaluation persistence. The
generic to_jsonable/from_jsonable serde handles the EvalRun/EvalResult models
(Enum status, nested EvalTarget, Mapping metadata)."""

import asyncio
import dataclasses
import json
from pathlib import Path

from ...evaluation.models import EvalResult, EvalRun, EvalRunStatus
from ...evaluation.store import (
    EvalResultConflictError,
    EvalRunNotFoundError,
)
from ...clock import Clock, SystemClock
from ...json import from_jsonable, to_jsonable
from ._util import _atomic_write, _validate_id_segment


class FileEvaluationStore:
    def __init__(self, root: Path, *, clock: "Clock | None" = None) -> None:
        self._root = Path(root)
        self._lock = asyncio.Lock()
        self._clock = clock or SystemClock()

    # ----------------------------------------------------------- paths --

    def _run_path(self, run_id: str) -> Path:
        return (
            self._root
            / "runs"
            / f"{_validate_id_segment(run_id, kind='run_id')}.json"
        )

    def _result_path(self, result_id: str) -> Path:
        return (
            self._root
            / "results"
            / f"{_validate_id_segment(result_id, kind='result_id')}.json"
        )

    # ----------------------------------------------------------- helpers --

    def _write(self, path: Path, record: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(to_jsonable(record)).encode("utf-8"))

    def _read(self, path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    # ----------------------------------------------------------- API --

    async def create_run(self, run: EvalRun) -> EvalRun:
        async with self._lock:
            return await asyncio.to_thread(self._create_run_sync, run)

    async def get_run(self, run_id: str) -> "EvalRun | None":
        async with self._lock:
            return await asyncio.to_thread(self._get_run_sync, run_id)

    async def transition_run(
        self,
        run_id: str,
        *,
        status: EvalRunStatus,
        started_at: "object | None" = None,
        finished_at: "object | None" = None,
    ) -> EvalRun:
        async with self._lock:
            return await asyncio.to_thread(
                self._transition_run_sync, run_id, status, started_at, finished_at
            )

    async def append_result(self, result: EvalResult) -> EvalResult:
        async with self._lock:
            return await asyncio.to_thread(self._append_result_sync, result)

    async def list_results(self, run_id: str) -> "tuple[EvalResult, ...]":
        async with self._lock:
            return await asyncio.to_thread(self._list_results_sync, run_id)

    async def get_result(self, result_id: str) -> "EvalResult | None":
        async with self._lock:
            return await asyncio.to_thread(self._get_result_sync, result_id)

    # ----------------------------------------------------------- sync impl --

    def _create_run_sync(self, run: EvalRun) -> EvalRun:
        path = self._run_path(run.id)
        if path.exists():
            raise EvalResultConflictError(f"eval run already exists: {run.id}")
        self._write(path, run)
        return run

    def _get_run_sync(self, run_id: str) -> "EvalRun | None":
        path = self._run_path(run_id)
        if not path.exists():
            return None
        return from_jsonable(EvalRun, self._read(path))  # type: ignore[return-value]

    def _transition_run_sync(
        self, run_id, status, started_at, finished_at
    ) -> EvalRun:
        path = self._run_path(run_id)
        if not path.exists():
            raise EvalRunNotFoundError(f"eval run not found: {run_id}")
        run: EvalRun = from_jsonable(EvalRun, self._read(path))  # type: ignore[assignment]
        updated = dataclasses.replace(
            run,
            status=status,
            started_at=started_at if started_at is not None else run.started_at,
            finished_at=(
                finished_at if finished_at is not None else run.finished_at
            ),
        )
        self._write(path, updated)
        return updated

    def _append_result_sync(self, result: EvalResult) -> EvalResult:
        path = self._result_path(result.id)
        if path.exists():
            raise EvalResultConflictError(f"eval result already exists: {result.id}")
        self._write(path, result)
        return result

    def _list_results_sync(self, run_id: str) -> "tuple[EvalResult, ...]":
        results_dir = self._root / "results"
        if not results_dir.exists():
            return ()
        out: "list[EvalResult]" = []
        for p in sorted(results_dir.glob("*.json")):
            rec = from_jsonable(EvalResult, self._read(p))  # type: ignore[assignment]
            if rec.eval_run_id == run_id:
                out.append(rec)
        return tuple(out)

    def _get_result_sync(self, result_id: str) -> "EvalResult | None":
        path = self._result_path(result_id)
        if not path.exists():
            return None
        return from_jsonable(EvalResult, self._read(path))  # type: ignore[return-value]


__all__: "list[str]" = ["FileEvaluationStore"]
