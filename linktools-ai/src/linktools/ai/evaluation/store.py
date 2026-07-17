#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvalStore: persistence for the evaluation plane.

Holds the EvalRun lifecycle (PENDING -> RUNNING -> SUCCEEDED/FAILED/CANCELLED)
and the per-case EvalResult rows. The Protocol lets the runner depend on a
domain-semantic interface; concrete backends (in-memory here, file/SQL later)
satisfy it. Mirrors the TaskStore shape: create/transition/append, not generic
CRUD."""

import dataclasses
from typing import Protocol, runtime_checkable

from .models import EvalResult, EvalRun, EvalRunStatus


class EvalRunNotFoundError(Exception):
    """Raised when a transition targets an EvalRun that does not exist."""


class EvalResultConflictError(Exception):
    """Raised when a result id or run id is already stored (immutable writes)."""


@runtime_checkable
class EvalStore(Protocol):
    async def create_run(self, run: EvalRun) -> EvalRun: ...

    async def get_run(self, run_id: str) -> "EvalRun | None": ...

    async def transition_run(
        self,
        run_id: str,
        *,
        status: EvalRunStatus,
        started_at: "object | None" = None,
        finished_at: "object | None" = None,
    ) -> EvalRun: ...

    async def append_result(self, result: EvalResult) -> EvalResult: ...

    async def list_results(self, run_id: str) -> "tuple[EvalResult, ...]": ...

    async def get_result(self, result_id: str) -> "EvalResult | None": ...


class InMemoryEvalStore:
    """A process-local EvalStore. Sufficient for the runner contract and for
    tests; the file and SQL backends satisfy the same Protocol for
    cross-process persistence."""

    def __init__(self) -> None:
        self._runs: "dict[str, EvalRun]" = {}
        self._results: "dict[str, EvalResult]" = {}
        self._order: "dict[str, list[str]]" = {}

    async def create_run(self, run: EvalRun) -> EvalRun:
        if run.id in self._runs:
            raise EvalResultConflictError(f"eval run already exists: {run.id}")
        self._runs[run.id] = run
        return run

    async def get_run(self, run_id: str) -> "EvalRun | None":
        return self._runs.get(run_id)

    async def transition_run(
        self,
        run_id: str,
        *,
        status: EvalRunStatus,
        started_at: "object | None" = None,
        finished_at: "object | None" = None,
    ) -> EvalRun:
        existing = self._runs.get(run_id)
        if existing is None:
            raise EvalRunNotFoundError(f"eval run not found: {run_id}")
        updated = dataclasses.replace(
            existing,
            status=status,
            started_at=started_at if started_at is not None else existing.started_at,
            finished_at=(
                finished_at if finished_at is not None else existing.finished_at
            ),
        )
        self._runs[run_id] = updated
        return updated

    async def append_result(self, result: EvalResult) -> EvalResult:
        if result.id in self._results:
            raise EvalResultConflictError(f"eval result already exists: {result.id}")
        self._results[result.id] = result
        self._order.setdefault(result.eval_run_id, []).append(result.id)
        return result

    async def list_results(self, run_id: str) -> "tuple[EvalResult, ...]":
        return tuple(self._results[i] for i in self._order.get(run_id, []))

    async def get_result(self, result_id: str) -> "EvalResult | None":
        return self._results.get(result_id)


__all__: "list[str]" = [
    "EvalStore",
    "InMemoryEvalStore",
    "EvalRunNotFoundError",
    "EvalResultConflictError",
]
