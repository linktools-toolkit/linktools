#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemSwarmStore: single-process file backend for SwarmStore (the Protocol in
swarm/store.py). One JSON file per SwarmRun under root/runs/{id}.json
and one per SwarmStep under root/tasks/{id}.json. Mirrors FilesystemRunStore's
atomic-write + path-traversal-guard patterns (see storage/filesystem/run.py).

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The ``asyncio.Lock`` is held in the async wrapper and spans the
``to_thread`` call (not the other way around), so the optimistic-concurrency
+ transition invariants still hold within one process."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from ...errors import (
    InvalidSwarmTransitionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmStepNotFoundError,
)
from ...run.models import RunErrorInfo, RunResult
from ...swarm.models import (
    ALLOWED_SWARM_TRANSITIONS,
    AttemptStatus,
    SwarmRun,
    SwarmStatus,
    SwarmStep,
    SwarmStepAttempt,
    SwarmStepStatus,
    TaskInput,
    TokenUsage,
)
from .run import _atomic_write, _validate_id_segment


def _run_to_json(run: SwarmRun) -> dict:
    return {
        "id": run.id,
        "run_id": run.run_id,
        "round": run.round,
        "status": run.status.value,
        "version": run.version,
        "token_usage": {
            "input_tokens": run.token_usage.input_tokens,
            "output_tokens": run.token_usage.output_tokens,
            "total_cost": str(run.token_usage.total_cost),
        },
        "cost": str(run.cost),
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "metadata": dict(run.metadata),
    }


def _run_from_json(raw: dict) -> SwarmRun:
    tu = raw["token_usage"]
    return SwarmRun(
        id=raw["id"],
        run_id=raw["run_id"],
        round=raw["round"],
        status=SwarmStatus(raw["status"]),
        version=raw["version"],
        token_usage=TokenUsage(
            input_tokens=tu["input_tokens"],
            output_tokens=tu["output_tokens"],
            total_cost=Decimal(tu["total_cost"]),
        ),
        cost=Decimal(raw["cost"]),
        created_at=datetime.fromisoformat(raw["created_at"]),
        updated_at=datetime.fromisoformat(raw["updated_at"]),
        metadata=raw["metadata"],
    )


def _result_to_json(result: RunResult) -> dict:
    return {
        "output": result.output,
        "token_usage": dict(result.token_usage),
        "metadata": dict(result.metadata),
    }


def _result_from_json(raw: dict) -> RunResult:
    return RunResult(
        output=raw["output"],
        token_usage=raw["token_usage"],
        metadata=raw["metadata"],
    )


def _error_to_json(error: RunErrorInfo) -> dict:
    return {
        "error_type": error.error_type,
        "message": error.message,
        "detail": dict(error.detail),
    }


def _error_from_json(raw: dict) -> RunErrorInfo:
    return RunErrorInfo(
        error_type=raw["error_type"],
        message=raw["message"],
        detail=raw["detail"],
    )


def _task_to_json(task: SwarmStep) -> dict:
    return {
        "id": task.id,
        "swarm_run_id": task.swarm_run_id,
        "parent_task_id": task.parent_task_id,
        "assigned_agent_id": task.assigned_agent_id,
        "description": task.description,
        "status": task.status.value,
        "dependencies": list(task.dependencies),
        "input": {"prompt": task.input.prompt, "metadata": dict(task.input.metadata)},
        "result": None if task.result is None else _result_to_json(task.result),
        "error": None if task.error is None else _error_to_json(task.error),
        "attempts": task.attempts,
        "version": task.version,
        "claimed_at": None if task.claimed_at is None else task.claimed_at.isoformat(),
        "lease_expires_at": None
        if task.lease_expires_at is None
        else task.lease_expires_at.isoformat(),
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        # active_run_id may be absent in older files, so the
        # reader falls back to None (dataclasses.replace + default).
        "active_run_id": task.active_run_id,
    }


def _task_from_json(raw: dict) -> SwarmStep:
    return SwarmStep(
        id=raw["id"],
        swarm_run_id=raw["swarm_run_id"],
        parent_task_id=raw["parent_task_id"],
        assigned_agent_id=raw["assigned_agent_id"],
        description=raw["description"],
        status=SwarmStepStatus(raw["status"]),
        dependencies=tuple(raw["dependencies"]),
        input=TaskInput(
            prompt=raw["input"]["prompt"], metadata=raw["input"]["metadata"]
        ),
        result=None if raw["result"] is None else _result_from_json(raw["result"]),
        error=None if raw["error"] is None else _error_from_json(raw["error"]),
        attempts=raw["attempts"],
        version=raw["version"],
        claimed_at=None
        if raw["claimed_at"] is None
        else datetime.fromisoformat(raw["claimed_at"]),
        lease_expires_at=None
        if raw["lease_expires_at"] is None
        else datetime.fromisoformat(raw["lease_expires_at"]),
        created_at=datetime.fromisoformat(raw["created_at"]),
        updated_at=datetime.fromisoformat(raw["updated_at"]),
        # Older files without active_run_id land as None.
        active_run_id=raw.get("active_run_id"),
    )


def _attempt_to_json(attempt: SwarmStepAttempt) -> dict:
    return {
        "id": attempt.id,
        "task_id": attempt.task_id,
        "run_id": attempt.run_id,
        "agent_id": attempt.agent_id,
        "attempt": attempt.attempt,
        "status": attempt.status.value,
        "started_at": attempt.started_at.isoformat(),
        "finished_at": None
        if attempt.finished_at is None
        else attempt.finished_at.isoformat(),
        "error": None if attempt.error is None else _error_to_json(attempt.error),
    }


def _attempt_from_json(raw: dict) -> SwarmStepAttempt:
    return SwarmStepAttempt(
        id=raw["id"],
        task_id=raw["task_id"],
        run_id=raw["run_id"],
        agent_id=raw["agent_id"],
        attempt=raw["attempt"],
        status=AttemptStatus(raw["status"]),
        started_at=datetime.fromisoformat(raw["started_at"]),
        finished_at=None
        if raw["finished_at"] is None
        else datetime.fromisoformat(raw["finished_at"]),
        error=None if raw["error"] is None else _error_from_json(raw["error"]),
    )


class FilesystemSwarmStore:
    """Single-process SwarmStore backed by per-record JSON files.

    ``SwarmRun`` records live at ``root/runs/{swarm_run_id}.json`` and
    ``SwarmStep`` records at ``root/tasks/{task_id}.json``. Writes are atomic
    (temp-file + ``os.replace``) and ids are validated to prevent path
    traversal. An ``asyncio.Lock`` serializes ``claim_task``/``update_run`` so
    that optimistic-concurrency invariants hold within one process.

    Explicit single-process scope:

      * Only ONE process may instantiate this store against a given ``root``.
        There is no cross-process file lock, so concurrent processes would
        observe torn writes and double-claim the same task.
      * No multi-worker ``claim_task`` race: the in-process ``asyncio.Lock`` is
        the only serializer. Two OS processes racing the same JSON file would
        corrupt the store.
      * No distributed lease: ``lease_expires_at`` is recorded but never
        observed by this backend, so ``reclaim_expired_tasks`` returns ``()``
        unconditionally. Lease semantics live on SqlAlchemySwarmStore.
      * Process restart: a caller that recovers from a crash invokes
        ``SwarmEngine.recover(swarm_run_id)``, which scans incomplete tasks and
        reconciles them with the child RunRecord + Checkpoint state (best-effort
        -- this is not a distributed coordination system). The store itself
        performs no automatic recovery; the caller owns the policy.

    ``reclaim_expired_tasks`` always returns the empty tuple: with the
    in-process lock, a task can never be observed with an expired lease while
    another coroutine holds the claim critical section, so lease expiry is
    impossible to detect at rest in this backend. (Multi-process reclaim is the
    SqlAlchemySwarmStore's responsibility.)
    """

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._runs_dir = self._root / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_dir = self._root / "tasks"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._attempts_dir = self._root / "attempts"
        self._attempts_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- paths ---------------------------------------------------------

    def _run_path(self, swarm_run_id: str) -> Path:
        return (
            self._runs_dir
            / f"{_validate_id_segment(swarm_run_id, kind='swarm_run_id')}.json"
        )

    def _task_path(self, task_id: str) -> Path:
        return self._tasks_dir / f"{_validate_id_segment(task_id, kind='task_id')}.json"

    def _attempt_path(self, attempt_id: str) -> Path:
        return (
            self._attempts_dir
            / f"{_validate_id_segment(attempt_id, kind='attempt_id')}.json"
        )

    # -- run lifecycle -------------------------------------------------

    def _create_run_sync(self, run: SwarmRun) -> SwarmRun:
        _atomic_write(
            self._run_path(run.id), json.dumps(_run_to_json(run)).encode("utf-8")
        )
        return run

    async def create_run(self, run: SwarmRun) -> SwarmRun:
        return await asyncio.to_thread(self._create_run_sync, run)

    def _get_run_sync(self, swarm_run_id: str) -> "SwarmRun | None":
        path = self._run_path(swarm_run_id)
        if not path.exists():
            return None
        return _run_from_json(json.loads(path.read_text()))

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None":
        return await asyncio.to_thread(self._get_run_sync, swarm_run_id)

    def _update_run_sync(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None",
        round: "int | None",
        token_usage: "TokenUsage | None",
        cost: "Decimal | None",
        metadata: "dict | None",
    ) -> SwarmRun:
        current = self._get_run_sync(swarm_run_id)
        if current is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if status is not None and status != current.status:
            if status not in ALLOWED_SWARM_TRANSITIONS.get(current.status, frozenset()):
                raise InvalidSwarmTransitionError(
                    f"cannot transition {current.status} -> {status}"
                )
        new_status = status if status is not None else current.status
        new_round = current.round if round is None else round
        new_token_usage = current.token_usage if token_usage is None else token_usage
        new_cost = current.cost if cost is None else cost
        new_metadata = current.metadata if metadata is None else metadata
        updated = SwarmRun(
            id=current.id,
            run_id=current.run_id,
            round=new_round,
            status=new_status,
            version=current.version + 1,
            token_usage=new_token_usage,
            cost=new_cost,
            created_at=current.created_at,
            updated_at=datetime.now(current.created_at.tzinfo),
            metadata=new_metadata,
        )
        _atomic_write(
            self._run_path(swarm_run_id),
            json.dumps(_run_to_json(updated)).encode("utf-8"),
        )
        return updated

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        round: "int | None" = None,
        token_usage: "TokenUsage | None" = None,
        cost: "Decimal | None" = None,
        metadata: "dict | None" = None,
    ) -> SwarmRun:
        async with self._lock:
            return await asyncio.to_thread(
                self._update_run_sync,
                swarm_run_id,
                expected_version=expected_version,
                status=status,
                round=round,
                token_usage=token_usage,
                cost=cost,
                metadata=metadata,
            )

    # -- task lifecycle ------------------------------------------------

    def _create_task_sync(self, task: SwarmStep) -> SwarmStep:
        _atomic_write(
            self._task_path(task.id), json.dumps(_task_to_json(task)).encode("utf-8")
        )
        return task

    async def create_task(self, task: SwarmStep) -> SwarmStep:
        return await asyncio.to_thread(self._create_task_sync, task)

    def _list_tasks_sync(
        self, swarm_run_id: str, *, status: "SwarmStepStatus | None"
    ) -> "tuple[SwarmStep, ...]":
        # swarm_run_id used as a filter, not a filename, so no path traversal risk here.
        out: list = []
        for path in self._tasks_dir.glob("*.json"):
            raw = json.loads(path.read_text())
            if raw["swarm_run_id"] != swarm_run_id:
                continue
            if status is not None and raw["status"] != status.value:
                continue
            out.append(_task_from_json(raw))
        out.sort(key=lambda t: t.created_at)
        return tuple(out)

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmStepStatus | None" = None
    ) -> "tuple[SwarmStep, ...]":
        return await asyncio.to_thread(
            self._list_tasks_sync, swarm_run_id, status=status
        )

    def _claim_task_sync(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None"
    ) -> "SwarmStep | None":
        # Snapshot of all tasks for this swarm, indexed by id for dependency lookups.
        tasks = self._list_tasks_sync(swarm_run_id, status=None)
        by_id = {t.id: t for t in tasks}
        for task in tasks:
            if task.status != SwarmStepStatus.PENDING:
                continue
            deps_ok = all(
                dep in by_id and by_id[dep].status == SwarmStepStatus.SUCCEEDED
                for dep in task.dependencies
            )
            if not deps_ok:
                continue
            # Match tz-awareness of the stored record (defensive: matches run.py style).
            src_tz = task.created_at.tzinfo
            now = datetime.now(src_tz if src_tz is not None else timezone.utc)
            lease_expires = None
            if lease_seconds is not None:
                lease_expires = now + timedelta(seconds=lease_seconds)
            claimed = SwarmStep(
                id=task.id,
                swarm_run_id=task.swarm_run_id,
                parent_task_id=task.parent_task_id,
                assigned_agent_id=agent_id,
                description=task.description,
                status=SwarmStepStatus.CLAIMED,
                dependencies=task.dependencies,
                input=task.input,
                result=task.result,
                error=task.error,
                attempts=task.attempts,
                version=task.version + 1,
                claimed_at=now,
                lease_expires_at=lease_expires,
                created_at=task.created_at,
                updated_at=now,
                # carry over any prior active_run_id (relevant on re-claim
                # after a reclaim reset; fresh PENDING tasks have None).
                active_run_id=task.active_run_id,
            )
            _atomic_write(
                self._task_path(task.id),
                json.dumps(_task_to_json(claimed)).encode("utf-8"),
            )
            return claimed
        return None

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmStep | None":
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_task_sync,
                swarm_run_id,
                agent_id,
                lease_seconds=lease_seconds,
            )

    def _set_active_run_sync(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmStep:
        # Status guard added alongside expected_version (mirrors
        # complete_task/fail_task's own fencing below): the strategy calls
        # this right after a successful claim_task (task is CLAIMED) with the
        # freshly-minted child RunRecord id. version alone already gives
        # sound optimistic concurrency here -- every mutating write bumps it,
        # so a version match implies no other write interleaved -- but the
        # explicit status check is defense-in-depth and a clearer error.
        path = self._task_path(task_id)
        if not path.exists():
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
        current = _task_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if current.status != SwarmStepStatus.CLAIMED:
            raise SwarmConflictError(
                f"task {task_id} is not claimed (status={current.status.value})"
            )
        now = datetime.now(current.created_at.tzinfo or timezone.utc)
        updated = SwarmStep(
            id=current.id,
            swarm_run_id=current.swarm_run_id,
            parent_task_id=current.parent_task_id,
            assigned_agent_id=current.assigned_agent_id,
            description=current.description,
            status=current.status,
            dependencies=current.dependencies,
            input=current.input,
            result=current.result,
            error=current.error,
            attempts=current.attempts,
            version=current.version + 1,
            claimed_at=current.claimed_at,
            lease_expires_at=current.lease_expires_at,
            created_at=current.created_at,
            updated_at=now,
            active_run_id=run_id,
        )
        _atomic_write(path, json.dumps(_task_to_json(updated)).encode("utf-8"))
        return updated

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmStep:
        async with self._lock:
            return await asyncio.to_thread(
                self._set_active_run_sync,
                task_id,
                run_id,
                expected_version=expected_version,
            )

    def _complete_task_sync(
        self,
        task_id: str,
        result: RunResult,
        *,
        expected_version: int,
        active_run_id: "str | None",
    ) -> SwarmStep:
        # expected_version is mandatory -- no more unconditional
        # fallback path. Held under self._lock (see complete_task) so the
        # read-check-write is atomic within this process.
        path = self._task_path(task_id)
        if not path.exists():
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
        current = _task_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if current.status != SwarmStepStatus.CLAIMED:
            raise SwarmConflictError(
                f"task {task_id} is not claimed (status={current.status.value})"
            )
        if active_run_id is not None and current.active_run_id != active_run_id:
            raise SwarmConflictError(
                f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                f"found {current.active_run_id!r}"
            )
        now = datetime.now(current.created_at.tzinfo or timezone.utc)
        updated = SwarmStep(
            id=current.id,
            swarm_run_id=current.swarm_run_id,
            parent_task_id=current.parent_task_id,
            assigned_agent_id=current.assigned_agent_id,
            description=current.description,
            status=SwarmStepStatus.SUCCEEDED,
            dependencies=current.dependencies,
            input=current.input,
            result=result,
            error=current.error,
            attempts=current.attempts,
            version=current.version + 1,
            claimed_at=current.claimed_at,
            lease_expires_at=current.lease_expires_at,
            created_at=current.created_at,
            updated_at=now,
            active_run_id=current.active_run_id,
        )
        _atomic_write(path, json.dumps(_task_to_json(updated)).encode("utf-8"))
        return updated

    async def complete_task(
        self,
        task_id: str,
        result: RunResult,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmStep:
        async with self._lock:
            return await asyncio.to_thread(
                self._complete_task_sync,
                task_id,
                result,
                expected_version=expected_version,
                active_run_id=active_run_id,
            )

    def _fail_task_sync(
        self,
        task_id: str,
        error: RunErrorInfo,
        *,
        expected_version: int,
        active_run_id: "str | None",
    ) -> SwarmStep:
        path = self._task_path(task_id)
        if not path.exists():
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
        current = _task_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if current.status != SwarmStepStatus.CLAIMED:
            raise SwarmConflictError(
                f"task {task_id} is not claimed (status={current.status.value})"
            )
        if active_run_id is not None and current.active_run_id != active_run_id:
            raise SwarmConflictError(
                f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                f"found {current.active_run_id!r}"
            )
        now = datetime.now(current.created_at.tzinfo or timezone.utc)
        updated = SwarmStep(
            id=current.id,
            swarm_run_id=current.swarm_run_id,
            parent_task_id=current.parent_task_id,
            assigned_agent_id=current.assigned_agent_id,
            description=current.description,
            status=SwarmStepStatus.FAILED,
            dependencies=current.dependencies,
            input=current.input,
            result=current.result,
            error=error,
            attempts=current.attempts + 1,
            version=current.version + 1,
            claimed_at=current.claimed_at,
            lease_expires_at=current.lease_expires_at,
            created_at=current.created_at,
            updated_at=now,
            active_run_id=current.active_run_id,
        )
        _atomic_write(path, json.dumps(_task_to_json(updated)).encode("utf-8"))
        return updated

    async def fail_task(
        self,
        task_id: str,
        error: RunErrorInfo,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmStep:
        async with self._lock:
            return await asyncio.to_thread(
                self._fail_task_sync,
                task_id,
                error,
                expected_version=expected_version,
                active_run_id=active_run_id,
            )

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmStep, ...]":
        # Single-process store: the in-process asyncio.Lock guarantees a task
        # cannot be observed with an expired lease while another coroutine is
        # mid-claim, so there is nothing to reclaim at rest. Returns empty.
        return ()

    # -- lease renewal ----------------------------------------

    def _renew_lease_sync(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmStep:
        path = self._task_path(task_id)
        if not path.exists():
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
        current = _task_from_json(json.loads(path.read_text()))
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if current.status != SwarmStepStatus.CLAIMED:
            raise InvalidSwarmTransitionError(
                f"renew_lease requires CLAIMED, task {task_id} is {current.status.value}"
            )
        now = datetime.now(current.created_at.tzinfo or timezone.utc)
        new_lease = now + timedelta(seconds=lease_seconds)
        updated = SwarmStep(
            id=current.id,
            swarm_run_id=current.swarm_run_id,
            parent_task_id=current.parent_task_id,
            assigned_agent_id=current.assigned_agent_id,
            description=current.description,
            status=current.status,
            dependencies=current.dependencies,
            input=current.input,
            result=current.result,
            error=current.error,
            attempts=current.attempts,
            version=current.version + 1,
            claimed_at=current.claimed_at,
            lease_expires_at=new_lease,
            created_at=current.created_at,
            updated_at=now,
            active_run_id=current.active_run_id,
        )
        _atomic_write(path, json.dumps(_task_to_json(updated)).encode("utf-8"))
        return updated

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmStep:
        async with self._lock:
            return await asyncio.to_thread(
                self._renew_lease_sync,
                task_id,
                expected_version=expected_version,
                lease_seconds=lease_seconds,
            )

    # -- attempts ---------------------------------------------

    def _record_attempt_sync(self, attempt: SwarmStepAttempt) -> SwarmStepAttempt:
        # Upsert keyed on attempt.id: the strategy writes the RUNNING row before
        # the worker call and the SUCCEEDED|FAILED row after, with the same id.
        _atomic_write(
            self._attempt_path(attempt.id),
            json.dumps(_attempt_to_json(attempt)).encode("utf-8"),
        )
        return attempt

    async def record_attempt(self, attempt: SwarmStepAttempt) -> SwarmStepAttempt:
        return await asyncio.to_thread(self._record_attempt_sync, attempt)

    def _list_attempts_sync(self, task_id: str) -> "tuple[SwarmStepAttempt, ...]":
        out: list = []
        for path in self._attempts_dir.glob("*.json"):
            raw = json.loads(path.read_text())
            if raw["task_id"] != task_id:
                continue
            out.append(_attempt_from_json(raw))
        out.sort(key=lambda a: (a.started_at, a.attempt))
        return tuple(out)

    async def list_attempts(self, task_id: str) -> "tuple[SwarmStepAttempt, ...]":
        return await asyncio.to_thread(self._list_attempts_sync, task_id)
