#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySwarmCommitCoordinator: atomic cross-store commit for the swarm
and swarm-step lifecycle points, with commit_id-keyed idempotent replay.

Mirrors the run-side SqlAlchemyRunCommitCoordinator shape: every operation
opens one ``Storage.transaction()`` UnitOfWork that the swarm-state write +
the commit-log entry share, and a retried call with the SAME (commit_id,
request_hash) returns the recorded result instead of re-executing. The
SAME commit_id with a DIFFERENT request_hash raises SwarmCommitConflictError.

The coordinator owns ONLY swarm-domain state (SwarmRun + SwarmStep). The
driving RunRecord's lifecycle is the RunCommitCoordinator's concern --
SwarmEngine delegates the driving Run's terminal commit to RunCoordinator
and uses this coordinator for the swarm's own state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy import Column, DateTime, LargeBinary, String
from sqlalchemy.ext.asyncio import AsyncSession

from linktools.ai.storage.sqlalchemy.models import Base

from ..commit import SwarmCommitConflictError, SwarmCommitPolicy
from .codec import SwarmCommitCodec

_CODEC = SwarmCommitCodec()


async def _append_lifecycle_event(tx: object, command: object, commit_id: str) -> None:
    """Append the typed lifecycle event inside the same SQL UoW when supplied."""
    from ...events.context import append_event_once

    event = None
    payload = getattr(command, "payload", None)
    event_context = getattr(payload, "event_context", None)
    for name in (
        "started_event", "step_event", "completed_event", "failed_event",
        "cancelled_event",
    ):
        event = getattr(payload, name, None)
        if event is not None:
            break
    if event_context is not None and event is not None:
        await append_event_once(
            tx.events,
            event_context,
            event,
            commit_id=commit_id,
            metadata={"commit_id": commit_id},
        )

if TYPE_CHECKING:
    from .sqlalchemy import SqlAlchemySwarmStore
    from ..commit import (
        CancelSwarmCommand,
        CompleteSwarmCommand,
        CompleteSwarmStepCommand,
        FailSwarmCommand,
        FailSwarmStepCommand,
        StartSwarmCommand,
        StartSwarmStepCommand,
        SwarmCommitResult,
    )
    # The Storage the coordinator participates in is referenced by a string
    # annotation only -- importing runtime.persistence.facade here would form
    # a runtime -> swarm -> runtime top-level 2-cycle (runtime imports
    # SwarmEngine; swarm must not import runtime).


class SwarmCommitLogRow(Base):
    """One row per committed swarm lifecycle point. Same shape as
    RunCommitLogRow: commit_id primary + unique, request_hash for replay-
    vs-conflict detection, result_json the deserializable recorded result."""

    __tablename__ = "swarm_commit_log"

    commit_id = Column(String(200), primary_key=True)
    operation = Column(String(64), nullable=False)
    swarm_run_id = Column(String(200), nullable=False, index=True)
    request_hash = Column(LargeBinary(32), nullable=False)
    result_json = Column(String, nullable=False)
    result_payload = Column(LargeBinary, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class SqlAlchemySwarmCommitLog:
    """Stateless read/record helper over SwarmCommitLogRow. Stateless because
    the caller owns the session (the Storage UoW); this object is just the
    table + canonical-hash plumbing, mirroring SqlAlchemyRunCommitLog."""

    async def find(
        self, session: AsyncSession, commit_id: str
    ) -> "Mapping[str, Any] | None":
        from sqlalchemy import select

        row = (
            await session.execute(
                select(SwarmCommitLogRow).where(
                    SwarmCommitLogRow.commit_id == commit_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "commit_id": row.commit_id,
            "operation": row.operation,
            "swarm_run_id": row.swarm_run_id,
            "request_hash": bytes(row.request_hash),
            "result": json.loads(row.result_json),
            "result_payload": bytes(row.result_payload),
        }

    async def record(
        self,
        session: AsyncSession,
        *,
        commit_id: str,
        operation: str,
        swarm_run_id: str,
        request_hash: bytes,
        result: Mapping[str, Any],
        result_payload: bytes | None = None,
    ) -> None:
        session.add(
            SwarmCommitLogRow(
                commit_id=commit_id,
                operation=operation,
                swarm_run_id=swarm_run_id,
                request_hash=request_hash,
                result_json=json.dumps(result, sort_keys=True),
                result_payload=result_payload or _CODEC.encode_result(operation, result),
            )
        )
        await session.flush()


async def _check_replay(
    log: SqlAlchemySwarmCommitLog,
    session: AsyncSession,
    *,
    commit_id: str,
    operation: str,
    request_payload: Mapping[str, Any],
) -> "Mapping[str, Any] | None":
    """Look up commit_id in the swarm commit log within the UoW's session.
    Return the recorded result dict if this is an idempotent replay (same id
    + same request_hash); raise SwarmCommitConflictError on a hash mismatch;
    return None on a fresh commit so the caller proceeds with the business
    writes and then records the result itself."""
    request_hash = _CODEC.request_hash(operation, request_payload)
    existing = await log.find(session, commit_id)
    if existing is None:
        return None
    if existing["request_hash"] != request_hash:
        raise SwarmCommitConflictError(
            f"swarm commit {commit_id!r} replayed with a different request"
        )
    return _CODEC.decode_result(operation, existing["result_payload"])


class SqlAlchemySwarmCommitCoordinator:
    """SQL reference implementation of SwarmCommitCoordinator. Every operation
    opens one Storage UoW that the swarm-state write + the commit-log entry
    share, and is idempotent by commit_id."""

    def __init__(
        self,
        storage: Any,
        *,
        policy: SwarmCommitPolicy,
        codec: SwarmCommitCodec,
    ) -> None:
        self._storage = storage
        self._log = SqlAlchemySwarmCommitLog()
        self._codec = codec
        self._policy = policy

    @property
    def state_store(self):
        return self._storage.swarms

    async def get_run(self, swarm_run_id: str):
        return await self._storage.swarms.get_run(swarm_run_id)

    async def update_run(self, swarm_run_id: str, *, expected_version: int, status=None, token_usage=None):
        async with self._storage.transaction() as tx:
            return await tx.swarms.update_run(
                swarm_run_id,
                expected_version=expected_version,
                status=status,
                token_usage=token_usage,
            )

    async def recover_incomplete_commits(self) -> None:
        """No-op: every SQL swarm commit is one atomic Storage.transaction()
        UoW, so a crash mid-commit rolls the whole thing back. Crash-recovery
        is the Filesystem coordinator's concern."""
        return None

    async def start(self, command: "StartSwarmCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="start",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ..models import SwarmRun

            created = await tx.swarms.create_run(command.payload.run)
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="start",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("start", command),
                result={"swarm_run_id": created.id, "version": created.version},
            )
            return {"swarm_run_id": created.id, "version": created.version}

    async def start_step(self, command: "StartSwarmStepCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="start_step",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ..models import SwarmStep

            created = await tx.swarms.create_task(command.payload.step)
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="start_step",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("start_step", command),
                result={"task_id": created.id, "version": created.version},
            )
            return {"task_id": created.id, "version": created.version}

    async def complete_step(self, command: "CompleteSwarmStepCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="complete_step",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ...run.models import RunResult

            task_id = command.payload.task_id
            result = command.payload.result
            updated = await tx.swarms.complete_task(
                task_id,
                result,
                expected_version=command.expected_version,
                active_run_id=command.payload.active_run_id,
            )
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="complete_step",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("complete_step", command),
                result={"task_id": updated.id, "version": updated.version},
            )
            return {"task_id": updated.id, "version": updated.version}

    async def fail_step(self, command: "FailSwarmStepCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="fail_step",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ...run.models import RunErrorInfo

            task_id = command.payload.task_id
            error = command.payload.error
            updated = await tx.swarms.fail_task(
                task_id,
                error,
                expected_version=command.expected_version,
                active_run_id=command.payload.active_run_id,
            )
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="fail_step",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("fail_step", command),
                result={"task_id": updated.id, "version": updated.version},
            )
            return {"task_id": updated.id, "version": updated.version}

    async def complete(self, command: "CompleteSwarmCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="complete",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ..models import SwarmStatus

            updated = await tx.swarms.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.SUCCEEDED,
            )
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="complete",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("complete", command),
                result={"swarm_run_id": updated.id, "version": updated.version},
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

    async def fail(self, command: "FailSwarmCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="fail",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ..models import SwarmStatus

            updated = await tx.swarms.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.FAILED,
            )
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="fail",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("fail", command),
                result={"swarm_run_id": updated.id, "version": updated.version},
            )
            return {"swarm_run_id": updated.id, "version": updated.version}

    async def cancel(self, command: "CancelSwarmCommand") -> "SwarmCommitResult":
        self._policy.validate(command.fence)
        commit_id = str(command.commit_id)
        async with self._storage.transaction() as tx:
            replay = await _check_replay(
                self._log,
                tx.session,
                commit_id=commit_id,
                operation="cancel",
                request_payload=command,
            )
            if replay is not None:
                return replay
            from ..models import SwarmStatus

            updated = await tx.swarms.update_run(
                command.swarm_run_id,
                expected_version=command.expected_version,
                status=SwarmStatus.CANCELLED,
            )
            await _append_lifecycle_event(tx, command, commit_id)
            await self._log.record(
                tx.session,
                commit_id=commit_id,
                operation="cancel",
                swarm_run_id=command.swarm_run_id,
                request_hash=self._codec.request_hash("cancel", command),
                result={"swarm_run_id": updated.id, "version": updated.version},
            )
            return {"swarm_run_id": updated.id, "version": updated.version}


__all__: "list[str]" = [
    "SqlAlchemySwarmCommitCoordinator",
    "SqlAlchemySwarmCommitLog",
    "SwarmCommitLogRow",
]
