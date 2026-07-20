#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A from-scratch in-memory EXTERNAL storage adapter covering the FULL agent-
run storage surface, built ONLY from the public linktools.ai Protocols and
domain models.

The companion module ``example_external_adapter.py`` proves the artifact-
domain Protocols (blob / record / lease) are sufficient on their own. This
module extends that proof to the full chain a Runtime run touches
(run -> approval -> resume -> artifact -> job): every Store Protocol is
implemented here from scratch against the public surface, then composed into a
``Storage`` whose ``root`` attribute satisfies the FilesystemRunCommit-
Coordinator's bridge contract.

IMPORT INVARIANT: the only ``linktools.*`` modules imported here are public
Protocols, their domain models, the ``Storage`` / features / transaction
composition surface, the typed error hierarchy in ``linktools.ai.errors``,
and the typed error hierarchy in ``linktools.ai.jobs.store`` (the jobs
domain's equivalent typed surface). No ``_runtime`` / ``storage.filesystem``
/ ``storage.sqlalchemy`` / ``storage.coordination`` reference backend is
imported. ``test_external_adapter_full_chain.py`` enforces this mechanically
(AST scan); a future change that reaches into a private module to satisfy a
Runtime contract is a regression of the public-surface guarantee.

ERROR MAPPING: each store raises the SAME typed error the in-repo Filesystem
reference raises for the same condition. For run / approval / swarm / memory
/ session the errors come from ``linktools.ai.errors``
(RunConflictError / RunNotFoundError / InvalidRunTransitionError,
ApprovalConflictError / ApprovalNotFoundError / InvalidApprovalTransitionError,
LostIdempotencyClaimError, SwarmConflictError / SwarmRunNotFoundError /
SwarmTaskNotFoundError / InvalidSwarmTransitionError, MemoryConflictError /
MemoryNotFoundError, SessionError). For jobs the errors come from
``linktools.ai.jobs.store`` (TaskClaimLostError / JobNotFoundError /
TaskNotFoundError / TaskBudgetExceededError / InvalidTaskCommandError /
RunnableBindingError) -- the jobs domain's own typed surface, mirroring the
Filesystem reference's ``from ...jobs.store import (...)``. The typed errors
are part of the public surface -- callers catch them by type (never by
string), so an adapter that raised stdlib ``ValueError`` / ``KeyError`` would
silently break the semantics callers depend on."""

import asyncio
import dataclasses
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from linktools.ai.agent.approval import (
    ALLOWED_APPROVAL_TRANSITIONS,
    ApprovalRequest,
    ApprovalStatus,
    build_approval_request,
    check_dedupe_conflict,
)
from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.store import AssetStore
from linktools.ai.errors import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    InvalidApprovalTransitionError,
    InvalidRunTransitionError,
    InvalidSwarmTransitionError,
    LostIdempotencyClaimError,
    MemoryConflictError,
    MemoryNotFoundError,
    RunConflictError,
    RunNotFoundError,
    SessionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from linktools.ai.events.envelope import EventEnvelope
from linktools.ai.events.payloads import EventPayload
from linktools.ai.events.store import EventPage
from linktools.ai.jobs.models import (
    ATTEMPT_TERMINAL,
    CLAIMABLE_JOB_STATUSES,
    JOB_TERMINAL,
    JOB_TRANSITIONS,
    TASK_TERMINAL,
    AttemptStatus,
    JobRecord,
    JobStatus,
    SideEffectMode,
    TaskAttemptRecord,
    TaskFailureKind,
    TaskPrincipal,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
    TaskWaitCondition,
    ActorChain,
    ActorRef,
    assert_attempt_transition,
    assert_job_transition,
    assert_task_transition,
    narrow_child_principal,
    resolve_effective_scopes,
)
from linktools.ai.jobs.protocols import (
    CancelJob,
    CancelTask,
    CompleteJob,
    CreateTask,
    SystemClock,
    TaskFailure,
    TaskSuccess,
    WaitSignal,
)
from linktools.ai.jobs.store import (
    ClaimedTask,
    InvalidTaskCommandError,
    JobNotFoundError,
    JobStore,
    RunnableBindingError,
    TaskBudgetExceededError,
    TaskClaim,
    TaskClaimLostError,
    TaskNotFoundError,
)
from linktools.ai.memory.models import MemoryRecord
from linktools.ai.memory.scope import MemoryScope
from linktools.ai.run.definition import RunDefinitionSnapshot
from linktools.ai.run.models import (
    ALLOWED_RUN_TRANSITIONS,
    NewRunCheckpoint,
    RunCheckpoint,
    RunErrorInfo,
    RunRecord,
    RunResult,
    RunStatus,
)
from linktools.ai.session.models import (
    MessageRole,
    NewSessionMessage,
    SessionMessage,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.facade import Storage
from linktools.ai.storage.features import FILE_STORAGE_FEATURES
from linktools.ai.storage.transaction import NoCrossStoreTransactions
from linktools.ai.swarm.models import (
    ALLOWED_SWARM_TRANSITIONS,
    AttemptStatus,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskAttempt,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.tool.idempotency import (
    ClaimDisposition,
    ClaimResult,
    IdempotencyClaim,
    IdempotencyRecord,
    IdempotencyStatus,
)

from .example_external_adapter import (
    InMemoryArtifactBlobStore,
    InMemoryArtifactRecordStore,
    InMemoryLeaseCoordinator,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _claim_from_record(record: IdempotencyRecord) -> IdempotencyClaim:
    """Reconstruct the fenced IdempotencyClaim a worker re-uses on a same-owner
    re-drive (RESERVED + same hash + same owner). Mirrors the Filesystem
    helper of the same name."""
    return IdempotencyClaim(
        scope=record.scope,
        key=record.key,
        request_hash=record.request_hash,
        owner_id=record.owner_id or "",
        generation=record.generation,
        claimed_at=record.claimed_at or record.created_at,
        lease_expires_at=record.lease_expires_at or record.created_at,
    )


def _fence_matches(
    record: IdempotencyRecord, claim: IdempotencyClaim, valid_statuses: "set[IdempotencyStatus]"
) -> bool:
    """A fenced transition is valid only if the record's status is one of
    ``valid_statuses`` and the owner_id + generation match -- a stale worker
    (older generation, or a lease that was stolen) cannot overwrite a newer
    owner's record. Mirrors the Filesystem fencing predicate."""
    return (
        record.status in valid_statuses
        and record.owner_id == claim.owner_id
        and record.generation == claim.generation
    )


#: Sentinel distinguishing ``approve`` (do not touch metadata) from ``reject``
#: (always record the rejection-reason key, even when the reason is None).
class _Unset:
    __slots__ = ()


_UNSET = _Unset()

#: Metadata key under which ``reject(reason=...)`` is recorded. Mirrors the
#: Filesystem constant of the same name so the audit shape round-trips.
REJECTION_REASON_METADATA_KEY = "rejection_reason"


class InMemoryRunStore:
    """Dict-backed RunStore. ``transition`` is the ONLY way status changes --
    callers never set ``record.status`` directly. The read-check-mutate cycle
    in ``transition`` / ``claim_execution`` is serialized by a single
    ``asyncio.Lock`` so two coroutines racing the same run within one process
    cannot both read the same version and both write (a lost update).

    Version bumps by +1 per transition. ``started_at`` is set on the first
    transition into RUNNING; ``finished_at`` is set on the first transition
    into a terminal status (SUCCEEDED / FAILED / CANCELLED). Both inherit the
    record's tzinfo (defensive: matches the Filesystem reference behavior of
    using ``datetime.now(current.created_at.tzinfo)``)."""

    def __init__(self) -> None:
        self._records: "dict[str, RunRecord]" = {}
        self._lock = asyncio.Lock()

    async def create(self, run: RunRecord) -> RunRecord:
        async with self._lock:
            self._records[run.id] = run
        return run

    async def get(self, run_id: str) -> "RunRecord | None":
        return self._records.get(run_id)

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
        cancel_requested_at: "datetime | None" = None,
        cancel_requested_by: "str | None" = None,
        cancel_reason: "str | None" = None,
    ) -> RunRecord:
        async with self._lock:
            current = self._records.get(run_id)
            if current is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            if current.version != expected_version:
                raise RunConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            if target not in ALLOWED_RUN_TRANSITIONS.get(current.status, frozenset()):
                raise InvalidRunTransitionError(
                    f"cannot transition {current.status} -> {target}"
                )
            tz = current.created_at.tzinfo or timezone.utc
            started_at = current.started_at
            finished_at = current.finished_at
            if target is RunStatus.RUNNING and started_at is None:
                started_at = datetime.now(tz)
            if target in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
                finished_at = datetime.now(tz)
            updated = RunRecord(
                id=current.id,
                root_run_id=current.root_run_id,
                parent_run_id=current.parent_run_id,
                session_id=current.session_id,
                runnable_id=current.runnable_id,
                runnable_type=current.runnable_type,
                status=target,
                input=current.input,
                result=result if result is not None else current.result,
                error=error if error is not None else current.error,
                version=current.version + 1,
                created_at=current.created_at,
                started_at=started_at,
                finished_at=finished_at,
                metadata=current.metadata,
                # Audit fields: a transition may carry a cancel request forward
                # (set when entering CANCELLING/CANCELLED via a Principal); once
                # set they are preserved across later transitions so the audit
                # trail survives the CANCELLING -> CANCELLED handoff.
                cancel_requested_at=(
                    cancel_requested_at
                    if cancel_requested_at is not None
                    else current.cancel_requested_at
                ),
                cancel_requested_by=(
                    cancel_requested_by
                    if cancel_requested_by is not None
                    else current.cancel_requested_by
                ),
                cancel_reason=(
                    cancel_reason
                    if cancel_reason is not None
                    else current.cancel_reason
                ),
                # Worker-fencing fields (worker_id / execution_token /
                # heartbeat_at) and manifest_id / resumability are intentionally
                # NOT carried across a transition -- mirrors the Filesystem
                # reference: a status change is a state handoff, the next
                # execution re-asserts the fence via claim_execution. Preserving
                # them would deadlock resume: the original run's execution_token
                # would conflict with resume's fresh claim.
            )
            self._records[run_id] = updated
            return updated

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]":
        return tuple(
            record for record in self._records.values() if record.parent_run_id == run_id
        )

    async def claim_execution(
        self, run_id: str, *, worker_id: str, execution_token: str
    ) -> RunRecord:
        async with self._lock:
            current = self._records.get(run_id)
            if current is None:
                raise RunNotFoundError(f"run not found: {run_id}")
            if (
                current.execution_token is not None
                and current.execution_token != execution_token
            ):
                raise RunConflictError(
                    "run execution is already fenced by another worker"
                )
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(
                current,
                worker_id=worker_id,
                execution_token=execution_token,
                heartbeat_at=datetime.now(tz),
            )
            self._records[run_id] = updated
            return updated

    async def heartbeat_execution(
        self, run_id: str, *, worker_id: str, execution_token: str
    ) -> RunRecord:
        async with self._lock:
            current = self._records.get(run_id)
            if (
                current is None
                or current.worker_id != worker_id
                or current.execution_token != execution_token
            ):
                raise RunConflictError("run execution heartbeat fencing failed")
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(current, heartbeat_at=datetime.now(tz))
            self._records[run_id] = updated
            return updated


class InMemorySessionStore:
    """Dict-backed SessionStore. The store is the SOLE sequence authority:
    ``append_messages`` reads the current max sequence and assigns fresh ones
    itself, so callers never compute ``len(prior) + 1``. Per-session locks
    serialize the read-max-then-write so two concurrent appends cannot collide.

    Each persisted message gets a fresh uuid4 id and a UTC ``created_at`` --
    both assigned here, never by the caller."""

    def __init__(self) -> None:
        self._records: "dict[str, SessionRecord]" = {}
        self._messages: "dict[str, list[SessionMessage]]" = {}
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._locks_guard = asyncio.Lock()

    async def _session_lock(self, session_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    async def create(self, session: SessionRecord) -> SessionRecord:
        lock = await self._session_lock(session.id)
        async with lock:
            self._records[session.id] = session
            self._messages.setdefault(session.id, [])
        return session

    async def get(self, session_id: str) -> "SessionRecord | None":
        return self._records.get(session_id)

    async def append_messages(
        self,
        session_id: str,
        messages: "tuple[NewSessionMessage, ...]",
    ) -> "tuple[SessionMessage, ...]":
        lock = await self._session_lock(session_id)
        async with lock:
            bucket = self._messages.setdefault(session_id, [])
            next_seq = (bucket[-1].sequence if bucket else 0) + 1
            persisted: "list[SessionMessage]" = []
            for offset, message in enumerate(messages):
                sequence = next_seq + offset
                full = SessionMessage(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    sequence=sequence,
                    role=message.role,
                    content=message.content,
                    run_id=message.run_id,
                    created_at=_utcnow(),
                    metadata=message.metadata,
                )
                bucket.append(full)
                persisted.append(full)
            return tuple(persisted)

    async def list_messages(
        self, session_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> "tuple[SessionMessage, ...]":
        bucket = self._messages.get(session_id, [])
        result = [
            message
            for message in bucket
            if message.sequence > after_sequence
        ]
        return tuple(result[:limit])

    async def update(
        self,
        session_id: str,
        *,
        status: "SessionStatus | None" = None,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> SessionRecord:
        lock = await self._session_lock(session_id)
        async with lock:
            current = self._records.get(session_id)
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
                updated_at=_utcnow(),
                metadata=metadata if metadata is not None else current.metadata,
            )
            self._records[session_id] = updated
            return updated


class InMemoryEventStore:
    """Dict-backed EventStore. The store owns sequence assignment (callers
    pass a payload + context, the store mints event_id + sequence +
    occurred_at and returns the EventEnvelope). Per-stream locks serialize
    appends so concurrent writers cannot reuse a sequence number."""

    def __init__(self) -> None:
        self._streams: "dict[str, list[EventEnvelope]]" = {}
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._locks_guard = asyncio.Lock()

    async def _stream_lock(self, stream_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(stream_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[stream_id] = lock
            return lock

    async def append(
        self,
        *,
        stream_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: "str | None",
        session_id: str,
        runnable_id: str,
        payload: EventPayload,
        metadata: "Mapping[str, Any] | None" = None,
    ) -> EventEnvelope:
        lock = await self._stream_lock(stream_id)
        async with lock:
            bucket = self._streams.setdefault(stream_id, [])
            next_seq = (bucket[-1].sequence if bucket else 0) + 1
            meta = dict(metadata) if metadata else {}
            envelope = EventEnvelope(
                event_id=str(uuid.uuid4()),
                stream_id=stream_id,
                sequence=next_seq,
                occurred_at=_utcnow(),
                run_id=run_id,
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
                session_id=session_id,
                runnable_id=runnable_id,
                payload=payload,
                metadata=meta,
            )
            bucket.append(envelope)
            return envelope

    async def list(
        self, stream_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> EventPage:
        bucket = self._streams.get(stream_id, [])
        items = [env for env in bucket if env.sequence > after_sequence]
        return EventPage(items=tuple(items[:limit]), cursor=None)


class InMemoryCheckpointStore:
    """Dict-backed CheckpointStore. Store owns sequence assignment: callers
    submit a ``NewRunCheckpoint`` and receive a persisted ``RunCheckpoint``
    with id / sequence / created_at filled in. ``latest`` returns the
    highest-sequence checkpoint for the run. Per-run locks serialize the
    read-max-then-write so two concurrent appends cannot collide."""

    def __init__(self) -> None:
        # run_id -> checkpoints in append order; index by checkpoint_id for O(1) get().
        self._by_run: "dict[str, list[RunCheckpoint]]" = {}
        self._by_id: "dict[str, RunCheckpoint]" = {}
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, run_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[run_id] = lock
            return lock

    async def append(self, checkpoint: NewRunCheckpoint) -> RunCheckpoint:
        lock = await self._lock_for(checkpoint.run_id)
        async with lock:
            bucket = self._by_run.setdefault(checkpoint.run_id, [])
            sequence = (bucket[-1].sequence if bucket else 0) + 1
            checkpoint_id = str(uuid.uuid4())
            created_at = _utcnow()
            persisted = RunCheckpoint(
                id=checkpoint_id,
                run_id=checkpoint.run_id,
                sequence=sequence,
                format=checkpoint.format,
                schema_version=checkpoint.schema_version,
                payload=checkpoint.payload,
                created_at=created_at,
                metadata=dict(checkpoint.metadata),
            )
            bucket.append(persisted)
            self._by_id[checkpoint_id] = persisted
            return persisted

    async def latest(self, run_id: str) -> "RunCheckpoint | None":
        bucket = self._by_run.get(run_id)
        if not bucket:
            return None
        return max(bucket, key=lambda checkpoint: checkpoint.sequence)

    async def get(self, checkpoint_id: str) -> "RunCheckpoint | None":
        return self._by_id.get(checkpoint_id)


class InMemoryApprovalStore:
    """Dict-backed ApprovalStore. Optimistic concurrency + transition rules
    mirror the Filesystem reference: ``approve``/``reject`` are fenced by
    ``expected_version`` and only succeed out of PENDING; ``reject`` always
    records ``metadata[REJECTION_REASON_METADATA_KEY]`` (even when reason is
    None) while ``approve`` never touches metadata (cannot shadow a prior
    rejection on the same record). ``create_or_get_pending`` dedupes on
    ``(run_id, tool_call_id)`` and surfaces a same-key/different-args conflict
    via ``check_dedupe_conflict``."""

    def __init__(self) -> None:
        self._records: "dict[str, ApprovalRequest]" = {}
        self._lock = asyncio.Lock()

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        async with self._lock:
            if request.id in self._records:
                raise ApprovalConflictError(f"approval already exists: {request.id}")
            self._records[request.id] = request
            return request

    async def create_or_get_pending(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        reason: "str | None",
        arguments: "Mapping[str, Any]",
        approval_id: str,
        binding: "Mapping[str, Any]",
    ) -> ApprovalRequest:
        async with self._lock:
            existing = [
                record
                for record in self._records.values()
                if record.run_id == run_id and record.tool_call_id == tool_call_id
            ]
            if existing:
                last = existing[-1]
                check_dedupe_conflict(
                    last,
                    tool_name=tool_name,
                    arguments=arguments,
                    arguments_hash=binding.get("arguments_hash") if binding else None,
                )
                return last
            request = build_approval_request(
                tenant_id=tenant_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason=reason,
                arguments=arguments,
                approval_id=approval_id,
                descriptor_fingerprint=binding.get("descriptor_fingerprint"),
                handler_revision=binding.get("handler_revision"),
                provider_revision=binding.get("provider_revision"),
                policy_revision=binding.get("policy_revision"),
                capability_revision=binding.get("capability_revision"),
                result_processor_revision=binding.get("result_processor_revision"),
                arguments_hash=binding.get("arguments_hash"),
            )
            self._records[request.id] = request
            return request

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        return self._records.get(approval_id)

    async def approve(
        self, approval_id: str, *, expected_version: int, resolved_by: str
    ) -> ApprovalRequest:
        return await self._resolve(
            approval_id,
            target=ApprovalStatus.APPROVED,
            expected_version=expected_version,
            resolved_by=resolved_by,
            rejection_reason=_UNSET,
        )

    async def reject(
        self,
        approval_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        reason: "str | None" = None,
    ) -> ApprovalRequest:
        return await self._resolve(
            approval_id,
            target=ApprovalStatus.REJECTED,
            expected_version=expected_version,
            resolved_by=resolved_by,
            rejection_reason=reason,
        )

    async def _resolve(
        self,
        approval_id: str,
        *,
        target: ApprovalStatus,
        expected_version: int,
        resolved_by: str,
        rejection_reason: object,
    ) -> ApprovalRequest:
        async with self._lock:
            current = self._records.get(approval_id)
            if current is None:
                raise ApprovalNotFoundError(f"approval not found: {approval_id}")
            if current.version != expected_version:
                raise ApprovalConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            if target not in ALLOWED_APPROVAL_TRANSITIONS.get(current.status, frozenset()):
                raise InvalidApprovalTransitionError(
                    f"cannot transition {current.status} -> {target}"
                )
            tz = current.created_at.tzinfo or timezone.utc
            new_metadata = dict(current.metadata)
            if rejection_reason is not _UNSET:
                new_metadata[REJECTION_REASON_METADATA_KEY] = rejection_reason
            resolved = dataclasses.replace(
                current,
                status=target,
                version=current.version + 1,
                resolved_at=datetime.now(tz),
                resolved_by=resolved_by,
                metadata=new_metadata,
            )
            self._records[approval_id] = resolved
            return resolved

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        out = [
            record
            for record in self._records.values()
            if record.run_id == run_id and record.status is ApprovalStatus.PENDING
        ]
        out.sort(key=lambda record: record.created_at)
        return tuple(out)

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        out = [record for record in self._records.values() if record.run_id == run_id]
        out.sort(key=lambda record: record.created_at)
        return tuple(out)


class InMemoryIdempotencyStore:
    """Dict-backed IdempotencyStore. The fenced-claim contract: a same-key
    same-hash RESERVED record held by the same owner is re-driven
    (ACQUIRED); held by another owner with a live lease is IN_PROGRESS;
    COMPLETED or EXECUTED is REPLAY (the side effect already happened);
    UNKNOWN is CONFLICT (cannot safely re-drive an unknowable side effect);
    FAILED or expired is re-claimed under a new generation + owner
    (ACQUIRED). ``complete``/``fail``/``mark_executed``/``mark_unknown``/``renew``
    verify the owner_id + generation match (fencing) before mutating."""

    def __init__(self) -> None:
        self._records: "dict[tuple[str, str], IdempotencyRecord]" = {}
        self._lock = asyncio.Lock()

    def _key(self, scope: str, key: str) -> "tuple[str, str]":
        return (scope, key)

    async def claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        lease_seconds: float = 300.0,
    ) -> ClaimResult:
        async with self._lock:
            k = self._key(scope, key)
            existing = self._records.get(k)
            now = _utcnow()
            lease_at = datetime.fromtimestamp(
                now.timestamp() + lease_seconds, tz=timezone.utc
            )
            if existing is None:
                return self._persist_fresh_claim(
                    scope=scope,
                    key=key,
                    request_hash=request_hash,
                    owner_id=owner_id,
                    now=now,
                    lease_at=lease_at,
                    generation=1,
                    existing_id=None,
                )
            if existing.request_hash != request_hash:
                return ClaimResult(disposition=ClaimDisposition.CONFLICT)
            if existing.status is IdempotencyStatus.COMPLETED:
                return ClaimResult(disposition=ClaimDisposition.REPLAY, record=existing)
            if existing.status is IdempotencyStatus.EXECUTED:
                return ClaimResult(disposition=ClaimDisposition.REPLAY, record=existing)
            if existing.status is IdempotencyStatus.UNKNOWN:
                return ClaimResult(disposition=ClaimDisposition.CONFLICT)
            if existing.status is IdempotencyStatus.RESERVED:
                lease_valid = (
                    existing.lease_expires_at is not None
                    and existing.lease_expires_at > now
                )
                if lease_valid and existing.owner_id == owner_id:
                    return ClaimResult(
                        disposition=ClaimDisposition.ACQUIRED,
                        claim=_claim_from_record(existing),
                    )
                if lease_valid:
                    return ClaimResult(
                        disposition=ClaimDisposition.IN_PROGRESS, record=existing
                    )
                return self._persist_fresh_claim(
                    scope=scope,
                    key=key,
                    request_hash=request_hash,
                    owner_id=owner_id,
                    now=now,
                    lease_at=lease_at,
                    generation=existing.generation + 1,
                    existing_id=existing.id,
                )
            # FAILED -> retry under a fresh generation + owner.
            return self._persist_fresh_claim(
                scope=scope,
                key=key,
                request_hash=request_hash,
                owner_id=owner_id,
                now=now,
                lease_at=lease_at,
                generation=existing.generation + 1,
                existing_id=existing.id,
            )

    def _persist_fresh_claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        now: datetime,
        lease_at: datetime,
        generation: int,
        existing_id: "str | None",
    ) -> ClaimResult:
        record = IdempotencyRecord(
            id=existing_id or str(uuid.uuid4()),
            scope=scope,
            key=key,
            request_hash=request_hash,
            status=IdempotencyStatus.RESERVED,
            result=None,
            error=None,
            created_at=now,
            completed_at=None,
            owner_id=owner_id,
            generation=generation,
            claimed_at=now,
            lease_expires_at=lease_at,
        )
        self._records[(scope, key)] = record
        return ClaimResult(
            disposition=ClaimDisposition.ACQUIRED,
            claim=_claim_from_record(record),
        )

    async def mark_executed(
        self,
        claim: IdempotencyClaim,
        result: Any,
        *,
        receipt_artifact_id: "str | None" = None,
        binding_fingerprint: "str | None" = None,
        result_processor_revision: "str | None" = None,
    ) -> None:
        async with self._lock:
            current = self._records.get((claim.scope, claim.key))
            if current is None or not _fence_matches(
                current, claim, {IdempotencyStatus.RESERVED}
            ):
                raise LostIdempotencyClaimError(
                    f"mark_executed lost the claim for ({claim.scope}, {claim.key})"
                )
            updated = dataclasses.replace(
                current,
                status=IdempotencyStatus.EXECUTED,
                result=result,
                error=None,
                receipt_artifact_id=receipt_artifact_id,
                binding_fingerprint=binding_fingerprint,
                result_processor_revision=result_processor_revision,
            )
            self._records[(claim.scope, claim.key)] = updated

    async def mark_unknown(self, claim: IdempotencyClaim) -> None:
        async with self._lock:
            current = self._records.get((claim.scope, claim.key))
            if current is None or not _fence_matches(
                current, claim, {IdempotencyStatus.RESERVED}
            ):
                raise LostIdempotencyClaimError(
                    f"mark_unknown lost the claim for ({claim.scope}, {claim.key})"
                )
            updated = dataclasses.replace(current, status=IdempotencyStatus.UNKNOWN)
            self._records[(claim.scope, claim.key)] = updated

    async def complete(self, claim: IdempotencyClaim, result: Any) -> None:
        async with self._lock:
            current = self._records.get((claim.scope, claim.key))
            if current is None or not _fence_matches(
                current,
                claim,
                {IdempotencyStatus.RESERVED, IdempotencyStatus.EXECUTED},
            ):
                raise LostIdempotencyClaimError(
                    f"complete lost the claim for ({claim.scope}, {claim.key})"
                )
            now = _utcnow()
            updated = IdempotencyRecord(
                id=current.id,
                scope=current.scope,
                key=current.key,
                request_hash=current.request_hash,
                status=IdempotencyStatus.COMPLETED,
                result=result,
                error=None,
                created_at=current.created_at,
                completed_at=now,
                owner_id=current.owner_id,
                generation=current.generation,
                claimed_at=current.claimed_at,
                lease_expires_at=current.lease_expires_at,
            )
            self._records[(claim.scope, claim.key)] = updated

    async def fail(self, claim: IdempotencyClaim, error: str) -> None:
        async with self._lock:
            current = self._records.get((claim.scope, claim.key))
            if current is None or not _fence_matches(
                current, claim, {IdempotencyStatus.RESERVED}
            ):
                raise LostIdempotencyClaimError(
                    f"fail lost the claim for ({claim.scope}, {claim.key})"
                )
            now = _utcnow()
            updated = IdempotencyRecord(
                id=current.id,
                scope=current.scope,
                key=current.key,
                request_hash=current.request_hash,
                status=IdempotencyStatus.FAILED,
                result=None,
                error=error,
                created_at=current.created_at,
                completed_at=now,
                owner_id=current.owner_id,
                generation=current.generation,
                claimed_at=current.claimed_at,
                lease_expires_at=current.lease_expires_at,
            )
            self._records[(claim.scope, claim.key)] = updated

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        return self._records.get((scope, key))

    async def renew(
        self,
        claim: IdempotencyClaim,
        *,
        now: datetime,
        lease_seconds: float,
    ) -> IdempotencyRecord:
        async with self._lock:
            current = self._records.get((claim.scope, claim.key))
            if current is None or not _fence_matches(
                current, claim, {IdempotencyStatus.RESERVED}
            ):
                raise LostIdempotencyClaimError(
                    f"renew lost the claim for ({claim.scope}, {claim.key})"
                )
            new_lease = datetime.fromtimestamp(
                now.timestamp() + lease_seconds, tz=timezone.utc
            )
            updated = dataclasses.replace(
                current, lease_expires_at=new_lease, claimed_at=now
            )
            self._records[(claim.scope, claim.key)] = updated
            return updated


class InMemoryRunDefinitionStore:
    """Dict-backed RunDefinitionStore: ``create`` stores the snapshot, ``get``
    returns it or None. The snapshot is opaque to the store (kept verbatim).
    Used by Runtime.resume to restore the EXACT AgentSpec + identity a run
    was launched with."""

    def __init__(self) -> None:
        self._snapshots: "dict[str, RunDefinitionSnapshot]" = {}

    async def create(self, snapshot: RunDefinitionSnapshot) -> None:
        self._snapshots[snapshot.run_id] = snapshot

    async def get(self, run_id: str) -> "RunDefinitionSnapshot | None":
        return self._snapshots.get(run_id)


class InMemorySwarmStore:
    """Dict-backed SwarmStore. Single-process scope: the in-process
    ``asyncio.Lock`` is the only serializer, so multi-process races are NOT
    guarded (mirrors FilesystemSwarmStore's documented scope). Optimistic-
    concurrency + status-transition rules mirror the Filesystem reference.
    ``reclaim_expired_tasks`` returns the empty tuple (a CLAIMED task can
    never be observed with an expired lease while another coroutine holds
    the in-process claim critical section)."""

    def __init__(self) -> None:
        self._runs: "dict[str, SwarmRun]" = {}
        self._tasks: "dict[str, SwarmTask]" = {}
        self._attempts: "dict[str, list[SwarmTaskAttempt]]" = {}
        self._lock = asyncio.Lock()

    async def create_run(self, run: SwarmRun) -> SwarmRun:
        async with self._lock:
            self._runs[run.id] = run
        return run

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None":
        return self._runs.get(swarm_run_id)

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        round: "int | None" = None,
        token_usage: "Any | None" = None,
        cost: "Any | None" = None,
        metadata: "dict | None" = None,
    ) -> SwarmRun:
        async with self._lock:
            current = self._runs.get(swarm_run_id)
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
            tz = current.created_at.tzinfo or timezone.utc
            updated = SwarmRun(
                id=current.id,
                run_id=current.run_id,
                round=current.round if round is None else round,
                status=status if status is not None else current.status,
                version=current.version + 1,
                token_usage=token_usage if token_usage is not None else current.token_usage,
                cost=cost if cost is not None else current.cost,
                created_at=current.created_at,
                updated_at=datetime.now(tz),
                metadata=metadata if metadata is not None else current.metadata,
            )
            self._runs[swarm_run_id] = updated
            return updated

    async def create_task(self, task: SwarmTask) -> SwarmTask:
        async with self._lock:
            self._tasks[task.id] = task
        return task

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmTask | None":
        async with self._lock:
            tasks = [
                task
                for task in self._tasks.values()
                if task.swarm_run_id == swarm_run_id
            ]
            tasks.sort(key=lambda task: task.created_at)
            by_id = {task.id: task for task in tasks}
            for task in tasks:
                if task.status is not SwarmTaskStatus.PENDING:
                    continue
                deps_ok = all(
                    dep in by_id and by_id[dep].status is SwarmTaskStatus.SUCCEEDED
                    for dep in task.dependencies
                )
                if not deps_ok:
                    continue
                now = _utcnow()
                lease_expires = (
                    None if lease_seconds is None else now + timedelta(seconds=lease_seconds)
                )
                claimed = dataclasses.replace(
                    task,
                    assigned_agent_id=agent_id,
                    status=SwarmTaskStatus.CLAIMED,
                    version=task.version + 1,
                    claimed_at=now,
                    lease_expires_at=lease_expires,
                    updated_at=now,
                )
                self._tasks[task.id] = claimed
                return claimed
            return None

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmTask:
        async with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
            if current.version != expected_version:
                raise SwarmConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(
                current,
                active_run_id=run_id,
                version=current.version + 1,
                updated_at=datetime.now(tz),
            )
            self._tasks[task_id] = updated
            return updated

    async def complete_task(
        self,
        task_id: str,
        result: RunResult,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmTask:
        async with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
            if current.version != expected_version:
                raise SwarmConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            if current.status is not SwarmTaskStatus.CLAIMED:
                raise SwarmConflictError(f"task {task_id} is not CLAIMED")
            if (
                active_run_id is not None
                and current.active_run_id is not None
                and current.active_run_id != active_run_id
            ):
                raise SwarmConflictError(
                    f"task {task_id} active_run_id mismatch: {current.active_run_id} != {active_run_id}"
                )
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(
                current,
                status=SwarmTaskStatus.SUCCEEDED,
                result=result,
                error=None,
                version=current.version + 1,
                updated_at=datetime.now(tz),
            )
            self._tasks[task_id] = updated
            return updated

    async def fail_task(
        self,
        task_id: str,
        error: RunErrorInfo,
        *,
        expected_version: int,
        active_run_id: "str | None" = None,
    ) -> SwarmTask:
        async with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
            if current.version != expected_version:
                raise SwarmConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            if current.status is not SwarmTaskStatus.CLAIMED:
                raise SwarmConflictError(f"task {task_id} is not CLAIMED")
            if (
                active_run_id is not None
                and current.active_run_id is not None
                and current.active_run_id != active_run_id
            ):
                raise SwarmConflictError(
                    f"task {task_id} active_run_id mismatch: {current.active_run_id} != {active_run_id}"
                )
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(
                current,
                status=SwarmTaskStatus.FAILED,
                error=error,
                attempts=current.attempts + 1,
                version=current.version + 1,
                updated_at=datetime.now(tz),
            )
            self._tasks[task_id] = updated
            return updated

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmTaskStatus | None" = None
    ) -> "tuple[SwarmTask, ...]":
        out = [
            task
            for task in self._tasks.values()
            if task.swarm_run_id == swarm_run_id
            and (status is None or task.status is status)
        ]
        out.sort(key=lambda task: task.created_at)
        return tuple(out)

    async def reclaim_expired_tasks(
        self, swarm_run_id: str
    ) -> "tuple[SwarmTask, ...]":
        # Single-process: a CLAIMED task can never be observed with an expired
        # lease while another coroutine holds the claim critical section.
        return ()

    async def record_attempt(self, attempt: SwarmTaskAttempt) -> SwarmTaskAttempt:
        async with self._lock:
            bucket = self._attempts.setdefault(attempt.task_id, [])
            existing_idx = next(
                (i for i, a in enumerate(bucket) if a.id == attempt.id), None
            )
            if existing_idx is None:
                bucket.append(attempt)
            else:
                bucket[existing_idx] = attempt
        return attempt

    async def list_attempts(self, task_id: str) -> "tuple[SwarmTaskAttempt, ...]":
        return tuple(self._attempts.get(task_id, ()))

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmTask:
        async with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
            if current.version != expected_version:
                raise SwarmConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            if current.status is not SwarmTaskStatus.CLAIMED:
                raise InvalidSwarmTransitionError(
                    f"renew_lease requires CLAIMED, task {task_id} is {current.status.value}"
                )
            now = _utcnow()
            tz = current.created_at.tzinfo or timezone.utc
            updated = dataclasses.replace(
                current,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                version=current.version + 1,
                updated_at=datetime.now(tz),
            )
            self._tasks[task_id] = updated
            return updated


class InMemoryMemoryStore:
    """Dict-backed MemoryStore. ``search`` is the tenant-isolation boundary:
    it scans ONLY records matching the scope's tenant_id (legacy records with
    the reserved legacy tenant are invisible to any real tenant). ``update`` /
    ``forget`` are fenced by ``expected_version``. The _UNSET sentinel
    distinguishes "leave the field alone" (omitted) from "clear the field"
    (passed as None)."""

    def __init__(self) -> None:
        self._records: "dict[str, MemoryRecord]" = {}
        self._lock = asyncio.Lock()

    async def get(self, memory_id: str) -> "MemoryRecord | None":
        return self._records.get(memory_id)

    async def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 10,
        category: "str | None" = None,
    ) -> "tuple[MemoryRecord, ...]":
        needle = query.lower()
        out: "list[MemoryRecord]" = []
        for record in self._records.values():
            if record.tenant_id != scope.tenant_id:
                continue
            if (
                scope.user_id is not None
                and record.user_id is not None
                and record.user_id != scope.user_id
            ):
                continue
            if (
                scope.workspace_id is not None
                and record.workspace_id is not None
                and record.workspace_id != scope.workspace_id
            ):
                continue
            if (
                scope.session_id is not None
                and record.session_id is not None
                and record.session_id != scope.session_id
            ):
                continue
            if category is not None and record.category != category:
                continue
            if needle and needle not in str(record.content).lower():
                continue
            out.append(record)
        out.sort(key=lambda record: record.created_at)
        return tuple(out[:limit])

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        async with self._lock:
            if record.id in self._records:
                raise MemoryConflictError(f"memory already exists: {record.id}")
            self._records[record.id] = record
        return record

    async def update(
        self,
        memory_id: str,
        *,
        expected_version: int,
        content: object = _UNSET,
        category: object = _UNSET,
        confidence: object = _UNSET,
        metadata: object = _UNSET,
    ) -> MemoryRecord:
        async with self._lock:
            current = self._records.get(memory_id)
            if current is None:
                raise MemoryNotFoundError(f"memory not found: {memory_id}")
            if current.version != expected_version:
                raise MemoryConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            new_content = current.content if content is _UNSET else content
            new_category = current.category if category is _UNSET else category
            new_confidence = current.confidence if confidence is _UNSET else confidence
            new_metadata = current.metadata if metadata is _UNSET else metadata
            tz = current.created_at.tzinfo or timezone.utc
            updated = MemoryRecord(
                id=current.id,
                tenant_id=current.tenant_id,
                owner_id=current.owner_id,
                content=new_content,  # type: ignore[arg-type]
                category=new_category,  # type: ignore[arg-type]
                confidence=new_confidence,  # type: ignore[arg-type]
                version=current.version + 1,
                created_at=current.created_at,
                updated_at=datetime.now(tz),
                metadata=new_metadata,  # type: ignore[arg-type]
                user_id=current.user_id,
                workspace_id=current.workspace_id,
                session_id=current.session_id,
            )
            self._records[memory_id] = updated
            return updated

    async def forget(self, memory_id: str, *, expected_version: int) -> None:
        async with self._lock:
            current = self._records.get(memory_id)
            if current is None:
                raise MemoryNotFoundError(f"memory not found: {memory_id}")
            if current.version != expected_version:
                raise MemoryConflictError(
                    f"expected version {expected_version}, found {current.version}"
                )
            del self._records[memory_id]


class InMemoryJobStore:
    """Dict-backed JobStore. Mirrors the Filesystem reference's semantics:
    fencing-token checks on renew_lease / bind_run / bind_runnable /
    commit_success / commit_failure (a stale worker whose lease was reclaimed
    or expired cannot overwrite the new owner's result), status-transition
    tables enforced on every move, attempt records persisted with their own
    state machine, and a per-job transition audit log.

    Single-process scope: the in-process ``asyncio.Lock`` is the only
    serializer, so multi-process races are NOT guarded (mirrors
    FilesystemTaskStore's documented scope). Dict writes are atomic in Python,
    so the Filesystem journal's crash-window proof does not apply here -- a
    mid-method crash either completes the dict mutation or leaves it
    untouched, never half-applied.

    Typed errors raised mirror FilesystemTaskStore exactly: TaskClaimLostError
    on a fencing failure, JobNotFoundError / TaskNotFoundError on absent
    records, TaskBudgetExceededError on max_tasks / max_depth breach,
    InvalidTaskCommandError on an illegal command combination,
    RunnableBindingError on a pinned-runnable drift."""

    def __init__(self, *, clock: "SystemClock | None" = None) -> None:
        self._jobs: "dict[str, JobRecord]" = {}
        # task_id -> TaskRecord. The owning job_id is on the record itself;
        # finding a task's job scans the task map (the Filesystem reference
        # walks the task files for the same result).
        self._tasks: "dict[str, TaskRecord]" = {}
        self._attempts: "dict[str, TaskAttemptRecord]" = {}
        self._transitions: "dict[str, list[TaskTransitionRecord]]" = {}
        # (job_id, signal_id) -> TaskSignalRecord. The composite key mirrors
        # the Filesystem layout (one signals dir per job, file name = signal id).
        self._signals: "dict[tuple[str, str], TaskSignalRecord]" = {}
        self._lock = asyncio.Lock()
        self._clock = clock or SystemClock()

    # --------------------------------------------------------- helpers --

    def _job_of_task(self, task_id: str) -> "tuple[str, TaskRecord] | None":
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return (task.job_id, task)

    def _append_transition(
        self,
        job_id: str,
        *,
        task_id: "str | None",
        attempt_id: "str | None",
        from_status: "str | None",
        to_status: str,
        reason: str,
        now: datetime,
        transition_id: "str | None" = None,
    ) -> None:
        bucket = self._transitions.setdefault(job_id, [])
        seq = len(bucket)
        tid = transition_id or f"{seq:010d}"
        # Idempotent on transition_id (mirrors Filesystem reference's path-
        # exists short-circuit): a journal replay that re-applies the same
        # transition is a no-op, not a duplicate audit row.
        if any(t.id == tid for t in bucket):
            return
        bucket.append(
            TaskTransitionRecord(
                id=tid,
                job_id=job_id,
                task_id=task_id,
                attempt_id=attempt_id,
                from_status=from_status,
                to_status=to_status,
                reason=reason,
                occurred_at=now,
            )
        )

    def _guard(self, job_id: str, claim: TaskClaim) -> TaskRecord:
        """Fencing predicate: the stored task must still belong to ``claim``
        (status CLAIMED or CANCELLING, matching lease_owner +
        active_attempt_id + fencing_token). A worker whose lease was reclaimed
        fails on lease_owner / fencing_token and is told it no longer owns the
        task."""
        task = self._tasks.get(claim.task_id)
        if task is None or task.job_id != job_id:
            raise TaskNotFoundError(claim.task_id)
        if (
            task.status not in (TaskStatus.CLAIMED, TaskStatus.CANCELLING)
            or task.lease_owner != claim.worker_id
            or task.active_attempt_id != claim.attempt_id
            or task.fencing_token != claim.fencing_token
        ):
            raise TaskClaimLostError(claim.task_id)
        return task

    def _converge_jobs(self, now: datetime) -> None:
        """Move jobs whose tasks are all terminal to their terminal target.
        Mirrors FilesystemTaskStore._converge_jobs_sync: RUNNING+only-WAITING-
        tasks parks at WAITING; WAITING+non-WAITING-active returns to RUNNING;
        all-terminal moves to SUCCEEDED / FAILED / CANCELLED via legal edges
        only (a WAITING job two-steps through RUNNING). Never raises -- a
        convergence pass is crash-proof."""
        for job_id, job in list(self._jobs.items()):
            if job.status in JOB_TERMINAL:
                continue
            tasks = [t for t in self._tasks.values() if t.job_id == job_id]
            if not tasks:
                continue
            statuses = {t.status for t in tasks}
            active = statuses - TASK_TERMINAL
            if active:
                if job.status is JobStatus.RUNNING and active <= {TaskStatus.WAITING}:
                    self._jobs[job_id] = dataclasses.replace(job, status=JobStatus.WAITING)
                elif (
                    job.status is JobStatus.WAITING
                    and not active <= {TaskStatus.WAITING}
                ):
                    self._jobs[job_id] = dataclasses.replace(job, status=JobStatus.RUNNING)
                continue
            if job.status is JobStatus.CANCELLING:
                target = JobStatus.CANCELLED
            elif statuses == {TaskStatus.SUCCEEDED}:
                target = JobStatus.SUCCEEDED
            elif TaskStatus.FAILED in statuses:
                target = JobStatus.FAILED
            else:
                target = JobStatus.CANCELLED
            steps = (
                [JobStatus.RUNNING, target]
                if job.status is JobStatus.WAITING
                else [target]
            )
            current = job.status
            record = job
            for nxt in steps:
                if nxt is current:
                    continue
                if nxt not in JOB_TRANSITIONS.get(current, frozenset()):
                    break
                record = dataclasses.replace(
                    record,
                    status=nxt,
                    finished_at=now if nxt in JOB_TERMINAL else record.finished_at,
                    started_at=record.started_at or now,
                )
                current = nxt
            if record.status is not job.status:
                self._jobs[job_id] = record

    def _resolve_dependencies(self, job_id: str, now: datetime) -> None:
        """Promote PENDING tasks whose dependencies have all SUCCEEDED to READY
        (claimable). Mirrors FilesystemTaskStore._resolve_dependencies."""
        for task in list(self._tasks.values()):
            if task.job_id != job_id or task.status is not TaskStatus.PENDING:
                continue
            if not task.dependencies:
                continue
            deps_ok = True
            for dep_id in task.dependencies:
                dep = self._tasks.get(dep_id)
                if dep is None or dep.status is not TaskStatus.SUCCEEDED:
                    deps_ok = False
                    break
            if deps_ok:
                assert_task_transition(task.status, TaskStatus.READY)
                ready = dataclasses.replace(
                    task,
                    status=TaskStatus.READY,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = ready
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.READY.value,
                    reason="deps_satisfied",
                    now=now,
                )

    # --------------------------------------------------------- public API --

    async def create_job(
        self, job: JobRecord, root_task: TaskRecord
    ) -> JobRecord:
        async with self._lock:
            if job.id in self._jobs:
                raise FileExistsError(job.id)
            now = job.created_at
            self._jobs[job.id] = job
            # Root task is created READY (dependencies satisfied) so it is
            # claimable; its effective delegated scopes resolve from the job's
            # actor chain (None = unrestricted = inherit).
            root = dataclasses.replace(
                root_task,
                job_id=job.id,
                status=TaskStatus.READY,
                delegated_scopes=resolve_effective_scopes(
                    root_task.delegated_scopes, job.actor_chain.delegated_scopes
                ),
            )
            self._tasks[root.id] = root
            self._append_transition(
                job.id,
                task_id=root.id,
                attempt_id=None,
                from_status=None,
                to_status=TaskStatus.READY.value,
                reason="created",
                now=now,
            )
            return job

    async def get_job(self, job_id: str) -> "JobRecord | None":
        async with self._lock:
            return self._jobs.get(job_id)

    async def get_task(self, task_id: str) -> "TaskRecord | None":
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(
        self,
        job_id: str,
        *,
        status: "TaskStatus | None" = None,
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            out = [
                task
                for task in self._tasks.values()
                if task.job_id == job_id and (status is None or task.status is status)
            ]
            out.sort(key=lambda t: t.created_at)
            return tuple(out)

    async def claim(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_seconds: float,
        handlers: "tuple[str, ...] | None" = None,
    ) -> "ClaimedTask | None":
        async with self._lock:
            # Earliest-due claimable: READY, or RETRY_WAIT whose available_at
            # has passed (promoted to READY on claim).
            best: "tuple[tuple[datetime, datetime], str, TaskRecord] | None" = None
            for task_id, task in self._tasks.items():
                if handlers is not None and task.handler not in handlers:
                    continue
                claimable = task.status is TaskStatus.READY
                if (
                    task.status is TaskStatus.RETRY_WAIT
                    and task.available_at <= now
                ):
                    claimable = True
                if not claimable or task.available_at > now:
                    continue
                job = self._jobs.get(task.job_id)
                if job is None or job.status not in CLAIMABLE_JOB_STATUSES:
                    continue
                key = (task.available_at, task.created_at)
                if best is None or key < best[0]:
                    best = (key, task.job_id, task)
            if best is None:
                return None
            _, job_id, task = best
            job = self._jobs[job_id]
            new_fencing = task.fencing_token + 1
            attempt_id = f"{task.id}-att{task.attempt_count + 1}"
            was_retry = task.status is TaskStatus.RETRY_WAIT
            pre_status = task.status.value
            if was_retry:
                assert_task_transition(task.status, TaskStatus.READY)
                task = dataclasses.replace(task, status=TaskStatus.READY)
            assert_task_transition(task.status, TaskStatus.CLAIMED)
            claimed = dataclasses.replace(
                task,
                status=TaskStatus.CLAIMED,
                lease_owner=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                fencing_token=new_fencing,
                attempt_count=task.attempt_count + 1,
                active_attempt_id=attempt_id,
                updated_at=now,
                version=task.version + 1,
            )
            self._tasks[claimed.id] = claimed
            attempt = TaskAttemptRecord(
                id=attempt_id,
                task_id=claimed.id,
                job_id=job_id,
                attempt=claimed.attempt_count,
                worker_id=worker_id,
                fencing_token=new_fencing,
                status=AttemptStatus.RUNNING,
                started_at=now,
                run_id=None,
                finished_at=None,
                failure_kind=None,
                error_type=None,
                error_message=None,
            )
            self._attempts[attempt_id] = attempt
            if was_retry:
                self._append_transition(
                    job_id,
                    task_id=claimed.id,
                    attempt_id=None,
                    from_status=pre_status,
                    to_status=TaskStatus.READY.value,
                    reason="retry_due",
                    now=now,
                )
            self._append_transition(
                job_id,
                task_id=claimed.id,
                attempt_id=attempt_id,
                from_status=TaskStatus.READY.value,
                to_status=TaskStatus.CLAIMED.value,
                reason="claimed",
                now=now,
            )
            # First claim starts the job.
            if job.status is JobStatus.PENDING:
                assert_job_transition(job.status, JobStatus.RUNNING)
                job = dataclasses.replace(
                    job,
                    status=JobStatus.RUNNING,
                    started_at=now,
                    version=job.version + 1,
                )
                self._jobs[job_id] = job
            return ClaimedTask(
                claim=TaskClaim(
                    task_id=claimed.id,
                    attempt_id=attempt_id,
                    worker_id=worker_id,
                    fencing_token=new_fencing,
                ),
                job=job,
                task=claimed,
                attempt=attempt,
            )

    async def commit_success(
        self, claim: TaskClaim, outcome: TaskSuccess
    ) -> TaskRecord:
        async with self._lock:
            now = self._clock.now()
            owner = self._job_of_task(claim.task_id)
            if owner is None:
                raise TaskNotFoundError(claim.task_id)
            job_id, _ = owner
            task = self._guard(job_id, claim)
            # Cancel precedence: a CANCELLING task lands CANCELLED regardless
            # of the handler's success.
            if task.status is TaskStatus.CANCELLING:
                return self._commit_cancelled(job_id, claim, task, now)
            # Atomic budget pre-check for child creation: count every CreateTask
            # command ONCE against the job's live task total before any child is
            # written, so a breach fails the whole commit (all or none).
            create_commands = [c for c in outcome.commands if isinstance(c, CreateTask)]
            create_keys = [c.key for c in create_commands]
            if len(create_keys) != len(set(create_keys)):
                raise ValueError("duplicate task key within commit")
            if create_commands:
                job = self._jobs[job_id]
                self._assert_child_budget(job_id, job, task, create_commands)
            # CompleteJob all-or-none gate: cannot both create children and
            # complete the job.
            if any(isinstance(c, CompleteJob) for c in outcome.commands):
                live_siblings = [
                    t
                    for t in self._tasks.values()
                    if t.job_id == job_id
                    and t.id != task.id
                    and t.status not in TASK_TERMINAL
                ]
                if create_commands or live_siblings:
                    raise InvalidTaskCommandError(
                        "CompleteJob requires the committing task to be the "
                        "only non-terminal task; it cannot combine with "
                        "CreateTask or run alongside live siblings"
                    )
            has_wait = any(isinstance(c, WaitSignal) for c in outcome.commands)
            target = TaskStatus.WAITING if has_wait else TaskStatus.SUCCEEDED
            assert_task_transition(task.status, target)
            from_status = task.status.value
            new_task = dataclasses.replace(
                task,
                status=target,
                output_artifact_id=(
                    outcome.output_artifact.id
                    if outcome.output_artifact is not None
                    else None
                ),
                lease_owner=None,
                lease_expires_at=None,
                active_attempt_id=None,
                updated_at=now,
                version=task.version + 1,
            )
            attempt = self._attempts.get(claim.attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.RUNNING:
                # Already-SUCCEEDED attempt is a journal replay; otherwise the
                # attempt state machine must allow the move.
                if attempt is not None and attempt.status is AttemptStatus.SUCCEEDED:
                    new_attempt = attempt
                else:
                    raise TaskClaimLostError(claim.task_id)
            else:
                assert_attempt_transition(attempt.status, AttemptStatus.SUCCEEDED)
                new_attempt = dataclasses.replace(
                    attempt, status=AttemptStatus.SUCCEEDED, finished_at=now
                )
                self._attempts[claim.attempt_id] = new_attempt
            if has_wait:
                wait_signals = [c for c in outcome.commands if isinstance(c, WaitSignal)]
                if len(wait_signals) > 1:
                    raise InvalidTaskCommandError(
                        "a task may wait for only one signal condition"
                    )
                ws = wait_signals[0]
                deadline = (
                    now + timedelta(seconds=ws.timeout_seconds)
                    if ws.timeout_seconds is not None
                    else None
                )
                new_task = dataclasses.replace(
                    new_task,
                    wait_conditions=(
                        TaskWaitCondition(
                            name=ws.name, correlation_key=ws.correlation_key
                        ),
                    ),
                    wait_deadline_at=deadline,
                )
            # Apply commands atomically (attempt + task flipped after).
            for cmd in outcome.commands:
                if isinstance(cmd, CreateTask):
                    self._apply_create_task(job_id, task, cmd, now)
                elif isinstance(cmd, CompleteJob):
                    self._complete_job(job_id, claim.task_id, cmd, now)
                elif isinstance(cmd, CancelTask):
                    self._cancel_task_in_job(job_id, cmd, now)
                elif isinstance(cmd, CancelJob):
                    self._request_cancel_impl(job_id, cmd.reason, now)
            self._tasks[new_task.id] = new_task
            self._append_transition(
                job_id,
                task_id=new_task.id,
                attempt_id=claim.attempt_id,
                from_status=from_status,
                to_status=target.value,
                reason="wait_signal" if has_wait else "succeeded",
                now=now,
            )
            self._resolve_dependencies(job_id, now)
            self._converge_jobs(now)
            return new_task

    def _assert_child_budget(
        self,
        job_id: str,
        job: JobRecord,
        parent: TaskRecord,
        create_commands: "list[CreateTask]",
    ) -> None:
        max_depth = job.budget.max_depth
        child_depth = parent.depth + 1
        if max_depth is not None and child_depth > max_depth:
            raise TaskBudgetExceededError(
                f"task depth {child_depth} exceeds max_depth {max_depth}"
            )
        max_tasks = job.budget.max_tasks
        if max_tasks is not None:
            current = sum(1 for t in self._tasks.values() if t.job_id == job_id)
            if current + len(create_commands) > max_tasks:
                raise TaskBudgetExceededError(
                    f"job {job_id} task budget exhausted: {current}/{max_tasks}"
                )

    def _apply_create_task(
        self, job_id: str, parent: TaskRecord, cmd: CreateTask, now: datetime
    ) -> None:
        job = self._jobs[job_id]
        for existing in [t for t in self._tasks.values() if t.job_id == job_id]:
            if existing.key == cmd.key:
                if existing.parent_task_id == parent.id:
                    expected_input = cmd.input_artifact.id if cmd.input_artifact else None
                    if (
                        existing.handler != cmd.handler
                        or existing.input_artifact_id != expected_input
                        or existing.dependencies != tuple(cmd.dependencies)
                        or existing.retry_policy != cmd.retry_policy
                        or existing.side_effect_policy != cmd.side_effect_policy
                        or existing.timeout_seconds != cmd.timeout_seconds
                        or dict(existing.metadata) != dict(cmd.metadata)
                    ):
                        raise ValueError(
                            f"conflicting task definition for key {cmd.key!r}"
                        )
                    return
                raise ValueError(f"duplicate task key {cmd.key!r} in job {job_id}")
        child_depth = parent.depth + 1
        if job.budget.max_depth is not None and child_depth > job.budget.max_depth:
            raise TaskBudgetExceededError(
                f"task depth {child_depth} exceeds max_depth {job.budget.max_depth}"
            )
        child_scopes, child_chain = narrow_child_principal(
            parent, cmd.delegated_scopes, cmd.handler, job.actor_chain
        )
        child_id = (
            f"{parent.id}-{cmd.key}-"
            f"{uuid.uuid5(uuid.NAMESPACE_URL, f'{job_id}:{parent.id}:{cmd.key}').hex[:8]}"
        )
        child = TaskRecord(
            id=child_id,
            job_id=job_id,
            parent_task_id=parent.id,
            key=cmd.key,
            handler=cmd.handler,
            status=TaskStatus.PENDING if cmd.dependencies else TaskStatus.READY,
            input_artifact_id=cmd.input_artifact.id if cmd.input_artifact else None,
            output_artifact_id=None,
            dependencies=cmd.dependencies,
            retry_policy=cmd.retry_policy,
            side_effect_policy=cmd.side_effect_policy,
            attempt_count=0,
            available_at=now,
            lease_owner=None,
            lease_expires_at=None,
            fencing_token=0,
            active_attempt_id=None,
            timeout_seconds=cmd.timeout_seconds,
            resource_snapshots=parent.resource_snapshots,
            version=1,
            created_at=now,
            updated_at=now,
            depth=child_depth,
            delegated_scopes=child_scopes,
            actor_chain=child_chain,
            metadata=dict(cmd.metadata),
        )
        self._tasks[child_id] = child
        self._append_transition(
            job_id,
            task_id=child_id,
            attempt_id=None,
            from_status=None,
            to_status=child.status.value,
            reason="created",
            now=now,
        )

    def _complete_job(
        self, job_id: str, current_task_id: str, cmd: CompleteJob, now: datetime
    ) -> None:
        job = self._jobs[job_id]
        if job.status in JOB_TERMINAL:
            return
        tasks = [t for t in self._tasks.values() if t.job_id == job_id]
        live = [
            t
            for t in tasks
            if t.id != current_task_id and t.status not in TASK_TERMINAL
        ]
        if live:
            ids = ", ".join(t.id for t in live[:5])
            raise InvalidTaskCommandError(
                "CompleteJob requires all sibling tasks to be terminal; "
                f"non-terminal tasks: {ids}"
            )
        output_artifact_id = cmd.output_artifact.id if cmd.output_artifact else None
        steps = (
            [JobStatus.RUNNING, JobStatus.SUCCEEDED]
            if job.status is JobStatus.WAITING
            else [JobStatus.SUCCEEDED]
        )
        current = job.status
        record = job
        for nxt in steps:
            if nxt is current:
                continue
            if nxt not in JOB_TRANSITIONS.get(current, frozenset()):
                break
            record = dataclasses.replace(
                record,
                status=nxt,
                finished_at=now if nxt in JOB_TERMINAL else record.finished_at,
                output_artifact_id=(
                    output_artifact_id
                    if nxt is JobStatus.SUCCEEDED
                    else record.output_artifact_id
                ),
                version=record.version + 1,
            )
            current = nxt
        if record.status is not job.status:
            self._jobs[job_id] = record

    def _cancel_task_in_job(self, job_id: str, cmd: CancelTask, now: datetime) -> None:
        ct = self._tasks.get(cmd.task_id)
        if ct is None or ct.job_id != job_id:
            raise InvalidTaskCommandError(
                f"CancelTask target {cmd.task_id!r} is not in job {job_id}"
            )
        if ct.status in TASK_TERMINAL:
            return
        pre = ct.status
        assert_task_transition(pre, TaskStatus.CANCELLING)
        assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
        cancelled = dataclasses.replace(
            ct,
            status=TaskStatus.CANCELLED,
            updated_at=now,
            version=ct.version + 1,
        )
        self._tasks[ct.id] = cancelled
        if pre is not TaskStatus.CANCELLING:
            self._append_transition(
                job_id,
                task_id=ct.id,
                attempt_id=None,
                from_status=pre.value,
                to_status=TaskStatus.CANCELLING.value,
                reason="cancelled",
                now=now,
            )
        self._append_transition(
            job_id,
            task_id=ct.id,
            attempt_id=None,
            from_status=TaskStatus.CANCELLING.value,
            to_status=TaskStatus.CANCELLED.value,
            reason="cancelled",
            now=now,
        )

    def _commit_cancelled(
        self, job_id: str, claim: TaskClaim, task: TaskRecord, now: datetime
    ) -> TaskRecord:
        attempt = self._attempts.get(claim.attempt_id)
        if attempt is None:
            raise TaskClaimLostError(claim.task_id)
        assert_attempt_transition(attempt.status, AttemptStatus.CANCELLED)
        self._attempts[claim.attempt_id] = dataclasses.replace(
            attempt, status=AttemptStatus.CANCELLED, finished_at=now
        )
        assert_task_transition(task.status, TaskStatus.CANCELLED)
        new_task = dataclasses.replace(
            task,
            status=TaskStatus.CANCELLED,
            lease_owner=None,
            lease_expires_at=None,
            active_attempt_id=None,
            updated_at=now,
            version=task.version + 1,
        )
        self._tasks[new_task.id] = new_task
        self._append_transition(
            job_id,
            task_id=new_task.id,
            attempt_id=claim.attempt_id,
            from_status=task.status.value,
            to_status=TaskStatus.CANCELLED.value,
            reason="cancelled",
            now=now,
        )
        self._converge_jobs(now)
        return new_task

    async def commit_failure(
        self, claim: TaskClaim, outcome: TaskFailure
    ) -> TaskRecord:
        async with self._lock:
            now = self._clock.now()
            owner = self._job_of_task(claim.task_id)
            if owner is None:
                raise TaskNotFoundError(claim.task_id)
            job_id, _ = owner
            task = self._guard(job_id, claim)
            if task.status is TaskStatus.CANCELLING:
                return self._commit_cancelled(job_id, claim, task, now)
            attempt = self._attempts.get(claim.attempt_id)
            if attempt is None:
                raise TaskClaimLostError(claim.task_id)
            assert_attempt_transition(attempt.status, AttemptStatus.FAILED)
            self._attempts[claim.attempt_id] = dataclasses.replace(
                attempt,
                status=AttemptStatus.FAILED,
                finished_at=now,
                failure_kind=outcome.kind,
                error_type=outcome.error_type,
                error_message=outcome.message,
            )
            retryable = outcome.retryable
            if retryable is None:
                retryable = outcome.kind in task.retry_policy.retryable_kinds
            non_idempotent = (
                task.side_effect_policy.mode is SideEffectMode.NON_IDEMPOTENT
            )
            can_retry = (
                retryable
                and not non_idempotent
                and task.attempt_count < task.retry_policy.max_attempts
            )
            from_status = task.status.value
            if can_retry:
                assert_task_transition(task.status, TaskStatus.RETRY_WAIT)
                new_task = dataclasses.replace(
                    task,
                    status=TaskStatus.RETRY_WAIT,
                    lease_owner=None,
                    lease_expires_at=None,
                    active_attempt_id=None,
                    available_at=now,
                    updated_at=now,
                    version=task.version + 1,
                )
                to_status = TaskStatus.RETRY_WAIT.value
                reason = "retry"
            else:
                assert_task_transition(task.status, TaskStatus.FAILED)
                new_task = dataclasses.replace(
                    task,
                    status=TaskStatus.FAILED,
                    lease_owner=None,
                    lease_expires_at=None,
                    active_attempt_id=None,
                    updated_at=now,
                    version=task.version + 1,
                )
                to_status = TaskStatus.FAILED.value
                reason = "failed"
            self._tasks[new_task.id] = new_task
            self._append_transition(
                job_id,
                task_id=new_task.id,
                attempt_id=claim.attempt_id,
                from_status=from_status,
                to_status=to_status,
                reason=reason,
                now=now,
            )
            self._converge_jobs(now)
            return new_task

    def _request_cancel_impl(
        self, job_id: str, reason: "str | None", now: datetime
    ) -> JobRecord:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status in JOB_TERMINAL:
            return job
        if job.status is not JobStatus.CANCELLING:
            assert_job_transition(job.status, JobStatus.CANCELLING)
        job = dataclasses.replace(
            job,
            status=JobStatus.CANCELLING,
            version=job.version + 1,
        )
        self._jobs[job_id] = job
        for task in list(self._tasks.values()):
            if task.job_id != job_id:
                continue
            if task.status in (
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.WAITING,
                TaskStatus.RETRY_WAIT,
            ):
                assert_task_transition(task.status, TaskStatus.CANCELLING)
                assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
                cancelled = dataclasses.replace(
                    task,
                    status=TaskStatus.CANCELLED,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = cancelled
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.CANCELLED.value,
                    reason="cancelled",
                    now=now,
                )
            elif task.status is TaskStatus.CLAIMED:
                assert_task_transition(task.status, TaskStatus.CANCELLING)
                cancelling = dataclasses.replace(
                    task,
                    status=TaskStatus.CANCELLING,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = cancelling
                self._append_transition(
                    job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.CANCELLING.value,
                    reason="cancelling",
                    now=now,
                )
        # Job cancels outright only when nothing is still in-flight.
        if not any(
            t.status in (TaskStatus.CLAIMED, TaskStatus.CANCELLING)
            for t in self._tasks.values()
            if t.job_id == job_id
        ):
            assert_job_transition(job.status, JobStatus.CANCELLED)
            job = dataclasses.replace(
                job, status=JobStatus.CANCELLED, finished_at=now
            )
            self._jobs[job_id] = job
        return job

    async def request_cancel(
        self, job_id: str, *, reason: "str | None" = None
    ) -> JobRecord:
        async with self._lock:
            return self._request_cancel_impl(job_id, reason, self._clock.now())

    async def submit_signal(self, signal: TaskSignalRecord) -> TaskSignalRecord:
        async with self._lock:
            now = signal.created_at
            key = (signal.job_id, signal.id)
            existing = self._signals.get(key)
            if existing is not None:
                return existing
            self._signals[key] = signal
            current_signal = signal
            for task in list(self._tasks.values()):
                if task.job_id != signal.job_id or task.status is not TaskStatus.WAITING:
                    continue
                matched = any(
                    c.name == signal.name and c.correlation_key == signal.correlation_key
                    for c in task.wait_conditions
                )
                if not matched:
                    continue
                assert_task_transition(task.status, TaskStatus.READY)
                woken = dataclasses.replace(
                    task,
                    status=TaskStatus.READY,
                    available_at=now,
                    wait_conditions=(),
                    wait_deadline_at=None,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = woken
                self._append_transition(
                    signal.job_id,
                    task_id=task.id,
                    attempt_id=None,
                    from_status=task.status.value,
                    to_status=TaskStatus.READY.value,
                    reason="signal",
                    now=now,
                )
                current_signal = dataclasses.replace(
                    current_signal, consumed_by_task_id=woken.id
                )
                self._signals[key] = current_signal
                break
            self._converge_jobs(now)
            return current_signal

    async def recover_expired(
        self, *, now: datetime, limit: int = 100
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            recovered: "list[TaskRecord]" = []
            for task in list(self._tasks.values()):
                if len(recovered) >= limit:
                    break
                expired = bool(
                    task.lease_expires_at is not None
                    and task.lease_expires_at < now
                )
                if task.status is TaskStatus.CANCELLING and expired:
                    assert_task_transition(task.status, TaskStatus.CANCELLED)
                    reset = dataclasses.replace(
                        task,
                        status=TaskStatus.CANCELLED,
                        lease_owner=None,
                        lease_expires_at=None,
                        active_attempt_id=None,
                        updated_at=now,
                        version=task.version + 1,
                    )
                    self._tasks[task.id] = reset
                    if task.active_attempt_id:
                        att = self._attempts.get(task.active_attempt_id)
                        if att is not None and att.status is AttemptStatus.RUNNING:
                            self._attempts[task.active_attempt_id] = dataclasses.replace(
                                att, status=AttemptStatus.CANCELLED, finished_at=now
                            )
                    self._append_transition(
                        task.job_id,
                        task_id=task.id,
                        attempt_id=task.active_attempt_id,
                        from_status=task.status.value,
                        to_status=TaskStatus.CANCELLED.value,
                        reason="cancelled",
                        now=now,
                    )
                    recovered.append(reset)
                    continue
                if not (task.status is TaskStatus.CLAIMED and expired):
                    continue
                non_idempotent = (
                    task.side_effect_policy.mode is SideEffectMode.NON_IDEMPOTENT
                )
                exhausted = task.attempt_count >= task.retry_policy.max_attempts
                target = (
                    TaskStatus.FAILED
                    if (non_idempotent or exhausted)
                    else TaskStatus.READY
                )
                assert_task_transition(task.status, target)
                reset = dataclasses.replace(
                    task,
                    status=target,
                    lease_owner=None,
                    lease_expires_at=None,
                    active_attempt_id=None,
                    available_at=now,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = reset
                if task.active_attempt_id:
                    att = self._attempts.get(task.active_attempt_id)
                    if att is not None and att.status is AttemptStatus.RUNNING:
                        self._attempts[task.active_attempt_id] = dataclasses.replace(
                            att, status=AttemptStatus.SUPERSEDED, finished_at=now
                        )
                if non_idempotent:
                    reason = "non_idempotent"
                elif exhausted:
                    reason = "attempts_exhausted"
                else:
                    reason = "lease_expired"
                self._append_transition(
                    task.job_id,
                    task_id=task.id,
                    attempt_id=task.active_attempt_id,
                    from_status=TaskStatus.CLAIMED.value,
                    to_status=target.value,
                    reason=reason,
                    now=now,
                )
                recovered.append(reset)
            self._converge_jobs(now)
            return tuple(recovered)

    async def reconcile_due(
        self, *, now: datetime, limit: int = 100
    ) -> "tuple[TaskRecord, ...]":
        async with self._lock:
            handled: "list[TaskRecord]" = []
            for task in list(self._tasks.values()):
                if len(handled) >= limit:
                    break
                if task.status is not TaskStatus.WAITING:
                    continue
                if task.wait_deadline_at is None or task.wait_deadline_at > now:
                    continue
                retryable = (
                    TaskFailureKind.TIMEOUT in task.retry_policy.retryable_kinds
                    and task.attempt_count < task.retry_policy.max_attempts
                )
                if retryable:
                    assert_task_transition(task.status, TaskStatus.READY)
                    reset = dataclasses.replace(
                        task,
                        status=TaskStatus.READY,
                        available_at=now,
                        wait_conditions=(),
                        wait_deadline_at=None,
                        updated_at=now,
                        version=task.version + 1,
                    )
                    self._tasks[task.id] = reset
                    self._append_transition(
                        task.job_id,
                        task_id=task.id,
                        attempt_id=None,
                        from_status=task.status.value,
                        to_status=TaskStatus.READY.value,
                        reason="signal_timeout_retry",
                        now=now,
                    )
                else:
                    assert_task_transition(task.status, TaskStatus.CANCELLING)
                    assert_task_transition(TaskStatus.CANCELLING, TaskStatus.CANCELLED)
                    reset = dataclasses.replace(
                        task,
                        status=TaskStatus.CANCELLED,
                        lease_owner=None,
                        lease_expires_at=None,
                        active_attempt_id=None,
                        wait_conditions=(),
                        wait_deadline_at=None,
                        updated_at=now,
                        version=task.version + 1,
                    )
                    self._tasks[task.id] = reset
                    self._append_transition(
                        task.job_id,
                        task_id=task.id,
                        attempt_id=None,
                        from_status=task.status.value,
                        to_status=TaskStatus.CANCELLING.value,
                        reason="signal_timeout",
                        now=now,
                    )
                    self._append_transition(
                        task.job_id,
                        task_id=task.id,
                        attempt_id=None,
                        from_status=TaskStatus.CANCELLING.value,
                        to_status=TaskStatus.CANCELLED.value,
                        reason="signal_timeout",
                        now=now,
                    )
                handled.append(reset)
            self._converge_jobs(now)
            return tuple(handled)

    async def list_orphan_run_ids(self, *, limit: int = 500) -> "tuple[str, ...]":
        async with self._lock:
            seen: "set[str]" = set()
            out: "list[str]" = []
            for attempt in self._attempts.values():
                if (
                    attempt.status is AttemptStatus.SUPERSEDED
                    and attempt.run_id
                    and attempt.run_id not in seen
                ):
                    seen.add(attempt.run_id)
                    out.append(attempt.run_id)
                    if len(out) >= limit:
                        break
            return tuple(out)

    async def list_attempts(
        self, task_id: str
    ) -> "tuple[TaskAttemptRecord, ...]":
        async with self._lock:
            out = [
                attempt
                for attempt in self._attempts.values()
                if attempt.task_id == task_id
            ]
            out.sort(key=lambda a: a.started_at)
            return tuple(out)

    async def list_transitions(
        self, job_id: str
    ) -> "tuple[TaskTransitionRecord, ...]":
        async with self._lock:
            return tuple(self._transitions.get(job_id, ()))

    async def renew_lease(self, **kwargs) -> TaskRecord:
        async with self._lock:
            now = kwargs["now"]
            owner = self._job_of_task(kwargs["task_id"])
            if owner is None:
                raise TaskNotFoundError(kwargs["task_id"])
            job_id, _ = owner
            claim = TaskClaim(
                task_id=kwargs["task_id"],
                attempt_id=kwargs["attempt_id"],
                worker_id=kwargs["worker_id"],
                fencing_token=kwargs["fencing_token"],
            )
            task = self._guard(job_id, claim)
            renewed = dataclasses.replace(
                task,
                lease_expires_at=now + timedelta(seconds=kwargs["lease_seconds"]),
                updated_at=now,
                version=task.version + 1,
            )
            self._tasks[task.id] = renewed
            return renewed

    async def bind_run(self, **kwargs) -> TaskAttemptRecord:
        async with self._lock:
            owner = self._job_of_task(kwargs["task_id"])
            if owner is None:
                raise TaskNotFoundError(kwargs["task_id"])
            job_id, _ = owner
            claim = TaskClaim(
                task_id=kwargs["task_id"],
                attempt_id=kwargs["attempt_id"],
                worker_id=kwargs["worker_id"],
                fencing_token=kwargs["fencing_token"],
            )
            self._guard(job_id, claim)
            attempt = self._attempts.get(kwargs["attempt_id"])
            if attempt is None:
                raise TaskNotFoundError(kwargs["attempt_id"])
            bound = dataclasses.replace(attempt, run_id=kwargs["run_id"])
            self._attempts[attempt.id] = bound
            return bound

    async def bind_runnable(self, **kwargs) -> TaskRecord:
        async with self._lock:
            owner = self._job_of_task(kwargs["task_id"])
            if owner is None:
                raise TaskNotFoundError(kwargs["task_id"])
            job_id, _ = owner
            claim = TaskClaim(
                task_id=kwargs["task_id"],
                attempt_id=kwargs["attempt_id"],
                worker_id=kwargs["worker_id"],
                fencing_token=kwargs["fencing_token"],
            )
            task = self._guard(job_id, claim)
            now = self._clock.now()
            runnable_id = kwargs["runnable_id"]
            revision = kwargs["revision"]
            fingerprint = kwargs["fingerprint"]
            if task.resolved_runnable_id is None:
                bound = dataclasses.replace(
                    task,
                    resolved_runnable_id=runnable_id,
                    resolved_runnable_revision=revision,
                    resolved_runnable_fingerprint=fingerprint,
                    updated_at=now,
                    version=task.version + 1,
                )
                self._tasks[task.id] = bound
                return bound
            if (
                task.resolved_runnable_id == runnable_id
                and task.resolved_runnable_revision == revision
                and task.resolved_runnable_fingerprint == fingerprint
            ):
                return task
            raise RunnableBindingError(
                f"runnable binding drift on task {task.id}: pinned "
                f"{task.resolved_runnable_id}/"
                f"{task.resolved_runnable_revision}/"
                f"{task.resolved_runnable_fingerprint} != resolved "
                f"{runnable_id}/{revision}/{fingerprint}"
            )


class InMemoryExternalStorage(Storage):
    """Composition of the in-memory chain stores into one frozen ``Storage``
    subclass. ``root`` is provided so the FilesystemRunCommitCoordinator (the
    default commit coordinator Runtime.build selects when
    ``features.transactions`` is PROCESS_LOCAL) can place its crash-recovery
    journal under ``{root}/transactions`` -- Runtime's internal choice, NOT
    an adapter import of a private module.

    The state is held by the per-store dicts (each store constructed once and
    shared across callers of this Storage); the frozen dataclass holds only
    immutable references to those stores, so the ``object.__setattr__`` bypass
    for ``_root`` is the only post-construction mutation and matches the
    pattern ``FilesystemStorage`` established."""

    def __init__(self, *, root: Path, **fields: Any) -> None:
        super().__init__(**fields)
        object.__setattr__(self, "_root", Path(root))

    @property
    def root(self) -> Path:
        """Directory the FilesystemRunCommitCoordinator uses for its crash-
        recovery journal (``{root}/transactions``). The in-memory stores
        themselves do not read or write this directory; only Runtime's
        commit-coordinator construction does."""
        return self._root


def build_in_memory_external_storage(*, root: Path) -> InMemoryExternalStorage:
    """Wire every Store Protocol to a fresh in-memory implementation and
    return a ``Storage`` whose ``root`` satisfies the FilesystemRunCommit-
    Coordinator bridge contract.

    Features are declared as ``FILE_STORAGE_FEATURES`` so Runtime.build
    selects the sequential ``FilesystemRunCommitCoordinator`` (the same it
    selects for ``FilesystemStorage``). Transactions are
    ``NoCrossStoreTransactions`` -- the in-memory stores are independent
    (no shared atomic scope), which the honest PROCESS_LOCAL declaration
    reflects. Coordination is the existing ``InMemoryLeaseCoordinator``;
    artifacts reuse the existing ``InMemoryArtifactBlobStore`` /
    ``InMemoryArtifactRecordStore``; assets use the public
    ``AssetStore(primary=MemoryAssetBackend())`` composition; tasks uses
    the new ``InMemoryJobStore`` so a downstream wiring ``JobRuntime`` on
    this Storage gets a real JobStore (not None)."""
    runs = InMemoryRunStore()
    sessions = InMemorySessionStore()
    events = InMemoryEventStore()
    checkpoints = InMemoryCheckpointStore()
    approvals = InMemoryApprovalStore()
    idempotency = InMemoryIdempotencyStore()
    run_definitions = InMemoryRunDefinitionStore()
    swarms = InMemorySwarmStore()
    memories = InMemoryMemoryStore()
    coordination = InMemoryLeaseCoordinator()
    blob_store = InMemoryArtifactBlobStore()
    record_store = InMemoryArtifactRecordStore()
    artifacts = ArtifactStore(blob_store, record_store)
    assets = AssetStore(primary=MemoryAssetBackend())
    tasks = InMemoryJobStore()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return InMemoryExternalStorage(
        root=root_path,
        assets=assets,
        sessions=sessions,
        runs=runs,
        events=events,
        checkpoints=checkpoints,
        swarms=swarms,
        memories=memories,
        approvals=approvals,
        idempotency=idempotency,
        features=FILE_STORAGE_FEATURES,
        coordination=coordination,
        transactions=NoCrossStoreTransactions("InMemoryExternalStorage"),
        run_definitions=run_definitions,
        artifacts=artifacts,
        tasks=tasks,
    )


__all__: "list[str]" = [
    "InMemoryApprovalStore",
    "InMemoryCheckpointStore",
    "InMemoryEventStore",
    "InMemoryExternalStorage",
    "InMemoryIdempotencyStore",
    "InMemoryJobStore",
    "InMemoryMemoryStore",
    "InMemoryRunDefinitionStore",
    "InMemoryRunStore",
    "InMemorySessionStore",
    "InMemorySwarmStore",
    "build_in_memory_external_storage",
]
