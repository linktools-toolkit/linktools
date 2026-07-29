#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunCommitCoordinator: the protocol for atomically committing the cross-store
state transitions each Run lifecycle point requires.

The SQL backend implements this via a shared UnitOfWork; the File backend via
sequential writes + fail-closed propagation. Every operation is idempotent by
``commit_id``: a retried call with the SAME commit_id returns the original
result instead of re-applying the write; a retried call with the SAME
commit_id but a DIFFERENT payload is a conflict. ``start``/``pause``/
``complete``/``fail``/``acknowledge_cancel`` each bundle the run's state
transition with its state-critical event in the SAME commit -- neither can
succeed without the other."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from ..events.payloads import (
    RunCancelled as RunCancelledEvent,
    RunCompleted as RunCompletedEvent,
    RunFailed as RunFailedEvent,
    RunPaused as RunPausedEvent,
    RunResumed as RunResumedEvent,
    RunStarted as RunStartedEvent,
)


class RunFenceRequiredError(Exception):
    pass


class RunFenceStateError(Exception):
    pass


class RunFenceLostError(Exception):
    pass


class RunFenceConfigurationError(Exception):
    pass

if TYPE_CHECKING:
    from typing import Any

    from ..events.context import EventStreamContext
    from .session import NewSessionMessage
    from .run import RunErrorInfo, RunRecord, RunResult


@dataclass(frozen=True, slots=True)
class RunCommitId:
    """A Run commit's deterministic id. Caller-supplied (e.g.
    ``start:{run_id}``); the coordinator recognizes an already-committed id
    and returns the original result instead of re-applying the write, while
    the same id with a DIFFERENT payload is a RunCommitConflictError. The
    value is bounded 1..200 chars so it fits a SQL unique column without
    truncation and so an empty string cannot slip through (the old default
    that defeated idempotency)."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or len(self.value) > 200:
            raise ValueError(
                "run commit id must be 1..200 characters; an empty id defeats "
                "the idempotent replay the commit log depends on"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ExecutionFence:
    """The claiming execution's fencing token, paired with the commit so the
    coordinator can refuse a stale execution's write. ``RunCommitPolicy``
    decides whether a fence is REQUIRED for a given topology; when it is, the
    command MUST carry an ExecutionFence, the stored RunRecord MUST carry a
    token, and the two MUST be equal. An empty token is forbidden: the old
    code allowed the empty-string default to bypass the check, which is the
    bug strict fencing closes."""

    token: str

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError(
                "execution fence token cannot be empty; pass a real claiming-"
                "execution token or change RunCommitPolicy(fencing_required="
                "False) and pass no fence at all"
            )

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True, slots=True)
class RunCommitPolicy:
    """Per-topology fencing policy. ``fencing_required=True`` for a multi-
    worker deployment (a stale worker MUST NOT commit after losing its lease);
    ``False`` for a single-process reference deployment. Constructed by the
    Runtime from the verified topology + StorageFeatures."""

    fencing_required: bool

    def validate(self, *, supplied: "ExecutionFence | None", stored_token: str | None) -> None:
        if self.fencing_required:
            if supplied is None:
                raise RunFenceRequiredError("execution fence is required")
            if not stored_token:
                raise RunFenceStateError("stored execution fence is missing")
            if supplied.token != stored_token:
                raise RunFenceLostError("execution fence does not match stored owner")
        elif supplied is not None or stored_token:
            raise RunFenceConfigurationError("fence supplied while fencing is disabled")


@dataclass(frozen=True, slots=True)
class ApprovalRequestData:
    approval_id: str
    tool_name: str
    reason: str
    arguments: Mapping[str, object]
    tenant_id: str
    tool_call_id: str | None
    binding: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StartRunCommand:
    record: "RunRecord"
    started_event: RunStartedEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars). The coordinator
    # recognizes an already-committed start and returns the existing record
    # instead of double-creating it; same id + different payload conflicts.
    commit_id: RunCommitId


@dataclass(frozen=True, slots=True)
class PauseRunCommand:
    run_id: str
    expected_version: int
    approval_request: ApprovalRequestData
    checkpoint_payload: bytes
    # The caller-built RunPaused event, carrying the ACTUAL pause reason
    # (from approval_request.reason) -- the coordinator appends this
    # verbatim rather than re-deriving a generic message from the approval
    # id, so the persisted event keeps full fidelity to why the run paused.
    paused_event: RunPausedEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId
    # Optional execution fence. When the topology requires fencing
    # (RunCommitPolicy.fencing_required=True), the command MUST carry a
    # fence, the stored RunRecord MUST carry a token, and the two MUST be
    # equal. None on either side is enforced as a fencing-policy violation
    # rather than silently bypassing the check (the empty-string-bypass bug
    # strict fencing closes).
    execution_fence: "ExecutionFence | None" = None
    # No session messages are appended on pause today (a turn isn't complete
    # yet) -- always empty in practice, but carried as a real field so a
    # future caller that DOES have partial-turn messages to record can
    # supply them without another signature change.
    messages: "tuple[NewSessionMessage, ...]" = ()


@dataclass(frozen=True, slots=True)
class ResumeRunCommand:
    run_id: str
    expected_version: int
    approval_id: str
    resumed_event: RunResumedEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId


@dataclass(frozen=True, slots=True)
class CompleteRunCommand:
    run_id: str
    session_id: str
    expected_version: int
    messages: "tuple[NewSessionMessage, ...]"
    checkpoint_payload: bytes
    result: "RunResult"
    # The caller-built RunCompleted event -- the coordinator appends this
    # verbatim instead of constructing its own, matching PauseRunCommand's
    # paused_event field.
    completed_event: RunCompletedEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId
    # See PauseRunCommand.execution_fence -- same fencing semantics.
    execution_fence: "ExecutionFence | None" = None


@dataclass(frozen=True, slots=True)
class FailRunCommand:
    run_id: str
    expected_version: int
    error: "RunErrorInfo"
    failed_event: RunFailedEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId
    execution_fence: "ExecutionFence | None" = None


@dataclass(frozen=True, slots=True)
class RequestCancelRunCommand:
    run_id: str
    expected_version: int
    requested_by: str
    reason: "str | None"
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId


@dataclass(frozen=True, slots=True)
class AcknowledgeCancelRunCommand:
    run_id: str
    expected_version: int
    cancelled_event: RunCancelledEvent
    event_context: "EventStreamContext"
    # Required typed commit id (RunCommitId, 1..200 chars).
    commit_id: RunCommitId
    execution_fence: "ExecutionFence | None" = None


@dataclass(frozen=True, slots=True)
class StartedRunCommit:
    record: "RunRecord"


@dataclass(frozen=True, slots=True)
class PausedRunCommit:
    approval_id: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class ResumedRunCommit:
    run_id: str


@dataclass(frozen=True, slots=True)
class CompletedRunCommit:
    result: "RunResult"


@dataclass(frozen=True, slots=True)
class FailedRunCommit:
    run_id: str


@dataclass(frozen=True, slots=True)
class CancellingRunCommit:
    run_id: str


@dataclass(frozen=True, slots=True)
class CancelledRunCommit:
    run_id: str


@runtime_checkable
class RunCommitCoordinator(Protocol):
    async def start(self, command: StartRunCommand) -> StartedRunCommit: ...

    async def pause(self, command: PauseRunCommand) -> PausedRunCommit: ...

    async def resume(self, command: ResumeRunCommand) -> ResumedRunCommit: ...

    async def complete(self, command: CompleteRunCommand) -> CompletedRunCommit: ...

    async def fail(self, command: FailRunCommand) -> FailedRunCommit: ...

    async def request_cancel(
        self, command: RequestCancelRunCommand
    ) -> CancellingRunCommit: ...

    async def acknowledge_cancel(
        self, command: AcknowledgeCancelRunCommand
    ) -> CancelledRunCommit: ...

    async def recover_incomplete_commits(self) -> None:
        """Finish or fail every in-flight commit left by a crash, BEFORE the
        Runtime accepts any new request. A coordinator whose backend has no
        crash-window (SQL: every commit is one atomic transaction) implements
        this as a no-op; a coordinator whose backend can crash mid-commit
        (Filesystem: sequential writes through a journal) implements real
        recovery. If recovery cannot complete, the Runtime build/start MUST
        surface that as a failure rather than silently accepting requests
        against an inconsistent state."""
        ...
