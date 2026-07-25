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
from typing import Any, Mapping, Protocol, runtime_checkable

from ..events.payloads import (
    RunCancelled as RunCancelledEvent,
    RunCompleted as RunCompletedEvent,
    RunFailed as RunFailedEvent,
    RunPaused as RunPausedEvent,
    RunResumed as RunResumedEvent,
    RunStarted as RunStartedEvent,
)


@dataclass(frozen=True, slots=True)
class StartRunCommand:
    record: Any
    started_event: RunStartedEvent
    event_context: Any
    # Deterministic id (caller sets e.g. ``start:{run_id}``) so a retried
    # start is idempotent -- the coordinator recognizes an already-committed
    # start and returns the existing record instead of double-creating it.
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class PauseRunCommand:
    run_id: str
    expected_version: int
    approval_request: Mapping[str, Any]
    checkpoint_payload: bytes
    # The caller-built RunPaused event, carrying the ACTUAL pause reason
    # (from approval_request["reason"]) -- the coordinator appends this
    # verbatim rather than re-deriving a generic message from the approval
    # id, so the persisted event keeps full fidelity to why the run paused.
    paused_event: RunPausedEvent
    event_context: Any
    # No session messages are appended on pause today (a turn isn't complete
    # yet) -- always empty in practice, but carried as a real field so a
    # future caller that DOES have partial-turn messages to record can
    # supply them without another signature change.
    messages: "tuple[Any, ...]" = ()
    # Deterministic id (caller sets e.g. ``pause:{run_id}:{approval_id}``) so a
    # retried pause is idempotent -- the coordinator recognizes an already-
    # committed pause and writes nothing instead of duplicating artifacts.
    commit_id: str = ""
    # The claiming execution's fencing token (empty when the backend does not
    # support claim_execution). Enforced by both commit implementations via
    # _check_execution_token; empty on either side skips the check.
    execution_token: str = ""


@dataclass(frozen=True, slots=True)
class ResumeRunCommand:
    run_id: str
    expected_version: int
    approval_id: str
    resumed_event: RunResumedEvent
    event_context: Any
    # Deterministic id (caller sets e.g. ``resume:{run_id}:{approval_id}``).
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class CompleteRunCommand:
    run_id: str
    session_id: str
    expected_version: int
    messages: "tuple[Any, ...]"
    checkpoint_payload: bytes
    result: Any
    # The caller-built RunCompleted event -- the coordinator appends this
    # verbatim instead of constructing its own, matching PauseRunCommand's
    # paused_event field.
    completed_event: RunCompletedEvent
    event_context: Any
    # Deterministic id (caller sets e.g. ``complete:{run_id}:{expected_version}``)
    # so a retried complete is idempotent.
    commit_id: str = ""
    # See PauseRunCommand.execution_token -- same fencing token, enforced by
    # both commit implementations.
    execution_token: str = ""


@dataclass(frozen=True, slots=True)
class FailRunCommand:
    run_id: str
    expected_version: int
    execution_token: str
    error: Any
    failed_event: RunFailedEvent
    event_context: Any
    # Deterministic id (caller sets e.g. ``fail:{run_id}:{expected_version}``).
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class RequestCancelRunCommand:
    run_id: str
    expected_version: int
    requested_by: str
    reason: "str | None"
    event_context: Any
    # Deterministic id (caller sets e.g. ``request-cancel:{run_id}``).
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class AcknowledgeCancelRunCommand:
    run_id: str
    expected_version: int
    execution_token: str
    cancelled_event: RunCancelledEvent
    event_context: Any
    # Deterministic id (caller sets e.g. ``ack-cancel:{run_id}``).
    commit_id: str = ""


@dataclass(frozen=True, slots=True)
class StartedRunCommit:
    record: Any


@dataclass(frozen=True, slots=True)
class PausedRunCommit:
    approval_id: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class ResumedRunCommit:
    run_id: str


@dataclass(frozen=True, slots=True)
class CompletedRunCommit:
    result: Any


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
