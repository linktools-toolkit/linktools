#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm commit boundary (P7 swarm commit boundary).

The SwarmCommitCoordinator owns the cross-store commit for every swarm +
swarm-step lifecycle point: state transition, state-critical event append,
and (for terminal commits) the commit-log entry that makes the operation
idempotent by commit_id. SwarmEngine receives the coordinator (along with
AgentCompiler, RunDispatcher, and Clock) and never touches a
RunStore / SessionStore / EventStore / RunDefinitionStore / append_event /
mark_completed / mark_failed directly -- the dep-cleanup rule.

The Protocol is the swarm-side mirror of RunCommitCoordinator. Each command
carries its deterministic commit_id, the swarm_run_id it targets, the
expected version for optimistic concurrency, an execution fence, and an
operation-specific typed payload containing the lifecycle event context. Same
commit_id + same payload = idempotent replay;
same commit_id + different payload = SwarmCommitConflictError.

The reference SQL implementation groups swarm-state, step-state, event, and
commit-log writes inside one Storage UoW so a swarm commit is all-or-nothing.
The Filesystem implementation journals the same shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from ..events.context import EventStreamContext

if TYPE_CHECKING:
    from ..events.payloads import (
        SwarmCancelled, SwarmCompleted, SwarmFailed, SwarmStarted,
        SwarmStepCompleted, SwarmStepCreated, SwarmStepFailed,
    )
    from ..run.models import RunErrorInfo, RunResult
    from .models import SwarmRun, SwarmStatus, SwarmStep
    from .store import SwarmStore


@dataclass(frozen=True, slots=True)
class SwarmCommitId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("swarm commit id cannot be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SwarmExecutionFence:
    token: str

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("swarm execution fence cannot be empty")


@dataclass(frozen=True, slots=True)
class SwarmCommitPolicy:
    fencing_required: bool = True

    def validate(
        self,
        *,
        supplied: SwarmExecutionFence | None,
        stored_token: str | None,
    ) -> None:
        """Validate the supplied fence against the persisted
        ``SwarmRun.execution_token`` (the run-level owner).

        fencing_required=True (the multi-worker default): a commit MUST carry
        a fence whose token EQUALS the persisted execution_token. A missing
        fence is ``SwarmFenceRequiredError``; a run with no persisted token is
        ``SwarmFenceStateError`` (the run was never claimed/started under
        fencing); a mismatched token is ``SwarmFenceLostError`` (a later
        reclaim owns this run now).

        fencing_required=False: a fence must NOT be supplied and the run must
        have no stored token; otherwise ``SwarmFenceConfigurationError``."""
        if self.fencing_required:
            if supplied is None:
                raise SwarmFenceRequiredError(
                    "swarm execution fence is required for this commit"
                )
            if not stored_token:
                raise SwarmFenceStateError(
                    "swarm run has no persisted execution token to fence against"
                )
            if supplied.token != stored_token:
                raise SwarmFenceLostError(
                    "swarm execution fence token does not match the persisted "
                    "owner; a later reclaim may own this run"
                )
            return
        if supplied is not None or stored_token:
            raise SwarmFenceConfigurationError(
                "swarm execution fencing is disabled for this deployment but a "
                "fence or stored token is present"
            )


class SwarmFenceRequiredError(Exception):
    """A commit that requires an execution fence was issued without one."""


class SwarmFenceStateError(Exception):
    """The persisted SwarmRun has no execution_token to fence against (e.g. it
    was created under a no-fencing deployment), so a fenced commit cannot
    prove ownership."""


class SwarmFenceLostError(Exception):
    """The supplied fence token does not match the persisted
    ``SwarmRun.execution_token`` -- a later reclaim/claim owns this run now,
    so the caller's commit is rejected rather than clobbering the current
    owner's progress."""


class SwarmFenceConfigurationError(Exception):
    """Fencing is disabled for this deployment but a fence (or a stored token)
    is present -- a deployment/configuration inconsistency."""


@dataclass(frozen=True, slots=True)
class StartSwarmPayload:
    run: "SwarmRun"
    started_event: "SwarmStarted"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class StartSwarmStepPayload:
    step: "SwarmStep"
    step_event: "SwarmStepCreated"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class CompleteSwarmStepPayload:
    task_id: str
    result: "RunResult"
    active_run_id: str | None
    completed_event: "SwarmStepCompleted"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class FailSwarmStepPayload:
    task_id: str
    error: "RunErrorInfo"
    active_run_id: str | None
    failed_event: "SwarmStepFailed"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class CompleteSwarmPayload:
    result: "RunResult"
    completed_event: "SwarmCompleted"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class FailSwarmPayload:
    error: "RunErrorInfo"
    failed_event: "SwarmFailed"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class CancelSwarmPayload:
    cancelled_event: "SwarmCancelled"
    event_context: EventStreamContext


@dataclass(frozen=True, slots=True)
class StartSwarmCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    expected_version: int
    payload: StartSwarmPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class StartSwarmStepCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: StartSwarmStepPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class CompleteSwarmStepCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: CompleteSwarmStepPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class FailSwarmStepCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    step_attempt_id: str
    expected_version: int
    payload: FailSwarmStepPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class CompleteSwarmCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    expected_version: int
    payload: CompleteSwarmPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class FailSwarmCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    expected_version: int
    payload: FailSwarmPayload
    fence: SwarmExecutionFence


@dataclass(frozen=True, slots=True)
class CancelSwarmCommand:
    commit_id: SwarmCommitId
    swarm_run_id: str
    expected_version: int
    payload: CancelSwarmPayload
    fence: SwarmExecutionFence


@runtime_checkable
class SwarmCommitCoordinator(Protocol):
    """Atomic cross-store commit for swarm + swarm-step lifecycle points.
    Each method runs the state transition + state-critical event append +
    commit-log entry in ONE Storage UoW so the commit is all-or-nothing,
    and is idempotent by commit_id (same id + same payload returns the
    recorded result; same id + different payload is a conflict)."""

    state_store: "SwarmStore"

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None": ...

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        token_usage: object | None = None,
    ) -> "SwarmRun": ...

    async def start(self, command: StartSwarmCommand) -> "SwarmCommitResult": ...
    async def start_step(self, command: StartSwarmStepCommand) -> "SwarmCommitResult": ...
    async def complete_step(self, command: CompleteSwarmStepCommand) -> "SwarmCommitResult": ...
    async def fail_step(self, command: FailSwarmStepCommand) -> "SwarmCommitResult": ...
    async def complete(self, command: CompleteSwarmCommand) -> "SwarmCommitResult": ...
    async def fail(self, command: FailSwarmCommand) -> "SwarmCommitResult": ...
    async def cancel(self, command: CancelSwarmCommand) -> "SwarmCommitResult": ...


class SwarmCommitResult(TypedDict, total=False):
    swarm_run_id: str
    task_id: str
    version: int


class SwarmCommitConflictError(Exception):
    """A retried swarm commit used the SAME commit_id with a DIFFERENT payload.
    Same-id-same-hash is an idempotent replay (returns the recorded result);
    same-id-different-hash is this conflict -- the caller is asserting two
    distinct operations under one id, which the commit log refuses to
    silently collapse."""


class SwarmCommitIntegrityError(Exception):
    """A commit would violate the durable swarm state machine."""


__all__: "list[str]" = [
    "CancelSwarmCommand",
    "CompleteSwarmCommand",
    "CompleteSwarmStepCommand",
    "FailSwarmCommand",
    "FailSwarmStepCommand",
    "StartSwarmCommand",
    "StartSwarmStepCommand",
    "SwarmCommitConflictError",
    "SwarmCommitIntegrityError",
    "SwarmCommitCoordinator",
    "SwarmCommitId",
    "SwarmCommitPolicy",
    "SwarmExecutionFence",
    "SwarmFenceConfigurationError",
    "SwarmFenceLostError",
    "SwarmFenceRequiredError",
    "SwarmFenceStateError",
    "CancelSwarmPayload",
    "CompleteSwarmPayload",
    "CompleteSwarmStepPayload",
    "FailSwarmPayload",
    "FailSwarmStepPayload",
    "StartSwarmPayload",
    "StartSwarmStepPayload",
    "SwarmCommitResult",
]
