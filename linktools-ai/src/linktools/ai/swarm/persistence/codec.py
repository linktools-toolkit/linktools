"""Strict canonical wire codec shared by Filesystem and SQL Swarm commits."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...events.context import EventStreamContext
from ...events.payloads import (
    SwarmCancelled,
    SwarmCompleted,
    SwarmFailed,
    SwarmStarted,
    SwarmStepCompleted,
    SwarmStepCreated,
    SwarmStepFailed,
)
from ...json import canonical_json
from ...run.models import RunErrorInfo, RunResult
from ...run.persistence.wire import JsonValue
from ..commit import (
    CancelSwarmCommand,
    CancelSwarmPayload,
    CompleteSwarmCommand,
    CompleteSwarmPayload,
    CompleteSwarmStepCommand,
    CompleteSwarmStepPayload,
    FailSwarmCommand,
    FailSwarmPayload,
    FailSwarmStepCommand,
    FailSwarmStepPayload,
    StartSwarmCommand,
    StartSwarmPayload,
    StartSwarmStepCommand,
    StartSwarmStepPayload,
    SwarmCommitId,
    SwarmExecutionFence,
)
from ..models import (
    SwarmRun,
    SwarmStatus,
    SwarmStep,
    SwarmStepStatus,
    TaskInput,
    TokenUsage,
)

SwarmCommitCommand = (
    StartSwarmCommand
    | StartSwarmStepCommand
    | CompleteSwarmStepCommand
    | FailSwarmStepCommand
    | CompleteSwarmCommand
    | FailSwarmCommand
    | CancelSwarmCommand
)
SwarmCommitResult = dict[str, str | int]
NonEmpty = Annotated[str, Field(min_length=1)]
NonNegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EventContextWire(_Wire):
    stream_id: NonEmpty
    run_id: NonEmpty
    root_run_id: NonEmpty
    parent_run_id: str | None
    session_id: NonEmpty
    runnable_id: NonEmpty


class TokenUsageWire(_Wire):
    input_tokens: NonNegative
    output_tokens: NonNegative
    total_cost: str


class RunResultWire(_Wire):
    output: JsonValue
    token_usage: dict[str, JsonValue]
    metadata: dict[str, JsonValue]


class RunErrorWire(_Wire):
    error_type: NonEmpty
    message: str
    detail: dict[str, JsonValue]


class TaskInputWire(_Wire):
    prompt: str
    metadata: dict[str, JsonValue]


class SwarmRunWire(_Wire):
    id: NonEmpty
    run_id: NonEmpty
    round: NonNegative
    status: str
    version: Positive
    token_usage: TokenUsageWire
    cost: str
    created_at: str
    updated_at: str
    metadata: dict[str, JsonValue]
    execution_token: str | None
    execution_owner_id: str | None
    execution_generation: NonNegative


class SwarmStepWire(_Wire):
    id: NonEmpty
    swarm_run_id: NonEmpty
    parent_task_id: str | None
    assigned_agent_id: str | None
    description: str
    status: str
    dependencies: list[str]
    input: TaskInputWire
    result: RunResultWire | None
    error: RunErrorWire | None
    attempts: NonNegative
    version: Positive
    claimed_at: str | None
    lease_expires_at: str | None
    created_at: str
    updated_at: str
    active_run_id: str | None


class SwarmStartedWire(_Wire):
    swarm_run_id: NonEmpty
    swarm_id: NonEmpty


class SwarmStepCreatedWire(_Wire):
    swarm_run_id: NonEmpty
    task_id: NonEmpty
    description: str


class SwarmStepCompletedWire(_Wire):
    swarm_run_id: NonEmpty
    task_id: NonEmpty


class SwarmStepFailedWire(_Wire):
    swarm_run_id: NonEmpty
    task_id: NonEmpty
    error_message: str


class SwarmCompletedWire(_Wire):
    swarm_run_id: NonEmpty


class SwarmFailedWire(_Wire):
    swarm_run_id: NonEmpty
    error: str


class SwarmCancelledWire(_Wire):
    swarm_run_id: NonEmpty


class _RequestWire(_Wire):
    commit_id: NonEmpty
    swarm_run_id: NonEmpty
    expected_version: NonNegative
    fence: NonEmpty


class StartSwarmRequestWire(_RequestWire):
    run: SwarmRunWire
    started_event: SwarmStartedWire
    event_context: EventContextWire


class StartSwarmStepRequestWire(_RequestWire):
    step_attempt_id: NonEmpty
    step: SwarmStepWire
    step_event: SwarmStepCreatedWire
    event_context: EventContextWire


class CompleteSwarmStepRequestWire(_RequestWire):
    step_attempt_id: NonEmpty
    task_id: NonEmpty
    result: RunResultWire
    active_run_id: str | None
    completed_event: SwarmStepCompletedWire
    event_context: EventContextWire


class FailSwarmStepRequestWire(_RequestWire):
    step_attempt_id: NonEmpty
    task_id: NonEmpty
    error: RunErrorWire
    active_run_id: str | None
    failed_event: SwarmStepFailedWire
    event_context: EventContextWire


class CompleteSwarmRequestWire(_RequestWire):
    result: RunResultWire
    token_usage: TokenUsageWire
    completed_event: SwarmCompletedWire
    event_context: EventContextWire


class FailSwarmRequestWire(_RequestWire):
    error: RunErrorWire
    token_usage: TokenUsageWire
    failed_event: SwarmFailedWire
    event_context: EventContextWire


class CancelSwarmRequestWire(_RequestWire):
    token_usage: TokenUsageWire
    cancelled_event: SwarmCancelledWire
    event_context: EventContextWire


class SwarmRunCommitResultWire(_Wire):
    swarm_run_id: NonEmpty
    version: Positive


class SwarmStepCommitResultWire(_Wire):
    task_id: NonEmpty
    version: Positive


class _Envelope(_Wire):
    schema_version: Literal[1]
    operation: str
    payload: dict[str, JsonValue]


_COMMAND_TYPES = {
    "start": StartSwarmCommand,
    "start_step": StartSwarmStepCommand,
    "complete_step": CompleteSwarmStepCommand,
    "fail_step": FailSwarmStepCommand,
    "complete": CompleteSwarmCommand,
    "fail": FailSwarmCommand,
    "cancel": CancelSwarmCommand,
}
_REQUEST_MODELS = {
    "start": StartSwarmRequestWire,
    "start_step": StartSwarmStepRequestWire,
    "complete_step": CompleteSwarmStepRequestWire,
    "fail_step": FailSwarmStepRequestWire,
    "complete": CompleteSwarmRequestWire,
    "fail": FailSwarmRequestWire,
    "cancel": CancelSwarmRequestWire,
}
_RESULT_MODELS = {
    "start": SwarmRunCommitResultWire,
    "start_step": SwarmStepCommitResultWire,
    "complete_step": SwarmStepCommitResultWire,
    "fail_step": SwarmStepCommitResultWire,
    "complete": SwarmRunCommitResultWire,
    "fail": SwarmRunCommitResultWire,
    "cancel": SwarmRunCommitResultWire,
}


def _ctx(value: EventStreamContext) -> EventContextWire:
    return EventContextWire(**{
        name: getattr(value, name)
        for name in (
            "stream_id", "run_id", "root_run_id", "parent_run_id",
            "session_id", "runnable_id",
        )
    })


def _ctx_from(value: EventContextWire) -> EventStreamContext:
    return EventStreamContext(**value.model_dump())


def _result(value: RunResult) -> RunResultWire:
    return RunResultWire(
        output=value.output,
        token_usage=dict(value.token_usage),
        metadata=dict(value.metadata),
    )


def _result_from(value: RunResultWire) -> RunResult:
    return RunResult(**value.model_dump())


def _error(value: RunErrorInfo) -> RunErrorWire:
    return RunErrorWire(
        error_type=value.error_type,
        message=value.message,
        detail=dict(value.detail),
    )


def _error_from(value: RunErrorWire) -> RunErrorInfo:
    return RunErrorInfo(**value.model_dump())


def _run(value: SwarmRun) -> SwarmRunWire:
    return SwarmRunWire(
        id=value.id,
        run_id=value.run_id,
        round=value.round,
        status=value.status.value,
        version=value.version,
        token_usage=TokenUsageWire(
            input_tokens=value.token_usage.input_tokens,
            output_tokens=value.token_usage.output_tokens,
            total_cost=str(value.token_usage.total_cost),
        ),
        cost=str(value.cost),
        created_at=value.created_at.isoformat(),
        updated_at=value.updated_at.isoformat(),
        metadata=dict(value.metadata),
        execution_token=value.execution_token,
        execution_owner_id=value.execution_owner_id,
        execution_generation=value.execution_generation,
    )


def _run_from(value: SwarmRunWire) -> SwarmRun:
    return SwarmRun(
        id=value.id,
        run_id=value.run_id,
        round=value.round,
        status=SwarmStatus(value.status),
        version=value.version,
        token_usage=TokenUsage(
            input_tokens=value.token_usage.input_tokens,
            output_tokens=value.token_usage.output_tokens,
            total_cost=Decimal(value.token_usage.total_cost),
        ),
        cost=Decimal(value.cost),
        created_at=datetime.fromisoformat(value.created_at),
        updated_at=datetime.fromisoformat(value.updated_at),
        metadata=value.metadata,
        execution_token=value.execution_token,
        execution_owner_id=value.execution_owner_id,
        execution_generation=value.execution_generation,
    )


def _step(value: SwarmStep) -> SwarmStepWire:
    return SwarmStepWire(
        id=value.id,
        swarm_run_id=value.swarm_run_id,
        parent_task_id=value.parent_task_id,
        assigned_agent_id=value.assigned_agent_id,
        description=value.description,
        status=value.status.value,
        dependencies=list(value.dependencies),
        input=TaskInputWire(
            prompt=value.input.prompt,
            metadata=dict(value.input.metadata),
        ),
        result=_result(value.result) if value.result is not None else None,
        error=_error(value.error) if value.error is not None else None,
        attempts=value.attempts,
        version=value.version,
        claimed_at=value.claimed_at.isoformat() if value.claimed_at else None,
        lease_expires_at=value.lease_expires_at.isoformat() if value.lease_expires_at else None,
        created_at=value.created_at.isoformat(),
        updated_at=value.updated_at.isoformat(),
        active_run_id=value.active_run_id,
    )


def _step_from(value: SwarmStepWire) -> SwarmStep:
    return SwarmStep(
        id=value.id,
        swarm_run_id=value.swarm_run_id,
        parent_task_id=value.parent_task_id,
        assigned_agent_id=value.assigned_agent_id,
        description=value.description,
        status=SwarmStepStatus(value.status),
        dependencies=tuple(value.dependencies),
        input=TaskInput(prompt=value.input.prompt, metadata=value.input.metadata),
        result=_result_from(value.result) if value.result else None,
        error=_error_from(value.error) if value.error else None,
        attempts=value.attempts,
        version=value.version,
        claimed_at=datetime.fromisoformat(value.claimed_at) if value.claimed_at else None,
        lease_expires_at=(
            datetime.fromisoformat(value.lease_expires_at)
            if value.lease_expires_at else None
        ),
        created_at=datetime.fromisoformat(value.created_at),
        updated_at=datetime.fromisoformat(value.updated_at),
        active_run_id=value.active_run_id,
    )


def _common(command: SwarmCommitCommand) -> dict[str, object]:
    return {
        "commit_id": command.commit_id.value,
        "swarm_run_id": command.swarm_run_id,
        "expected_version": command.expected_version,
        "fence": command.fence.token,
    }


def _usage(value: TokenUsage) -> TokenUsageWire:
    return TokenUsageWire(
        input_tokens=value.input_tokens,
        output_tokens=value.output_tokens,
        total_cost=str(value.total_cost),
    )


def _usage_from(value: TokenUsageWire) -> TokenUsage:
    return TokenUsage(
        input_tokens=value.input_tokens,
        output_tokens=value.output_tokens,
        total_cost=Decimal(value.total_cost),
    )


def _request_wire(operation: str, command: SwarmCommitCommand) -> _Wire:
    expected = _COMMAND_TYPES.get(operation)
    if expected is None or not isinstance(command, expected):
        name = expected.__name__ if expected else "known Swarm command"
        raise ValueError(
            f"{operation!r} requires {name}, got {type(command).__name__}"
        )
    common = _common(command)
    payload = command.payload
    if operation == "start":
        return StartSwarmRequestWire(
            **common,
            run=_run(payload.run),
            started_event=SwarmStartedWire(
                swarm_run_id=payload.started_event.swarm_run_id,
                swarm_id=payload.started_event.swarm_id,
            ),
            event_context=_ctx(payload.event_context),
        )
    if operation == "start_step":
        return StartSwarmStepRequestWire(
            **common,
            step_attempt_id=command.step_attempt_id,
            step=_step(payload.step),
            step_event=SwarmStepCreatedWire(**{
                name: getattr(payload.step_event, name)
                for name in ("swarm_run_id", "task_id", "description")
            }),
            event_context=_ctx(payload.event_context),
        )
    if operation == "complete_step":
        return CompleteSwarmStepRequestWire(
            **common,
            step_attempt_id=command.step_attempt_id,
            task_id=payload.task_id,
            result=_result(payload.result),
            active_run_id=payload.active_run_id,
            completed_event=SwarmStepCompletedWire(**{
                name: getattr(payload.completed_event, name)
                for name in ("swarm_run_id", "task_id")
            }),
            event_context=_ctx(payload.event_context),
        )
    if operation == "fail_step":
        return FailSwarmStepRequestWire(
            **common,
            step_attempt_id=command.step_attempt_id,
            task_id=payload.task_id,
            error=_error(payload.error),
            active_run_id=payload.active_run_id,
            failed_event=SwarmStepFailedWire(**{
                name: getattr(payload.failed_event, name)
                for name in ("swarm_run_id", "task_id", "error_message")
            }),
            event_context=_ctx(payload.event_context),
        )
    if operation == "complete":
        return CompleteSwarmRequestWire(
            **common,
            result=_result(payload.result),
            token_usage=_usage(payload.token_usage),
            completed_event=SwarmCompletedWire(
                swarm_run_id=payload.completed_event.swarm_run_id
            ),
            event_context=_ctx(payload.event_context),
        )
    if operation == "fail":
        return FailSwarmRequestWire(
            **common,
            error=_error(payload.error),
            token_usage=_usage(payload.token_usage),
            failed_event=SwarmFailedWire(
                swarm_run_id=payload.failed_event.swarm_run_id,
                error=payload.failed_event.error,
            ),
            event_context=_ctx(payload.event_context),
        )
    return CancelSwarmRequestWire(
        **common,
        token_usage=_usage(payload.token_usage),
        cancelled_event=SwarmCancelledWire(
            swarm_run_id=payload.cancelled_event.swarm_run_id
        ),
        event_context=_ctx(payload.event_context),
    )


def _command(operation: str, wire: _Wire) -> SwarmCommitCommand:
    common = {
        "commit_id": SwarmCommitId(wire.commit_id),
        "swarm_run_id": wire.swarm_run_id,
        "expected_version": wire.expected_version,
        "fence": SwarmExecutionFence(wire.fence),
    }
    if operation == "start":
        assert isinstance(wire, StartSwarmRequestWire)
        payload = StartSwarmPayload(
            run=_run_from(wire.run),
            started_event=SwarmStarted(**wire.started_event.model_dump()),
            event_context=_ctx_from(wire.event_context),
        )
        return StartSwarmCommand(payload=payload, **common)
    if operation == "start_step":
        assert isinstance(wire, StartSwarmStepRequestWire)
        payload = StartSwarmStepPayload(
            step=_step_from(wire.step),
            step_event=SwarmStepCreated(**wire.step_event.model_dump()),
            event_context=_ctx_from(wire.event_context),
        )
        return StartSwarmStepCommand(
            step_attempt_id=wire.step_attempt_id, payload=payload, **common
        )
    if operation == "complete_step":
        assert isinstance(wire, CompleteSwarmStepRequestWire)
        payload = CompleteSwarmStepPayload(
            task_id=wire.task_id,
            result=_result_from(wire.result),
            active_run_id=wire.active_run_id,
            completed_event=SwarmStepCompleted(**wire.completed_event.model_dump()),
            event_context=_ctx_from(wire.event_context),
        )
        return CompleteSwarmStepCommand(
            step_attempt_id=wire.step_attempt_id, payload=payload, **common
        )
    if operation == "fail_step":
        assert isinstance(wire, FailSwarmStepRequestWire)
        payload = FailSwarmStepPayload(
            task_id=wire.task_id,
            error=_error_from(wire.error),
            active_run_id=wire.active_run_id,
            failed_event=SwarmStepFailed(**wire.failed_event.model_dump()),
            event_context=_ctx_from(wire.event_context),
        )
        return FailSwarmStepCommand(
            step_attempt_id=wire.step_attempt_id, payload=payload, **common
        )
    if operation == "complete":
        assert isinstance(wire, CompleteSwarmRequestWire)
        return CompleteSwarmCommand(
            payload=CompleteSwarmPayload(
                result=_result_from(wire.result),
                token_usage=_usage_from(wire.token_usage),
                completed_event=SwarmCompleted(**wire.completed_event.model_dump()),
                event_context=_ctx_from(wire.event_context),
            ),
            **common,
        )
    if operation == "fail":
        assert isinstance(wire, FailSwarmRequestWire)
        return FailSwarmCommand(
            payload=FailSwarmPayload(
                error=_error_from(wire.error),
                token_usage=_usage_from(wire.token_usage),
                failed_event=SwarmFailed(**wire.failed_event.model_dump()),
                event_context=_ctx_from(wire.event_context),
            ),
            **common,
        )
    assert isinstance(wire, CancelSwarmRequestWire)
    return CancelSwarmCommand(
        payload=CancelSwarmPayload(
            token_usage=_usage_from(wire.token_usage),
            cancelled_event=SwarmCancelled(**wire.cancelled_event.model_dump()),
            event_context=_ctx_from(wire.event_context),
        ),
        **common,
    )


class SwarmCommitCodec:
    schema_version = 1

    def encode_request(self, operation: str, command: SwarmCommitCommand) -> bytes:
        wire = _request_wire(operation, command)
        return canonical_json({
            "schema_version": self.schema_version,
            "operation": operation,
            "payload": wire.model_dump(mode="json"),
        }).encode()

    def decode_request(self, operation: str, payload: bytes) -> SwarmCommitCommand:
        try:
            envelope = _Envelope.model_validate_json(payload)
            if envelope.operation != operation:
                raise ValueError("Swarm request operation mismatch")
            model = _REQUEST_MODELS.get(operation)
            if model is None:
                raise ValueError(f"unknown Swarm operation {operation!r}")
            return _command(operation, model.model_validate(envelope.payload))
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Swarm request payload") from exc

    def encode_result(self, operation: str, result: SwarmCommitResult) -> bytes:
        model = _RESULT_MODELS.get(operation)
        if model is None:
            raise ValueError(f"unknown Swarm operation {operation!r}")
        try:
            wire = model.model_validate(result)
        except ValidationError as exc:
            raise ValueError("invalid Swarm result") from exc
        return canonical_json({
            "schema_version": self.schema_version,
            "operation": operation,
            "payload": wire.model_dump(mode="json"),
        }).encode()

    def decode_result(self, operation: str, payload: bytes) -> SwarmCommitResult:
        try:
            envelope = _Envelope.model_validate_json(payload)
            if envelope.operation != operation:
                raise ValueError("Swarm result operation mismatch")
            model = _RESULT_MODELS.get(operation)
            if model is None:
                raise ValueError(f"unknown Swarm operation {operation!r}")
            return model.model_validate(envelope.payload).model_dump()
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Swarm result payload") from exc

    def request_hash(self, operation: str, command: SwarmCommitCommand) -> bytes:
        return sha256(self.encode_request(operation, command)).digest()


__all__ = [
    "CancelSwarmRequestWire",
    "CompleteSwarmRequestWire",
    "CompleteSwarmStepRequestWire",
    "FailSwarmRequestWire",
    "FailSwarmStepRequestWire",
    "StartSwarmRequestWire",
    "StartSwarmStepRequestWire",
    "SwarmCommitCodec",
    "SwarmCommitCommand",
    "SwarmCommitResult",
    "SwarmRunCommitResultWire",
    "SwarmStepCommitResultWire",
]
