"""The explicit, versioned Run commit wire protocol.

This module deliberately contains no object reflection.  A persisted Run
payload is a small JSON document whose shape is owned by this module, so a
restart does not need to import an arbitrary Python type to read it.
"""
from __future__ import annotations

import base64
import json
import math
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, TypeVar, Annotated
from typing import Literal
from typing_extensions import TypeAliasType

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ...events.context import EventStreamContext
from ...events.payloads import (
    RunCancelled, RunCompleted, RunFailed, RunPaused, RunResumed, RunStarted,
)
from ...session.models import MessageRole, NewSessionMessage
from ..commit import (
    AcknowledgeCancelRunCommand, ApprovalRequestData, CancelledRunCommit,
    CancellingRunCommit, CompleteRunCommand, CompletedRunCommit,
    ExecutionFence, FailRunCommand, FailedRunCommit, PauseRunCommand,
    PausedRunCommit, RequestCancelRunCommand, ResumedRunCommit,
    ResumeRunCommand, RunCommitId, StartRunCommand, StartedRunCommit,
)
from ..models import RunErrorInfo, RunInput, RunRecord, RunResult, RunStatus, RunnableType

SCHEMA_VERSION = 1


class RunCommitOperation(str, Enum):
    """The seven fixed Run commit operation tags."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    COMPLETE = "complete"
    FAIL = "fail"
    REQUEST_CANCEL = "request_cancel"
    ACKNOWLEDGE_CANCEL = "acknowledge_cancel"


class RunCommitCodecError(Exception):
    """The wire payload is malformed, unknown, or violates its schema."""


class RunCommitIntegrityError(Exception):
    """Required persisted replay evidence is missing or unreadable."""


JsonScalar = str | int | float | bool | None
JsonValue = TypeAliasType("JsonValue", JsonScalar | list["JsonValue"] | dict[str, "JsonValue"])
DecodedRunCommand = (StartRunCommand | PauseRunCommand | ResumeRunCommand |
                     CompleteRunCommand | FailRunCommand | RequestCancelRunCommand |
                     AcknowledgeCancelRunCommand)
DecodedRunResult = (StartedRunCommit | PausedRunCommit | ResumedRunCommit |
                    CompletedRunCommit | FailedRunCommit | CancellingRunCommit |
                    CancelledRunCommit)


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunCommitWireEnvelope(_WireModel):
    schema_version: Literal[1]
    operation: str
    kind: Literal["request", "result"]
    payload: dict[str, JsonValue]


class DateTimeWire(_WireModel):
    value: str


class RunInputWire(_WireModel):
    prompt: str
    metadata: dict[str, JsonValue]


class RunResultWire(_WireModel):
    output: JsonValue
    token_usage: dict[str, JsonValue]
    metadata: dict[str, JsonValue]


class RunErrorWire(_WireModel):
    error_type: str
    message: str
    detail: dict[str, JsonValue]


class NewSessionMessageWire(_WireModel):
    role: str
    content: JsonValue
    run_id: str | None
    metadata: dict[str, JsonValue]


class EventContextWire(_WireModel):
    stream_id: str
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    session_id: str
    runnable_id: str


class EventWire(_WireModel):
    event_type: str
    schema_version: int = Field(ge=1)
    payload: dict[str, JsonValue]


NonEmptyId = Annotated[str, Field(min_length=1, max_length=512)]
CommitIdValue = Annotated[str, Field(min_length=1, max_length=200)]
NonEmptyFence = Annotated[str, Field(min_length=1)]
NonNegativeVersion = Annotated[int, Field(ge=0)]


class RunStartedEventPayloadWire(_WireModel):
    run_id: NonEmptyId
    runnable_id: NonEmptyId


class RunPausedEventPayloadWire(_WireModel):
    run_id: NonEmptyId
    reason: str


class RunResumedEventPayloadWire(_WireModel):
    run_id: NonEmptyId


class RunCompletedEventPayloadWire(_WireModel):
    run_id: NonEmptyId
    result_summary: JsonValue


class RunFailedEventPayloadWire(_WireModel):
    run_id: NonEmptyId
    error_type: NonEmptyId
    message: str


class RunCancelledEventPayloadWire(_WireModel):
    run_id: NonEmptyId
    reason: str | None


class RunRecordWire(_WireModel):
    id: NonEmptyId
    root_run_id: NonEmptyId
    parent_run_id: str | None
    session_id: NonEmptyId
    runnable_id: NonEmptyId
    runnable_type: str
    status: str
    input: RunInputWire
    result: RunResultWire | None
    error: RunErrorWire | None
    version: NonNegativeVersion
    created_at: str
    started_at: str | None
    finished_at: str | None
    metadata: dict[str, JsonValue]
    cancel_requested_at: str | None
    cancel_requested_by: str | None
    cancel_reason: str | None
    worker_id: str | None
    execution_token: str | None
    heartbeat_at: str | None
    manifest_id: str | None
    resumability: str | None


class ApprovalRequestWire(_WireModel):
    approval_id: NonEmptyId
    tool_name: NonEmptyId
    reason: str
    arguments: dict[str, JsonValue]
    tenant_id: NonEmptyId
    tool_call_id: str | None
    binding: dict[str, JsonValue]


class _CheckpointWire(_WireModel):
    checkpoint_payload_b64: str

    @field_validator("checkpoint_payload_b64")
    @classmethod
    def validate_checkpoint_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid base64") from exc
        return value


def _json_message_sequence(value: Any) -> tuple[NewSessionMessageWire, ...]:
    if not isinstance(value, list):
        raise ValueError("messages must be a JSON array")
    return tuple(NewSessionMessageWire.model_validate(item) for item in value)


class StartRunRequestWire(_WireModel):
    record: RunRecordWire
    started_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue


class PauseRunRequestWire(_CheckpointWire):
    run_id: NonEmptyId
    expected_version: NonNegativeVersion
    approval_request: ApprovalRequestWire
    paused_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue
    execution_fence: NonEmptyFence | None
    messages: tuple[NewSessionMessageWire, ...]

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, value: Any) -> tuple[NewSessionMessageWire, ...]:
        return _json_message_sequence(value)


class ResumeRunRequestWire(_WireModel):
    run_id: NonEmptyId
    expected_version: NonNegativeVersion
    approval_id: NonEmptyId
    resumed_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue


class CompleteRunRequestWire(_CheckpointWire):
    run_id: NonEmptyId
    session_id: NonEmptyId
    expected_version: NonNegativeVersion
    messages: tuple[NewSessionMessageWire, ...]
    result: RunResultWire
    completed_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue
    execution_fence: NonEmptyFence | None

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages(cls, value: Any) -> tuple[NewSessionMessageWire, ...]:
        return _json_message_sequence(value)


class FailRunRequestWire(_WireModel):
    run_id: NonEmptyId
    expected_version: NonNegativeVersion
    error: RunErrorWire
    failed_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue
    execution_fence: NonEmptyFence | None


class RequestCancelRunRequestWire(_WireModel):
    run_id: NonEmptyId
    expected_version: NonNegativeVersion
    requested_by: NonEmptyId
    reason: str | None
    event_context: EventContextWire
    commit_id: CommitIdValue


class AcknowledgeCancelRunRequestWire(_WireModel):
    run_id: NonEmptyId
    expected_version: NonNegativeVersion
    cancelled_event: EventWire
    event_context: EventContextWire
    commit_id: CommitIdValue
    execution_fence: NonEmptyFence | None


class StartedRunResultWire(_WireModel):
    record: RunRecordWire


class PausedRunResultWire(_WireModel):
    approval_id: NonEmptyId
    checkpoint_id: NonEmptyId


class ResumedRunResultWire(_WireModel):
    run_id: NonEmptyId


class CompletedRunResultWire(_WireModel):
    result: RunResultWire


class FailedRunResultWire(_WireModel):
    run_id: NonEmptyId


class CancellingRunResultWire(_WireModel):
    run_id: NonEmptyId


class CancelledRunResultWire(_WireModel):
    run_id: NonEmptyId


def require_json_value(value: Any, path: str = "value") -> JsonValue:
    """Validate the boundary used by free-form domain values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunCommitCodecError(f"{path}: non-finite float is not JSON-compatible")
        return value
    if isinstance(value, (list, tuple)):
        return [require_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RunCommitCodecError(f"{path}: mapping key must be str")
            result[key] = require_json_value(item, f"{path}.{key}")
        return result
    raise RunCommitCodecError(f"{path}: {type(value).__name__} is not JSON-compatible")


def _json(value: Any, path: str) -> JsonValue:
    return require_json_value(value, path)


def _dt(value: datetime, path: str) -> str:
    if value.tzinfo is None:
        raise RunCommitCodecError(f"{path}: datetime must have timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_dt(value: Any, path: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RunCommitCodecError(f"{path}: datetime must be UTC Z string")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RunCommitCodecError(f"{path}: invalid datetime") from exc


def _b64(value: bytes, path: str) -> str:
    if not isinstance(value, bytes):
        raise RunCommitCodecError(f"{path}: expected bytes")
    return base64.b64encode(value).decode("ascii")


def _unb64(value: Any, path: str) -> bytes:
    if not isinstance(value, str):
        raise RunCommitCodecError(f"{path}: expected base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise RunCommitCodecError(f"{path}: invalid base64") from exc


def _ctx(value: EventStreamContext) -> dict[str, Any]:
    return {"stream_id": value.stream_id, "run_id": value.run_id,
            "root_run_id": value.root_run_id, "parent_run_id": value.parent_run_id,
            "session_id": value.session_id, "runnable_id": value.runnable_id}


def _ctx_from(value: EventContextWire) -> EventStreamContext:
    return EventStreamContext(**value.model_dump())


_EVENTS = {RunStarted: ("run.started", ("run_id", "runnable_id")), RunPaused: ("run.paused", ("run_id", "reason")),
           RunResumed: ("run.resumed", ("run_id",)), RunCompleted: ("run.completed", ("run_id", "result_summary")),
           RunFailed: ("run.failed", ("run_id", "error_type", "message")), RunCancelled: ("run.cancelled", ("run_id", "reason"))}
_EVENT_BY_NAME = {name: cls for cls, (name, _) in _EVENTS.items()}
_EVENT_PAYLOAD_MODELS = {
    "run.started": RunStartedEventPayloadWire,
    "run.paused": RunPausedEventPayloadWire,
    "run.resumed": RunResumedEventPayloadWire,
    "run.completed": RunCompletedEventPayloadWire,
    "run.failed": RunFailedEventPayloadWire,
    "run.cancelled": RunCancelledEventPayloadWire,
}


def _event(value: Any) -> dict[str, Any]:
    for cls, (name, fields) in _EVENTS.items():
        if isinstance(value, cls):
            payload = {field: _json(getattr(value, field), f"event.{field}") for field in fields}
            try:
                payload_model = _EVENT_PAYLOAD_MODELS[name].model_validate(payload)
            except ValidationError as exc:
                raise RunCommitCodecError(f"invalid {name} payload") from exc
            return {"event_type": name, "schema_version": 1, "payload": payload_model.model_dump(mode="json")}
    raise RunCommitCodecError(f"event: unsupported lifecycle event {type(value).__name__}")


def _event_from(value: EventWire) -> Any:
    if value.schema_version != 1:
        raise RunCommitCodecError("event: unknown schema")
    payload_model = _EVENT_PAYLOAD_MODELS.get(value.event_type)
    event_cls = _EVENT_BY_NAME.get(value.event_type)
    if payload_model is None or event_cls is None:
        raise RunCommitCodecError(f"unknown lifecycle event {value.event_type!r}")
    try:
        payload = payload_model.model_validate(value.payload)
    except ValidationError as exc:
        raise RunCommitCodecError(f"invalid {value.event_type} payload") from exc
    return event_cls(**payload.model_dump())


def _message(value: NewSessionMessage) -> dict[str, Any]:
    return {"role": value.role.value, "content": _json(value.content, "message.content"),
            "run_id": value.run_id, "metadata": _json(value.metadata, "message.metadata")}


def _message_from(value: NewSessionMessageWire) -> NewSessionMessage:
    try: role = MessageRole(value.role)
    except ValueError as exc: raise RunCommitCodecError("message.role: unknown role") from exc
    return NewSessionMessage(role=role, content=value.content, run_id=value.run_id, metadata=value.metadata)


def _input(value: RunInput) -> dict[str, Any]:
    return {"prompt": value.prompt, "metadata": _json(value.metadata, "input.metadata")}


def _input_from(value: RunInputWire) -> RunInput:
    return RunInput(prompt=value.prompt, metadata=value.metadata)


def _result(value: RunResult | None) -> dict[str, Any] | None:
    if value is None: return None
    return {"output": _json(value.output, "result.output"), "token_usage": _json(value.token_usage, "result.token_usage"), "metadata": _json(value.metadata, "result.metadata")}


def run_result_from_wire(value: RunResultWire) -> RunResult:
    return RunResult(output=value.output, token_usage=value.token_usage, metadata=value.metadata)


def _error(value: RunErrorInfo | None) -> dict[str, Any] | None:
    if value is None: return None
    return {"error_type": value.error_type, "message": value.message, "detail": _json(value.detail, "error.detail")}


def run_error_from_wire(value: RunErrorWire) -> RunErrorInfo:
    return RunErrorInfo(error_type=value.error_type, message=value.message, detail=value.detail)


def run_record_to_wire(value: RunRecord) -> dict[str, Any]:
    return {"id": value.id, "root_run_id": value.root_run_id, "parent_run_id": value.parent_run_id,
            "session_id": value.session_id, "runnable_id": value.runnable_id, "runnable_type": value.runnable_type.value,
            "status": value.status.value, "input": _input(value.input), "result": _result(value.result), "error": _error(value.error),
            "version": value.version, "created_at": _dt(value.created_at, "record.created_at"),
            "started_at": _dt(value.started_at, "record.started_at") if value.started_at else None,
            "finished_at": _dt(value.finished_at, "record.finished_at") if value.finished_at else None,
            "metadata": _json(value.metadata, "record.metadata"), "cancel_requested_at": _dt(value.cancel_requested_at, "record.cancel_requested_at") if value.cancel_requested_at else None,
            "cancel_requested_by": value.cancel_requested_by, "cancel_reason": value.cancel_reason, "worker_id": value.worker_id,
            "execution_token": value.execution_token, "heartbeat_at": _dt(value.heartbeat_at, "record.heartbeat_at") if value.heartbeat_at else None,
            "manifest_id": value.manifest_id, "resumability": value.resumability}


def run_record_from_wire(value: RunRecordWire) -> RunRecord:
    try:
        return RunRecord(
            id=value.id, root_run_id=value.root_run_id, parent_run_id=value.parent_run_id,
            session_id=value.session_id, runnable_id=value.runnable_id,
            runnable_type=RunnableType(value.runnable_type), status=RunStatus(value.status),
            input=_input_from(value.input),
            result=None if value.result is None else run_result_from_wire(value.result),
            error=None if value.error is None else run_error_from_wire(value.error),
            version=value.version, created_at=_parse_dt(value.created_at, "record.created_at"),
            started_at=_parse_dt(value.started_at, "record.started_at") if value.started_at else None,
            finished_at=_parse_dt(value.finished_at, "record.finished_at") if value.finished_at else None,
            metadata=value.metadata,
            cancel_requested_at=_parse_dt(value.cancel_requested_at, "record.cancel_requested_at") if value.cancel_requested_at else None,
            cancel_requested_by=value.cancel_requested_by, cancel_reason=value.cancel_reason,
            worker_id=value.worker_id, execution_token=value.execution_token,
            heartbeat_at=_parse_dt(value.heartbeat_at, "record.heartbeat_at") if value.heartbeat_at else None,
            manifest_id=value.manifest_id, resumability=value.resumability,
        )
    except (ValueError, TypeError) as exc:
        raise RunCommitCodecError("record contains invalid domain values") from exc


def _fence(value: ExecutionFence | None) -> str | None:
    return value.token if value else None


def _fence_from(value: str | None) -> ExecutionFence | None:
    return ExecutionFence(value) if value is not None else None


def _approval(value: ApprovalRequestData) -> dict[str, Any]:
    return {"approval_id": value.approval_id, "tool_name": value.tool_name, "reason": value.reason, "arguments": _json(value.arguments, "approval.arguments"), "tenant_id": value.tenant_id, "tool_call_id": value.tool_call_id, "binding": _json(value.binding, "approval.binding")}


def _approval_from(value: ApprovalRequestWire) -> ApprovalRequestData:
    return ApprovalRequestData(**value.model_dump())


def _request_wire(value: DecodedRunCommand, operation: RunCommitOperation) -> _WireModel:
    _require_domain_type(value, operation, result=False)
    common = {"commit_id": value.commit_id.value}
    if operation is RunCommitOperation.START:
        return StartRunRequestWire(record=RunRecordWire.model_validate(run_record_to_wire(value.record)), started_event=EventWire.model_validate(_event(value.started_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), **common)
    if operation is RunCommitOperation.PAUSE:
        return PauseRunRequestWire(run_id=value.run_id, expected_version=value.expected_version, approval_request=ApprovalRequestWire.model_validate(_approval(value.approval_request)), checkpoint_payload_b64=_b64(value.checkpoint_payload, "checkpoint_payload"), paused_event=EventWire.model_validate(_event(value.paused_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), execution_fence=_fence(value.execution_fence), messages=[NewSessionMessageWire.model_validate(_message(x)) for x in value.messages], **common)
    if operation is RunCommitOperation.RESUME:
        return ResumeRunRequestWire(run_id=value.run_id, expected_version=value.expected_version, approval_id=value.approval_id, resumed_event=EventWire.model_validate(_event(value.resumed_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), **common)
    if operation is RunCommitOperation.COMPLETE:
        return CompleteRunRequestWire(run_id=value.run_id, session_id=value.session_id, expected_version=value.expected_version, messages=[NewSessionMessageWire.model_validate(_message(x)) for x in value.messages], checkpoint_payload_b64=_b64(value.checkpoint_payload, "checkpoint_payload"), result=RunResultWire.model_validate(_result(value.result)), completed_event=EventWire.model_validate(_event(value.completed_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), execution_fence=_fence(value.execution_fence), **common)
    if operation is RunCommitOperation.FAIL:
        return FailRunRequestWire(run_id=value.run_id, expected_version=value.expected_version, error=RunErrorWire.model_validate(_error(value.error)), failed_event=EventWire.model_validate(_event(value.failed_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), execution_fence=_fence(value.execution_fence), **common)
    if operation is RunCommitOperation.REQUEST_CANCEL:
        return RequestCancelRunRequestWire(run_id=value.run_id, expected_version=value.expected_version, requested_by=value.requested_by, reason=value.reason, event_context=EventContextWire.model_validate(_ctx(value.event_context)), **common)
    if operation is RunCommitOperation.ACKNOWLEDGE_CANCEL:
        return AcknowledgeCancelRunRequestWire(run_id=value.run_id, expected_version=value.expected_version, cancelled_event=EventWire.model_validate(_event(value.cancelled_event)), event_context=EventContextWire.model_validate(_ctx(value.event_context)), execution_fence=_fence(value.execution_fence), **common)
    raise RunCommitCodecError(f"unsupported operation {operation!r}")


_REQUEST_DOMAIN_TYPES = {
    RunCommitOperation.START: StartRunCommand,
    RunCommitOperation.PAUSE: PauseRunCommand,
    RunCommitOperation.RESUME: ResumeRunCommand,
    RunCommitOperation.COMPLETE: CompleteRunCommand,
    RunCommitOperation.FAIL: FailRunCommand,
    RunCommitOperation.REQUEST_CANCEL: RequestCancelRunCommand,
    RunCommitOperation.ACKNOWLEDGE_CANCEL: AcknowledgeCancelRunCommand,
}
_RESULT_DOMAIN_TYPES = {
    RunCommitOperation.START: StartedRunCommit,
    RunCommitOperation.PAUSE: PausedRunCommit,
    RunCommitOperation.RESUME: ResumedRunCommit,
    RunCommitOperation.COMPLETE: CompletedRunCommit,
    RunCommitOperation.FAIL: FailedRunCommit,
    RunCommitOperation.REQUEST_CANCEL: CancellingRunCommit,
    RunCommitOperation.ACKNOWLEDGE_CANCEL: CancelledRunCommit,
}


def _require_domain_type(value: object, operation: RunCommitOperation, *, result: bool) -> None:
    expected_type = (_RESULT_DOMAIN_TYPES if result else _REQUEST_DOMAIN_TYPES)[operation]
    if not isinstance(value, expected_type):
        raise RunCommitCodecError(
            f"{operation.value} requires {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


REQUEST_WIRE_MODELS = {
    RunCommitOperation.START: StartRunRequestWire,
    RunCommitOperation.PAUSE: PauseRunRequestWire,
    RunCommitOperation.RESUME: ResumeRunRequestWire,
    RunCommitOperation.COMPLETE: CompleteRunRequestWire,
    RunCommitOperation.FAIL: FailRunRequestWire,
    RunCommitOperation.REQUEST_CANCEL: RequestCancelRunRequestWire,
    RunCommitOperation.ACKNOWLEDGE_CANCEL: AcknowledgeCancelRunRequestWire,
}
RESULT_WIRE_MODELS = {
    RunCommitOperation.START: StartedRunResultWire,
    RunCommitOperation.PAUSE: PausedRunResultWire,
    RunCommitOperation.RESUME: ResumedRunResultWire,
    RunCommitOperation.COMPLETE: CompletedRunResultWire,
    RunCommitOperation.FAIL: FailedRunResultWire,
    RunCommitOperation.REQUEST_CANCEL: CancellingRunResultWire,
    RunCommitOperation.ACKNOWLEDGE_CANCEL: CancelledRunResultWire,
}


def _require_event_type(value: EventWire, expected: str, path: str) -> None:
    if value.event_type != expected:
        raise RunCommitCodecError(f"{path}.event_type must be {expected!r}")


def _request_from_wire(value: _WireModel, operation: RunCommitOperation) -> DecodedRunCommand:
    if operation is RunCommitOperation.START:
        wire = value
        assert isinstance(wire, StartRunRequestWire)
        _require_event_type(wire.started_event, "run.started", "start.started_event")
        return StartRunCommand(run_record_from_wire(wire.record), _event_from(wire.started_event), _ctx_from(wire.event_context), RunCommitId(wire.commit_id))
    if operation is RunCommitOperation.PAUSE:
        wire = value
        assert isinstance(wire, PauseRunRequestWire)
        _require_event_type(wire.paused_event, "run.paused", "pause.paused_event")
        return PauseRunCommand(run_id=wire.run_id, expected_version=wire.expected_version, approval_request=_approval_from(wire.approval_request), checkpoint_payload=_unb64(wire.checkpoint_payload_b64, "pause.checkpoint_payload_b64"), paused_event=_event_from(wire.paused_event), event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id), execution_fence=_fence_from(wire.execution_fence), messages=tuple(_message_from(item) for item in wire.messages))
    if operation is RunCommitOperation.RESUME:
        wire = value
        assert isinstance(wire, ResumeRunRequestWire)
        _require_event_type(wire.resumed_event, "run.resumed", "resume.resumed_event")
        return ResumeRunCommand(run_id=wire.run_id, expected_version=wire.expected_version, approval_id=wire.approval_id, resumed_event=_event_from(wire.resumed_event), event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id))
    if operation is RunCommitOperation.COMPLETE:
        wire = value
        assert isinstance(wire, CompleteRunRequestWire)
        _require_event_type(wire.completed_event, "run.completed", "complete.completed_event")
        return CompleteRunCommand(run_id=wire.run_id, session_id=wire.session_id, expected_version=wire.expected_version, messages=tuple(_message_from(item) for item in wire.messages), checkpoint_payload=_unb64(wire.checkpoint_payload_b64, "complete.checkpoint_payload_b64"), result=run_result_from_wire(wire.result), completed_event=_event_from(wire.completed_event), event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id), execution_fence=_fence_from(wire.execution_fence))
    if operation is RunCommitOperation.FAIL:
        wire = value
        assert isinstance(wire, FailRunRequestWire)
        _require_event_type(wire.failed_event, "run.failed", "fail.failed_event")
        return FailRunCommand(run_id=wire.run_id, expected_version=wire.expected_version, error=run_error_from_wire(wire.error), failed_event=_event_from(wire.failed_event), event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id), execution_fence=_fence_from(wire.execution_fence))
    if operation is RunCommitOperation.REQUEST_CANCEL:
        wire = value
        assert isinstance(wire, RequestCancelRunRequestWire)
        return RequestCancelRunCommand(run_id=wire.run_id, expected_version=wire.expected_version, requested_by=wire.requested_by, reason=wire.reason, event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id))
    wire = value
    assert isinstance(wire, AcknowledgeCancelRunRequestWire)
    _require_event_type(wire.cancelled_event, "run.cancelled", "acknowledge_cancel.cancelled_event")
    return AcknowledgeCancelRunCommand(run_id=wire.run_id, expected_version=wire.expected_version, cancelled_event=_event_from(wire.cancelled_event), event_context=_ctx_from(wire.event_context), commit_id=RunCommitId(wire.commit_id), execution_fence=_fence_from(wire.execution_fence))


def _decode_result_model(value: _WireModel, operation: RunCommitOperation) -> DecodedRunResult:
    if operation is RunCommitOperation.START:
        assert isinstance(value, StartedRunResultWire)
        return StartedRunCommit(run_record_from_wire(value.record))
    if operation is RunCommitOperation.PAUSE:
        assert isinstance(value, PausedRunResultWire)
        return PausedRunCommit(approval_id=value.approval_id, checkpoint_id=value.checkpoint_id)
    if operation is RunCommitOperation.RESUME:
        assert isinstance(value, ResumedRunResultWire)
        return ResumedRunCommit(run_id=value.run_id)
    if operation is RunCommitOperation.COMPLETE:
        assert isinstance(value, CompletedRunResultWire)
        return CompletedRunCommit(result=run_result_from_wire(value.result))
    if operation is RunCommitOperation.FAIL:
        assert isinstance(value, FailedRunResultWire)
        return FailedRunCommit(run_id=value.run_id)
    if operation is RunCommitOperation.REQUEST_CANCEL:
        assert isinstance(value, CancellingRunResultWire)
        return CancellingRunCommit(run_id=value.run_id)
    assert isinstance(value, CancelledRunResultWire)
    return CancelledRunCommit(run_id=value.run_id)


def _result_wire(value: DecodedRunResult, operation: RunCommitOperation) -> _WireModel:
    _require_domain_type(value, operation, result=True)
    if operation is RunCommitOperation.START:
        return StartedRunResultWire(record=RunRecordWire.model_validate(run_record_to_wire(value.record)))
    if operation is RunCommitOperation.PAUSE:
        return PausedRunResultWire(approval_id=value.approval_id, checkpoint_id=value.checkpoint_id)
    if operation is RunCommitOperation.COMPLETE:
        return CompletedRunResultWire(result=RunResultWire.model_validate(_result(value.result)))
    model = RESULT_WIRE_MODELS[operation]
    return model(run_id=value.run_id)


def _canonical(model: Mapping[str, Any]) -> bytes:
    return json.dumps(model, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def encode_envelope(operation: RunCommitOperation | str, payload: Any, *, kind: str = "request") -> bytes:
    """Encode a validated request or result envelope canonically."""
    try:
        op = RunCommitOperation(operation)
        if kind not in ("request", "result"):
            raise RunCommitCodecError("unknown envelope kind")
        body = (_request_wire(payload, op) if kind == "request" else _result_wire(payload, op)).model_dump(mode="json")
        envelope = RunCommitWireEnvelope(
            schema_version=SCHEMA_VERSION, operation=op, kind=kind, payload=body
        )
        return _canonical(envelope.model_dump(mode="json"))
    except RunCommitCodecError:
        raise
    except (AttributeError, ValidationError, KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise RunCommitCodecError("cannot encode malformed Run payload") from exc


def decode_envelope(payload: bytes, *, expected_operation: RunCommitOperation | str | None = None, expected_kind: str = "request") -> Any:
    """Decode and strictly validate one expected operation and payload kind."""
    try:
        envelope = RunCommitWireEnvelope.model_validate_json(payload)
        if envelope.schema_version != SCHEMA_VERSION or envelope.kind != expected_kind:
            raise RunCommitCodecError("unsupported schema or envelope kind")
        operation = RunCommitOperation(envelope.operation)
        if expected_operation is not None and operation is not RunCommitOperation(expected_operation):
            raise RunCommitCodecError("operation mismatch")
        model_type = (REQUEST_WIRE_MODELS if expected_kind == "request" else RESULT_WIRE_MODELS)[operation]
        wire = model_type.model_validate(envelope.payload)
        return (_request_from_wire(wire, operation) if expected_kind == "request" else _decode_result_model(wire, operation))
    except RunCommitCodecError:
        raise
    except (ValidationError, KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise RunCommitCodecError(
            f"invalid {expected_kind} Run payload"
        ) from exc


__all__ = ["JsonValue", "RunCommitOperation", "RunCommitCodecError", "RunCommitIntegrityError", "SCHEMA_VERSION", "RunCommitWireEnvelope", "DateTimeWire", "RunInputWire", "RunResultWire", "RunErrorWire", "NewSessionMessageWire", "EventContextWire", "EventWire", "RunStartedEventPayloadWire", "RunPausedEventPayloadWire", "RunResumedEventPayloadWire", "RunCompletedEventPayloadWire", "RunFailedEventPayloadWire", "RunCancelledEventPayloadWire", "RunRecordWire", "ApprovalRequestWire", "StartRunRequestWire", "PauseRunRequestWire", "ResumeRunRequestWire", "CompleteRunRequestWire", "FailRunRequestWire", "RequestCancelRunRequestWire", "AcknowledgeCancelRunRequestWire", "StartedRunResultWire", "PausedRunResultWire", "ResumedRunResultWire", "CompletedRunResultWire", "FailedRunResultWire", "CancellingRunResultWire", "CancelledRunResultWire", "REQUEST_WIRE_MODELS", "RESULT_WIRE_MODELS", "require_json_value", "run_record_to_wire", "run_record_from_wire", "run_result_from_wire", "run_error_from_wire", "encode_envelope", "decode_envelope"]
