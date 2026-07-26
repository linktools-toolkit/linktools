"""The explicit, versioned Run commit wire protocol.

This module deliberately contains no object reflection.  A persisted Run
payload is a small JSON document whose shape is owned by this module, so a
restart does not need to import an arbitrary Python type to read it.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, TypeVar
from typing import Literal
from typing_extensions import TypeAliasType

from pydantic import BaseModel, ConfigDict, Field

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


def _obj(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunCommitCodecError(f"{path}: expected object")
    return value


def _exact(value: Any, fields: set[str], path: str) -> dict[str, Any]:
    obj = _obj(value, path)
    extra = set(obj) - fields
    missing = fields - set(obj)
    if extra:
        raise RunCommitCodecError(f"{path}: unknown fields {sorted(extra)!r}")
    if missing:
        raise RunCommitCodecError(f"{path}: missing fields {sorted(missing)!r}")
    return obj


def _ctx(value: EventStreamContext) -> dict[str, Any]:
    return {"stream_id": value.stream_id, "run_id": value.run_id,
            "root_run_id": value.root_run_id, "parent_run_id": value.parent_run_id,
            "session_id": value.session_id, "runnable_id": value.runnable_id}


def _ctx_from(value: Any) -> EventStreamContext:
    o = _exact(value, {"stream_id", "run_id", "root_run_id", "parent_run_id", "session_id", "runnable_id"}, "event_context")
    return EventStreamContext(**o)


_EVENTS = {RunStarted: ("started", ("run_id", "runnable_id")), RunPaused: ("paused", ("run_id", "reason")),
           RunResumed: ("resumed", ("run_id",)), RunCompleted: ("completed", ("run_id", "result_summary")),
           RunFailed: ("failed", ("run_id", "error_type", "message")), RunCancelled: ("cancelled", ("run_id", "reason"))}
_EVENT_BY_NAME = {name: cls for cls, (name, _) in _EVENTS.items()}


def _event(value: Any) -> dict[str, Any]:
    for cls, (name, fields) in _EVENTS.items():
        if isinstance(value, cls):
            return {"event_type": name, "schema_version": 1, "payload": {field: _json(getattr(value, field), f"event.{field}") for field in fields}}
    raise RunCommitCodecError(f"event: unsupported lifecycle event {type(value).__name__}")


def _event_from(value: Any) -> Any:
    o = _exact(value, {"event_type", "schema_version", "payload"}, "event")
    if o["schema_version"] != 1 or o["event_type"] not in _EVENT_BY_NAME:
        raise RunCommitCodecError("event: unknown type or schema")
    cls = _EVENT_BY_NAME[o["event_type"]]
    fields = _EVENTS[cls][1]
    p = _exact(o["payload"], set(fields), "event.payload")
    return cls(**p)


def _message(value: NewSessionMessage) -> dict[str, Any]:
    return {"role": value.role.value, "content": _json(value.content, "message.content"),
            "run_id": value.run_id, "metadata": _json(value.metadata, "message.metadata")}


def _message_from(value: Any) -> NewSessionMessage:
    o = _exact(value, {"role", "content", "run_id", "metadata"}, "message")
    try: role = MessageRole(o["role"])
    except ValueError as exc: raise RunCommitCodecError("message.role: unknown role") from exc
    return NewSessionMessage(role=role, content=o["content"], run_id=o["run_id"], metadata=o["metadata"])


def _input(value: RunInput) -> dict[str, Any]:
    return {"prompt": value.prompt, "metadata": _json(value.metadata, "input.metadata")}


def _input_from(value: Any) -> RunInput:
    o = _exact(value, {"prompt", "metadata"}, "input")
    return RunInput(prompt=o["prompt"], metadata=o["metadata"])


def _result(value: RunResult | None) -> dict[str, Any] | None:
    if value is None: return None
    return {"output": _json(value.output, "result.output"), "token_usage": _json(value.token_usage, "result.token_usage"), "metadata": _json(value.metadata, "result.metadata")}


def _result_from(value: Any) -> RunResult | None:
    if value is None: return None
    o = _exact(value, {"output", "token_usage", "metadata"}, "result")
    return RunResult(output=o["output"], token_usage=o["token_usage"], metadata=o["metadata"])


def _error(value: RunErrorInfo | None) -> dict[str, Any] | None:
    if value is None: return None
    return {"error_type": value.error_type, "message": value.message, "detail": _json(value.detail, "error.detail")}


def _error_from(value: Any) -> RunErrorInfo | None:
    if value is None: return None
    o = _exact(value, {"error_type", "message", "detail"}, "error")
    return RunErrorInfo(error_type=o["error_type"], message=o["message"], detail=o["detail"])


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


def run_record_from_wire(value: Any) -> RunRecord:
    fields = {"id", "root_run_id", "parent_run_id", "session_id", "runnable_id", "runnable_type", "status", "input", "result", "error", "version", "created_at", "started_at", "finished_at", "metadata", "cancel_requested_at", "cancel_requested_by", "cancel_reason", "worker_id", "execution_token", "heartbeat_at", "manifest_id", "resumability"}
    o = _exact(value, fields, "record")
    return RunRecord(id=o["id"], root_run_id=o["root_run_id"], parent_run_id=o["parent_run_id"], session_id=o["session_id"], runnable_id=o["runnable_id"], runnable_type=RunnableType(o["runnable_type"]), status=RunStatus(o["status"]), input=_input_from(o["input"]), result=_result_from(o["result"]), error=_error_from(o["error"]), version=o["version"], created_at=_parse_dt(o["created_at"], "record.created_at"), started_at=_parse_dt(o["started_at"], "record.started_at") if o["started_at"] else None, finished_at=_parse_dt(o["finished_at"], "record.finished_at") if o["finished_at"] else None, metadata=o["metadata"], cancel_requested_at=_parse_dt(o["cancel_requested_at"], "record.cancel_requested_at") if o["cancel_requested_at"] else None, cancel_requested_by=o["cancel_requested_by"], cancel_reason=o["cancel_reason"], worker_id=o["worker_id"], execution_token=o["execution_token"], heartbeat_at=_parse_dt(o["heartbeat_at"], "record.heartbeat_at") if o["heartbeat_at"] else None, manifest_id=o["manifest_id"], resumability=o["resumability"])


def _fence(value: ExecutionFence | None) -> str | None:
    return value.token if value else None


def _fence_from(value: Any) -> ExecutionFence | None:
    return ExecutionFence(value) if value is not None else None


def _approval(value: ApprovalRequestData) -> dict[str, Any]:
    return {"approval_id": value.approval_id, "tool_name": value.tool_name, "reason": value.reason, "arguments": _json(value.arguments, "approval.arguments"), "tenant_id": value.tenant_id, "tool_call_id": value.tool_call_id, "binding": _json(value.binding, "approval.binding")}


def _approval_from(value: Any) -> ApprovalRequestData:
    o = _exact(value, {"approval_id", "tool_name", "reason", "arguments", "tenant_id", "tool_call_id", "binding"}, "approval")
    return ApprovalRequestData(**o)


def _command(value: Any, operation: RunCommitOperation) -> dict[str, Any]:
    common = {"commit_id": value.commit_id.value}
    if operation is RunCommitOperation.START: return {**common, "record": run_record_to_wire(value.record), "started_event": _event(value.started_event), "event_context": _ctx(value.event_context)}
    if operation is RunCommitOperation.PAUSE: return {**common, "run_id": value.run_id, "expected_version": value.expected_version, "approval_request": _approval(value.approval_request), "checkpoint_payload_b64": _b64(value.checkpoint_payload, "checkpoint_payload"), "paused_event": _event(value.paused_event), "event_context": _ctx(value.event_context), "execution_fence": _fence(value.execution_fence), "messages": [_message(x) for x in value.messages]}
    if operation is RunCommitOperation.RESUME: return {**common, "run_id": value.run_id, "expected_version": value.expected_version, "approval_id": value.approval_id, "resumed_event": _event(value.resumed_event), "event_context": _ctx(value.event_context)}
    if operation is RunCommitOperation.COMPLETE: return {**common, "run_id": value.run_id, "session_id": value.session_id, "expected_version": value.expected_version, "messages": [_message(x) for x in value.messages], "checkpoint_payload_b64": _b64(value.checkpoint_payload, "checkpoint_payload"), "result": _result(value.result), "completed_event": _event(value.completed_event), "event_context": _ctx(value.event_context), "execution_fence": _fence(value.execution_fence)}
    if operation is RunCommitOperation.FAIL: return {**common, "run_id": value.run_id, "expected_version": value.expected_version, "error": _error(value.error), "failed_event": _event(value.failed_event), "event_context": _ctx(value.event_context), "execution_fence": _fence(value.execution_fence)}
    if operation is RunCommitOperation.REQUEST_CANCEL: return {**common, "run_id": value.run_id, "expected_version": value.expected_version, "requested_by": value.requested_by, "reason": value.reason, "event_context": _ctx(value.event_context)}
    if operation is RunCommitOperation.ACKNOWLEDGE_CANCEL: return {**common, "run_id": value.run_id, "expected_version": value.expected_version, "cancelled_event": _event(value.cancelled_event), "event_context": _ctx(value.event_context), "execution_fence": _fence(value.execution_fence)}
    raise RunCommitCodecError(f"unsupported operation {operation!r}")


def _result_wire(value: Any, operation: RunCommitOperation) -> dict[str, Any]:
    if operation is RunCommitOperation.START: return {"record": run_record_to_wire(value.record)}
    if operation is RunCommitOperation.PAUSE: return {"approval_id": value.approval_id, "checkpoint_id": value.checkpoint_id}
    if operation is RunCommitOperation.COMPLETE: return {"result": _result(value.result)}
    return {"run_id": value.run_id}


def _parse_common(o: dict[str, Any]) -> RunCommitId:
    return RunCommitId(o["commit_id"])


def _command_from(value: Any, operation: RunCommitOperation) -> Any:
    o = _obj(value, "payload")
    if operation is RunCommitOperation.START:
        _exact(o, {"commit_id", "record", "started_event", "event_context"}, "payload")
        return StartRunCommand(run_record_from_wire(o["record"]), _event_from(o["started_event"]), _ctx_from(o["event_context"]), _parse_common(o))
    base = {"commit_id": _parse_common(o), "run_id": o["run_id"], "expected_version": o["expected_version"], "event_context": _ctx_from(o["event_context"])}
    if operation is RunCommitOperation.PAUSE:
        _exact(o, {"commit_id", "run_id", "expected_version", "approval_request", "checkpoint_payload_b64", "paused_event", "event_context", "execution_fence", "messages"}, "payload")
        return PauseRunCommand(**base, approval_request=_approval_from(o["approval_request"]), checkpoint_payload=_unb64(o["checkpoint_payload_b64"], "checkpoint_payload"), paused_event=_event_from(o["paused_event"]), execution_fence=_fence_from(o["execution_fence"]), messages=tuple(_message_from(x) for x in o["messages"]))
    if operation is RunCommitOperation.RESUME:
        _exact(o, {"commit_id", "run_id", "expected_version", "approval_id", "resumed_event", "event_context"}, "payload")
        return ResumeRunCommand(**base, approval_id=o["approval_id"], resumed_event=_event_from(o["resumed_event"]))
    if operation is RunCommitOperation.COMPLETE:
        _exact(o, {"commit_id", "run_id", "session_id", "expected_version", "messages", "checkpoint_payload_b64", "result", "completed_event", "event_context", "execution_fence"}, "payload")
        return CompleteRunCommand(**base, session_id=o["session_id"], messages=tuple(_message_from(x) for x in o["messages"]), checkpoint_payload=_unb64(o["checkpoint_payload_b64"], "checkpoint_payload"), result=_result_from(o["result"]), completed_event=_event_from(o["completed_event"]), execution_fence=_fence_from(o["execution_fence"]))
    if operation is RunCommitOperation.FAIL:
        _exact(o, {"commit_id", "run_id", "expected_version", "error", "failed_event", "event_context", "execution_fence"}, "payload")
        return FailRunCommand(**base, error=_error_from(o["error"]), failed_event=_event_from(o["failed_event"]), execution_fence=_fence_from(o["execution_fence"]))
    if operation is RunCommitOperation.REQUEST_CANCEL:
        _exact(o, {"commit_id", "run_id", "expected_version", "requested_by", "reason", "event_context"}, "payload")
        return RequestCancelRunCommand(**base, requested_by=o["requested_by"], reason=o["reason"])
    _exact(o, {"commit_id", "run_id", "expected_version", "cancelled_event", "event_context", "execution_fence"}, "payload")
    return AcknowledgeCancelRunCommand(**base, cancelled_event=_event_from(o["cancelled_event"]), execution_fence=_fence_from(o["execution_fence"]))


def _result_from_wire(value: Any, operation: RunCommitOperation) -> Any:
    o = _obj(value, "payload")
    if operation is RunCommitOperation.START: return StartedRunCommit(run_record_from_wire(_exact(o, {"record"}, "payload")["record"]))
    if operation is RunCommitOperation.PAUSE: _exact(o, {"approval_id", "checkpoint_id"}, "payload"); return PausedRunCommit(**o)
    if operation is RunCommitOperation.RESUME: _exact(o, {"run_id"}, "payload"); return ResumedRunCommit(**o)
    if operation is RunCommitOperation.COMPLETE: _exact(o, {"result"}, "payload"); return CompletedRunCommit(result=_result_from(o["result"]))
    if operation is RunCommitOperation.FAIL: _exact(o, {"run_id"}, "payload"); return FailedRunCommit(**o)
    if operation is RunCommitOperation.REQUEST_CANCEL: _exact(o, {"run_id"}, "payload"); return CancellingRunCommit(**o)
    _exact(o, {"run_id"}, "payload"); return CancelledRunCommit(**o)


def _canonical(model: Mapping[str, Any]) -> bytes:
    return json.dumps(model, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def encode_envelope(operation: RunCommitOperation | str, payload: Any, *, kind: str = "request") -> bytes:
    """Encode a validated request or result envelope canonically."""
    try: op = RunCommitOperation(operation)
    except ValueError as exc: raise RunCommitCodecError(f"unknown operation {operation!r}") from exc
    if kind not in ("request", "result"): raise RunCommitCodecError("unknown envelope kind")
    body = _command(payload, op) if kind == "request" else _result_wire(payload, op)
    envelope = RunCommitWireEnvelope(
        schema_version=SCHEMA_VERSION, operation=op, kind=kind, payload=body
    )
    return _canonical(envelope.model_dump(mode="json"))


def decode_envelope(payload: bytes, *, expected_operation: RunCommitOperation | str | None = None, expected_kind: str = "request") -> Any:
    """Decode and strictly validate one expected operation and payload kind."""
    try: raw = json.loads(payload)
    except (TypeError, ValueError) as exc: raise RunCommitCodecError("wire payload is not valid JSON") from exc
    try:
        envelope = RunCommitWireEnvelope.model_validate(raw)
    except Exception as exc:
        raise RunCommitCodecError("malformed wire envelope") from exc
    o = envelope.model_dump(mode="json")
    if o["schema_version"] != SCHEMA_VERSION or o["kind"] != expected_kind: raise RunCommitCodecError("unsupported schema or envelope kind")
    try: op = RunCommitOperation(o["operation"])
    except ValueError as exc: raise RunCommitCodecError("unknown operation") from exc
    if expected_operation is not None and op is not RunCommitOperation(expected_operation): raise RunCommitCodecError("operation mismatch")
    return (_command_from(o["payload"], op) if expected_kind == "request" else _result_from_wire(o["payload"], op))


__all__ = ["JsonValue", "RunCommitOperation", "RunCommitCodecError", "RunCommitIntegrityError", "SCHEMA_VERSION", "RunCommitWireEnvelope", "DateTimeWire", "RunInputWire", "RunResultWire", "RunErrorWire", "NewSessionMessageWire", "EventContextWire", "EventWire", "require_json_value", "run_record_to_wire", "run_record_from_wire", "encode_envelope", "decode_envelope"]
