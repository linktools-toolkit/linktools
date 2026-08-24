#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical versioned codecs for Runtime persistence values."""

import base64
import binascii
import json
import math
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from operator import attrgetter
from types import MappingProxyType
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai_harness.step_persistence import (
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from ...agent import AgentBindingSnapshot
from ...core import (
    ApprovalDecision,
    ApprovalStatus,
    EvaluationStatus,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    StopReason,
    TaskStatus,
    ToolOperationStatus,
    UsageMetrics,
    canonical_json_bytes,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, StoredPayload
from ...task import (
    TaskGraph,
    TaskGraphLimits,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskTerminalRecord,
)
from .._message import decode_model_messages, encode_model_messages
from .._tool import ToolOperationRecord
from ._contracts import (
    AgentAttemptClaim,
    ApprovalRecord,
    ArtifactRecord,
    ContextProjection,
    ConversationCursor,
    ConversationHistoryIndexNodeRecord,
    ConversationHistoryRecord,
    ConversationHistorySegmentRef,
    EvaluationRecord,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionEventRecord,
    ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
    ExecutionRecord,
    ExecutionRunSealHead,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    HistoryQuality,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    MemoryRecord,
    OperationTerminalUpdate,
    RecoveryActiveRecord,
    RecoveryAdmissionRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
    RecoveryIntegrityReport,
    RecoveryStateRecord,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    ResultRecord,
    RuntimePayloadRef,
    SessionForkResultRecord,
    SessionRecord,
    StoredStepSnapshot,
    ToolOperationAdmission,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptOwnerDomain,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
)
from ._plan import RuntimeDomain
from ._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    validate_record_identity,
)

CURRENT_DATA_VERSION = 1
DomainT = TypeVar("DomainT")
_logger = environ.get_logger("ai.runtime.state.codec")

_V1_WIRE_TYPES: tuple[tuple[str, type[object]], ...] = (
    ("approval_record", ApprovalRecord),
    ("agent_attempt_claim", AgentAttemptClaim),
    ("artifact_record", ArtifactRecord),
    ("conversation_cursor", ConversationCursor),
    ("conversation_history_index_node", ConversationHistoryIndexNodeRecord),
    ("conversation_history", ConversationHistoryRecord),
    ("conversation_history_segment", ConversationHistorySegmentRef),
    ("context_projection", ContextProjection),
    ("evaluation_record", EvaluationRecord),
    ("execution_event", ExecutionEventRecord),
    ("execution_history_head", ExecutionHistoryHeadRecord),
    ("execution_history_seal", ExecutionHistorySealRecord),
    ("execution_history_state", ExecutionHistoryState),
    ("execution_record", ExecutionRecord),
    ("execution_run_seal_head", ExecutionRunSealHead),
    ("execution_start_claim", ExecutionStartClaim),
    ("execution_start_unknown_commit", ExecutionStartUnknownCommit),
    ("execution_cancel_request_commit", ExecutionCancelRequestCommit),
    ("execution_start_reservation", ExecutionStartReservation),
    ("execution_start_reservation_result", ExecutionStartReservationResult),
    ("execution_terminal_commit", ExecutionTerminalCommit),
    ("execution_terminal_commit_result", ExecutionTerminalCommitResult),
    ("execution_event_append", ExecutionEventAppend),
    ("external_call_record", ExternalCallRecord),
    ("idempotency_record", IdempotencyRecord),
    ("memory_record", MemoryRecord),
    ("operation_ledger_input", OperationLedgerInput),
    ("operation_ledger_record", OperationLedgerRecord),
    ("principal", Principal),
    ("recovery_checkpoint", RecoveryCheckpoint),
    ("recovery_admission", RecoveryAdmissionRecord),
    ("recovery_active", RecoveryActiveRecord),
    ("recovery_conversation_intent", RecoveryConversationIntent),
    ("recovery_execution_input", RecoveryExecutionInput),
    ("recovery_idempotency_input", RecoveryIdempotencyInput),
    ("recovery_integrity_report", RecoveryIntegrityReport),
    ("recovery_state", RecoveryStateRecord),
    ("recovery_terminal_handoff", RecoveryTerminalHandoff),
    ("recovery_terminal_outcome", RecoveryTerminalOutcome),
    ("resource_ref", ResourceRef),
    ("result_record", ResultRecord),
    ("session_record", SessionRecord),
    ("session_fork_result", SessionForkResultRecord),
    ("stored_step_snapshot", StoredStepSnapshot),
    ("object_ref", ObjectRef),
    ("stored_payload", StoredPayload),
    ("history_quality", HistoryQuality),
    ("inline_context_block", InlineContextBlock),
    ("loaded_context_message", LoadedContextMessage),
    ("loaded_model_context", LoadedModelContext),
    ("runtime_payload_ref", RuntimePayloadRef),
    ("transcript_chunk", TranscriptChunk),
    ("transcript_head", TranscriptHeadRecord),
    ("transcript_message_ref", TranscriptMessageRef),
    ("transcript_origin", TranscriptOrigin),
    ("transcript_owner_domain", TranscriptOwnerDomain),
    ("transcript_seek_dimension", TranscriptSeekDimension),
    ("transcript_seek", TranscriptSeekRecord),
    ("transcript_span_ref", TranscriptSpanRef),
    ("tool_operation_admission", ToolOperationAdmission),
    ("runtime_domain", RuntimeDomain),
    ("task_graph", TaskGraph),
    ("task_graph_limits", TaskGraphLimits),
    ("task_graph_view", TaskGraphView),
    ("task_lease", TaskLease),
    ("task_node", TaskNode),
    ("task_node_view", TaskNodeView),
    ("task_terminal", TaskTerminalRecord),
    ("tool_operation", ToolOperationRecord),
    ("usage_metrics", UsageMetrics),
    ("run_record", RunRecord),
    ("step_event", StepEvent),
    ("tool_effect", ToolEffectRecord),
)
_V1_WIRE_IDS = MappingProxyType(
    {target: wire_id for wire_id, target in _V1_WIRE_TYPES}
)
_V1_DOMAIN_TYPES = MappingProxyType(
    {wire_id: target for wire_id, target in _V1_WIRE_TYPES}
)

_V1_ENUM_WIRE_TYPES: tuple[tuple[str, type[Enum]], ...] = (
    ("approval_decision", ApprovalDecision),
    ("approval_status", ApprovalStatus),
    ("evaluation_status", EvaluationStatus),
    ("execution_event_type", ExecutionEventType),
    ("execution_history_state", ExecutionHistoryState),
    ("execution_lineage_kind", ExecutionLineageKind),
    ("execution_status", ExecutionStatus),
    ("external_call_status", ExternalCallStatus),
    ("history_quality", HistoryQuality),
    ("idempotency_status", IdempotencyStatus),
    ("operation_kind", OperationKind),
    ("operation_status", OperationStatus),
    ("resource_kind", ResourceKind),
    ("runtime_domain", RuntimeDomain),
    ("recovery_checkpoint_state", RecoveryCheckpointState),
    ("recovery_handoff_phase", RecoveryHandoffPhase),
    ("session_status", SessionStatus),
    ("stop_reason", StopReason),
    ("task_status", TaskStatus),
    ("tool_operation_status", ToolOperationStatus),
    ("transcript_origin", TranscriptOrigin),
    ("transcript_owner_domain", TranscriptOwnerDomain),
    ("transcript_seek_dimension", TranscriptSeekDimension),
)
_V1_ENUM_WIRE_IDS = MappingProxyType(
    {target: wire_id for wire_id, target in _V1_ENUM_WIRE_TYPES}
)
_V1_ENUM_TYPES = MappingProxyType(
    {wire_id: target for wire_id, target in _V1_ENUM_WIRE_TYPES}
)

DataclassEncoder = Callable[
    [object, "_VersionCodec", bool],
    Mapping[str, JsonValue],
]
DataclassDecoder = Callable[
    [Mapping[str, object], "_VersionCodec", bool],
    object,
]


@dataclass(frozen=True, slots=True)
class _VersionCodec:
    version: int
    wire_ids: Mapping[type[object], str]
    domain_types: Mapping[str, type[object]]
    enum_wire_ids: Mapping[type[Enum], str]
    enum_types: Mapping[str, type[Enum]]
    dataclass_encoders: Mapping[str, DataclassEncoder]
    dataclass_decoders: Mapping[str, DataclassDecoder]
    external_schema_types: Mapping[type[object], JsonValue]


def _encode_v1_task_node(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskNode):
        raise TypeError("V1 task_node encoder received the wrong type")
    return {
        "node_id": _encode_domain(value.node_id, codec, persisted=persisted),
        "dependencies": _encode_domain(
            value.dependencies, codec, persisted=persisted
        ),
        "input": _encode_domain(value.input, codec, persisted=persisted),
        "budget_cost": _encode_domain(
            value.budget_cost, codec, persisted=persisted
        ),
    }


def _decode_v1_task_node(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> TaskNode:
    _require_exact_keys(
        raw_fields,
        frozenset(
            {
                "node_id",
                "dependencies",
                "input",
                "budget_cost",
            }
        ),
    )
    return TaskNode(
        str(_decode_domain(raw_fields["node_id"], str, codec, persisted=persisted)),
        tuple(
            _decode_domain(
                raw_fields["dependencies"],
                tuple[str, ...],
                codec,
                persisted=persisted,
            )
        ),
        input=_decode_domain(
            raw_fields["input"],
            Any,
            codec,
            persisted=persisted,
        ),
        budget_cost=int(
            _decode_domain(
                raw_fields["budget_cost"], int, codec, persisted=persisted
            )
        ),
    )


_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {"task_node": _encode_v1_task_node}
)
_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(
    {"task_node": _decode_v1_task_node}
)


_V1_EXTERNAL_SCHEMA_TYPES: Mapping[type[object], JsonValue] = MappingProxyType(
    {
        IdempotencyTerminalUpdate: (
            "linktools.ai.runtime.state.IdempotencyTerminalUpdate"
        ),
        OperationTerminalUpdate: (
            "linktools.ai.runtime.state.OperationTerminalUpdate"
        ),
        AgentBindingSnapshot: "linktools.ai.agent.AgentBindingSnapshot@1",
        ModelRequest: "pydantic_ai.messages.ModelRequest",
        ModelResponse: "pydantic_ai.messages.ModelResponse",
    }
)


_V1_CODEC = _VersionCodec(
    version=1,
    wire_ids=_V1_WIRE_IDS,
    domain_types=_V1_DOMAIN_TYPES,
    enum_wire_ids=_V1_ENUM_WIRE_IDS,
    enum_types=_V1_ENUM_TYPES,
    dataclass_encoders=_V1_DATACLASS_ENCODERS,
    dataclass_decoders=_V1_DATACLASS_DECODERS,
    external_schema_types=_V1_EXTERNAL_SCHEMA_TYPES,
)
_VERSION_CODECS: Mapping[int, _VersionCodec] = MappingProxyType(
    {
        1: _V1_CODEC,
    }
)
_CURRENT_CODEC = _VERSION_CODECS[CURRENT_DATA_VERSION]


@dataclass(frozen=True, slots=True)
class CanonicalEnvelope:
    version: int
    value: Mapping[str, JsonValue]


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    if set(value.keys()) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _unwrap_tagged_list(
    value: object,
    tag: str,
) -> list[object]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({tag}))
    items = value[tag]
    if not isinstance(items, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return items


def encode_envelope(
    value: Mapping[str, JsonValue],
    *,
    version: int = CURRENT_DATA_VERSION,
) -> dict[str, JsonValue]:
    if version != CURRENT_DATA_VERSION:
        raise ValueError("only the frozen current data version may be written")
    if not isinstance(value, Mapping):
        raise TypeError("canonical data value must be a mapping")
    return {"v": version, "value": dict(value)}


def parse_envelope(value: Mapping[str, JsonValue]) -> CanonicalEnvelope:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "canonical data must be an object")
    _require_exact_keys(value, frozenset({"v", "value"}))
    version = value.get("v")
    payload = value.get("value")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "canonical data version is invalid")
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "canonical data value is invalid")
    return CanonicalEnvelope(version, dict(payload))


def decode_envelope(value: Mapping[str, JsonValue]) -> CanonicalEnvelope:
    envelope = parse_envelope(value)
    if envelope.version not in _VERSION_CODECS:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if _VERSION_CODECS[envelope.version].version != envelope.version:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return envelope


def encode_record(record: StoredRecord) -> dict[str, JsonValue]:
    return {
        "key": record.key_digest.hex(),
        "partition": record.partition_digest.hex(),
        "scope": None if record.scope_digest is None else record.scope_digest.hex(),
        "parent": None if record.parent_digest is None else record.parent_digest.hex(),
        "kind": record.kind,
        "sort": record.sort_key,
        "state": record.state,
        "storage_version": record.storage_version,
        "lease": {
            "owner": record.lease_owner,
            "fence": record.lease_fence,
            "expires_at": None
            if record.lease_expires_at is None
            else record.lease_expires_at.isoformat(),
        },
        "data": dict(record.data),
    }


def decode_record(value: Mapping[str, JsonValue]) -> StoredRecord:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({
        "key", "partition", "scope", "parent", "kind", "sort", "state",
        "storage_version", "lease", "data",
    }))
    lease = value["lease"]
    if not isinstance(lease, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(lease, frozenset({"owner", "fence", "expires_at"}))
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        record = StoredRecord(
            _digest_wire(_string(value, "key")),
            _digest_wire(_string(value, "partition")),
            _optional_digest(value["scope"]),
            _optional_digest(value["parent"]),
            _string(value, "kind"),
            _string(value, "sort"),
            _optional_string(value["state"]),
            _integer(value, "storage_version"),
            _optional_string(lease["owner"]),
            _integer(lease, "fence"),
            _optional_datetime(lease["expires_at"]),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        validate_record_identity(record)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return record


def encode_fact(fact: StoredFact) -> dict[str, JsonValue]:
    return {
        "stream": fact.stream_digest.hex(),
        "sequence": fact.sequence,
        "owner": fact.owner_key_digest.hex(),
        "kind": fact.kind,
        "subject": None if fact.subject_digest is None else fact.subject_digest.hex(),
        "state": fact.state,
        "data": dict(fact.data),
    }


def decode_fact(value: Mapping[str, JsonValue]) -> StoredFact:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({
        "stream", "sequence", "owner", "kind", "subject", "state", "data",
    }))
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredFact(
            _digest_wire(_string(value, "stream")),
            _integer(value, "sequence"),
            _digest_wire(_string(value, "owner")),
            _string(value, "kind"),
            _optional_digest(value["subject"]),
            _optional_string(value["state"]),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def encode_operation(operation: StoredOperation) -> dict[str, JsonValue]:
    return {
        "key": operation.key_digest.hex(),
        "stream": operation.stream_digest.hex(),
        "sequence": operation.sequence,
        "state": operation.state,
        "compactable": operation.compactable,
        "data": dict(operation.data),
    }


def decode_operation(value: Mapping[str, JsonValue]) -> StoredOperation:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({
        "key", "stream", "sequence", "state", "compactable", "data",
    }))
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredOperation(
            _digest_wire(_string(value, "key")),
            _digest_wire(_string(value, "stream")),
            _integer(value, "sequence"),
            _string(value, "state"),
            _bool(value, "compactable"),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def encode_alias(alias: StoredAlias) -> dict[str, JsonValue]:
    return {"alias": alias.alias_digest.hex(), "record": alias.record_key_digest.hex()}


def decode_alias(value: Mapping[str, JsonValue]) -> StoredAlias:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({"alias", "record"}))
    try:
        return StoredAlias(
            _digest_wire(_string(value, "alias")),
            _digest_wire(_string(value, "record")),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def canonical_digest(value: Mapping[str, JsonValue]) -> str:
    """Return the digest used for replay and compact evidence."""
    import hashlib

    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def wire_type_id(value: DomainT | type[DomainT]) -> str:
    """Return the explicit stable protocol id for one persisted type."""
    target = value if isinstance(value, type) else type(value)
    if isinstance(target, type) and issubclass(target, Enum):
        try:
            return _CURRENT_CODEC.enum_wire_ids[target]
        except KeyError as error:
            raise TypeError(f"unsupported enum type: {target.__name__}") from error
    try:
        return _CURRENT_CODEC.wire_ids[target]
    except KeyError as error:
        raise TypeError(f"unsupported domain type: {target.__name__}") from error


def _codec_wire_type_id(
    value: DomainT | type[DomainT],
    codec: _VersionCodec,
) -> str:
    target = value if isinstance(value, type) else type(value)
    if isinstance(target, type) and issubclass(target, Enum):
        try:
            return codec.enum_wire_ids[target]
        except KeyError as error:
            raise TypeError(f"unsupported enum type: {target.__name__}") from error
    try:
        return codec.wire_ids[target]
    except KeyError as error:
        raise TypeError(f"unsupported domain type: {target.__name__}") from error


def _encode_external(value: object, codec: _VersionCodec) -> JsonValue:
    if type(value) not in codec.external_schema_types:
        raise TypeError(f"unsupported external type: {type(value).__name__}")
    if isinstance(value, AgentBindingSnapshot):
        return value.to_payload()
    if isinstance(value, IdempotencyTerminalUpdate):
        return {
            "scope": _encode_domain(value.scope, codec),
            "idempotency_key_digest": _encode_domain(value.idempotency_key_digest, codec),
            "expected_status": _encode_domain(value.expected_status, codec),
            "next_status": _encode_domain(value.next_status, codec),
            "request_digest": _encode_domain(value.request_digest, codec),
            "result_digest": _encode_domain(value.result_digest, codec),
            "error_code": _encode_domain(value.error_code, codec),
        }
    if isinstance(value, OperationTerminalUpdate):
        return {
            "operation_id": _encode_domain(value.operation_id, codec),
            "expected_status": _encode_domain(value.expected_status, codec),
            "next_status": _encode_domain(value.next_status, codec),
            "result_ref": _encode_domain(value.result_ref, codec),
            "result_digest": _encode_domain(value.result_digest, codec),
            "error_code": _encode_domain(value.error_code, codec),
        }
    if isinstance(value, (ModelRequest, ModelResponse)):
        raw = encode_model_messages((value,))
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise RuntimeError("model message codec returned an invalid single-message value")
        return cast(JsonValue, decoded[0])
    raise TypeError(f"unsupported external type: {type(value).__name__}")


def _decode_external(value: object, target: type[object], codec: _VersionCodec) -> object:
    if target not in codec.external_schema_types:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if target is AgentBindingSnapshot:
        try:
            return AgentBindingSnapshot.from_payload(value)
        except AIError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if target in (ModelRequest, ModelResponse):
        try:
            raw = canonical_json_bytes([cast(JsonValue, value)])
            messages = decode_model_messages(raw)
        except AIError:
            raise
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if len(messages) != 1 or not isinstance(messages[0], target):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return messages[0]
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if target is IdempotencyTerminalUpdate:
        _require_exact_keys(value, frozenset({
            "scope", "idempotency_key_digest", "expected_status", "next_status",
            "request_digest", "result_digest", "error_code",
        }))
        return IdempotencyTerminalUpdate(
            scope=cast(str, _decode_domain(value["scope"], str, codec)),
            idempotency_key_digest=cast(str, _decode_domain(value["idempotency_key_digest"], str, codec)),
            expected_status=cast(IdempotencyStatus, _decode_domain(value["expected_status"], IdempotencyStatus, codec)),
            next_status=cast(IdempotencyStatus, _decode_domain(value["next_status"], IdempotencyStatus, codec)),
            request_digest=cast(str, _decode_domain(value["request_digest"], str, codec)),
            result_digest=cast(str | None, _decode_domain(value["result_digest"], str | None, codec)),
            error_code=cast(str | None, _decode_domain(value["error_code"], str | None, codec)),
        )
    if target is OperationTerminalUpdate:
        _require_exact_keys(value, frozenset({
            "operation_id", "expected_status", "next_status", "result_ref",
            "result_digest", "error_code",
        }))
        return OperationTerminalUpdate(
            operation_id=cast(str, _decode_domain(value["operation_id"], str, codec)),
            expected_status=cast(OperationStatus, _decode_domain(value["expected_status"], OperationStatus, codec)),
            next_status=cast(OperationStatus, _decode_domain(value["next_status"], OperationStatus, codec)),
            result_ref=cast(str | None, _decode_domain(value["result_ref"], str | None, codec)),
            result_digest=cast(str | None, _decode_domain(value["result_digest"], str | None, codec)),
            error_code=cast(str | None, _decode_domain(value["error_code"], str | None, codec)),
        )
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _encode_enum_value(value: object) -> JsonValue:
    if value is None or type(value) in (str, bool, int):
        return cast(JsonValue, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("enum value must be finite")
        return value
    raise TypeError("enum value must use a canonical primitive")


def _encode_domain(
    value: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> JsonValue:
    if isinstance(value, Enum):
        wire_id = codec.enum_wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"unsupported enum type: {type(value).__name__}")
        return {"$enum": wire_id, "value": _encode_enum_value(value.value)}
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("GA v1 wire requires finite floats")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("domain timestamp must be timezone-aware")
        return {"$datetime": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if type(value) in codec.external_schema_types:
        return _encode_external(value, codec)
    if is_dataclass(value):
        wire_id = codec.wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"unsupported dataclass type: {type(value).__name__}")
        encoder = codec.dataclass_encoders.get(wire_id)
        if encoder is not None:
            encoded_fields = encoder(value, codec, persisted)
        else:
            if any(field.name.startswith("_") for field in fields(value)):
                raise TypeError("private dataclass fields require an explicit codec")
            encoded_fields = {
                field.name: _encode_domain(
                    attrgetter(field.name)(value), codec, persisted=persisted
                )
                for field in fields(value)
            }
        wire: dict[str, JsonValue] = {
            "$dataclass": wire_id,
            "fields": dict(encoded_fields),
        }
        if persisted:
            wire = {
                "$dataclass": wire_id,
                "schema": CURRENT_DATA_VERSION,
                "fields": dict(encoded_fields),
            }
        if encoder is not None:
            try:
                _decode_dataclass(wire, type(value), codec, persisted=persisted)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                    raise
                raise TypeError(
                    f"{type(value).__name__} does not match its custom V1 schema"
                ) from error
            except (TypeError, ValueError, KeyError) as error:
                raise TypeError(
                    f"{type(value).__name__} does not match its custom V1 schema"
                ) from error
        return wire
    if isinstance(value, Mapping):
        encoded_pairs: list[tuple[bytes, JsonValue, JsonValue]] = []
        for key, item in value.items():
            encoded_key = _encode_domain(key, codec, persisted=persisted)
            encoded_item = _encode_domain(item, codec, persisted=persisted)
            encoded_pairs.append(
                (canonical_json_bytes(encoded_key), encoded_key, encoded_item)
            )
        key_bytes = [item[0] for item in encoded_pairs]
        if len(key_bytes) != len(set(key_bytes)):
            raise ValueError("canonical mapping keys collide")
        encoded_pairs.sort(key=lambda item: item[0])
        return {
            "$mapping": [
                [encoded_key, encoded_item]
                for _key_bytes, encoded_key, encoded_item in encoded_pairs
            ]
        }
    if isinstance(value, tuple):
        return {
            "$tuple": [
                _encode_domain(item, codec, persisted=persisted) for item in value
            ]
        }
    if isinstance(value, list):
        return [_encode_domain(item, codec, persisted=persisted) for item in value]
    if isinstance(value, frozenset):
        encoded_items = [
            (canonical_json_bytes(encoded_item), encoded_item)
            for item in value
            for encoded_item in (
                _encode_domain(item, codec, persisted=persisted),
            )
        ]
        item_bytes = [item[0] for item in encoded_items]
        if len(item_bytes) != len(set(item_bytes)):
            raise ValueError("canonical frozenset items collide")
        encoded_items.sort(key=lambda item: item[0])
        return {"$frozenset": [item[1] for item in encoded_items]}
    raise TypeError(f"unsupported domain value: {type(value).__name__}")


def encode_domain(value: DomainT) -> JsonValue:
    """Encode one domain value into the shared canonical JSON representation."""
    return _encode_domain(value, _CURRENT_CODEC)


def _encode_persisted_domain(value: DomainT) -> JsonValue:
    """Encode one domain value for a Runtime persistence envelope."""
    wire = _encode_domain(value, _CURRENT_CODEC, persisted=True)
    try:
        _decode_domain(
            wire,
            type(value),
            _CURRENT_CODEC,
            persisted=True,
        )
    except AIError as error:
        if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
            raise
        _logger.warning(
            "persisted domain writer closure rejected value: type=%s",
            type(value).__name__,
        )
        raise TypeError(
            f"{type(value).__name__} persisted V1 wire is not readable by current decoder"
        ) from error
    except (KeyError, TypeError, ValueError) as error:
        _logger.warning(
            "persisted domain writer closure rejected value: type=%s",
            type(value).__name__,
        )
        raise TypeError(
            f"{type(value).__name__} persisted V1 wire is not readable by current decoder"
        ) from error
    return wire


def decode_domain(value: JsonValue, target: type[DomainT]) -> DomainT:
    """Decode a canonical value using the declared domain type."""
    return _decode_domain(value, target, _CURRENT_CODEC)  # type: ignore[return-value]


def _decode_enveloped_domain(
    value: Mapping[str, JsonValue],
    target: type[DomainT],
    *,
    payload_transform: Callable[[JsonValue], JsonValue] | None = None,
) -> DomainT:
    """Decode persisted domain data without losing its envelope version."""
    envelope = decode_envelope(value)
    _require_exact_keys(envelope.value, frozenset({"type", "payload"}))
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    expected_type = _codec_wire_type_id(target, codec)
    actual_type = envelope.value.get("type")
    payload = envelope.value.get("payload")
    if not isinstance(actual_type, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if actual_type not in codec.domain_types and actual_type not in codec.enum_types:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if actual_type != expected_type or payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if payload_transform is not None:
        payload = payload_transform(payload)
    return _decode_domain(payload, target, codec, persisted=True)


def iter_runtime_object_refs(
    value: JsonValue,
    *,
    default_domain: RuntimeDomain,
) -> Iterator[tuple[RuntimeDomain, ObjectRef]]:
    """Yield object references without depending on a storage backend."""
    yield from _iter_runtime_object_refs(value, default_domain, _CURRENT_CODEC)


def _iter_enveloped_runtime_object_refs(
    value: Mapping[str, JsonValue],
    *,
    default_domain: RuntimeDomain,
) -> Iterator[tuple[RuntimeDomain, ObjectRef]]:
    """Traverse object references using the envelope's own version codec."""
    envelope = decode_envelope(value)
    _require_exact_keys(envelope.value, frozenset({"type", "payload"}))
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    payload = envelope.value.get("payload")
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    yield from _iter_runtime_object_refs(payload, default_domain, codec)


def _iter_runtime_object_refs(
    value: object,
    domain: RuntimeDomain,
    codec: _VersionCodec,
) -> Iterator[tuple[RuntimeDomain, ObjectRef]]:
    if isinstance(value, StoredPayload):
        if value.ref is not None:
            yield domain, value.ref
        return
    if isinstance(value, ObjectRef):
        yield domain, value
        return
    if isinstance(value, RuntimePayloadRef):
        source_domain = value.source_domain or domain
        yield from _iter_runtime_object_refs(value.payload, source_domain, codec)
        return
    if isinstance(value, Mapping):
        dataclass_name = value.get("$dataclass")
        if dataclass_name == codec.wire_ids[RuntimePayloadRef]:
            decoded = _decode_domain(value, RuntimePayloadRef, codec, persisted=True)
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[StoredPayload]:
            decoded = _decode_domain(value, StoredPayload, codec, persisted=True)
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[ObjectRef]:
            decoded = _decode_domain(value, ObjectRef, codec, persisted=True)
            yield domain, decoded
            return
        if dataclass_name == codec.wire_ids.get(RecoveryTerminalOutcome):
            fields_value = value.get("fields")
            if isinstance(fields_value, Mapping):
                output = fields_value.get("output")
                source_domain = _decode_runtime_domain(
                    fields_value.get("object_source_domain"),
                    domain,
                    codec,
                )
                if output is not None:
                    yield from _iter_runtime_object_refs(
                        output,
                        source_domain,
                        codec,
                    )
                for key, item in fields_value.items():
                    if key not in {"output", "object_source_domain"}:
                        yield from _iter_runtime_object_refs(item, domain, codec)
                return
        if {"kind", "digest", "size"}.issubset(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for item in value.values():
            yield from _iter_runtime_object_refs(item, domain, codec)
        return
    if isinstance(value, (list, tuple, frozenset)):
        for item in value:
            yield from _iter_runtime_object_refs(item, domain, codec)


def _decode_runtime_domain(
    value: object,
    default: RuntimeDomain,
    codec: _VersionCodec,
) -> RuntimeDomain:
    if value is None:
        return default
    try:
        result = _decode_domain(value, RuntimeDomain, codec)
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(result, RuntimeDomain):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_domain(
    value: object,
    target: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> object:
    if target is Any or target is object:
        return _decode_any(value, codec, persisted=persisted)
    origin = get_origin(target)
    arguments = get_args(target)
    if origin is Literal:
        for literal in arguments:
            if isinstance(literal, float) and not math.isfinite(literal):
                raise TypeError("GA v1 schema literal requires finite floats")
        if isinstance(value, float):
            _require_finite_wire_float(value)
        if not any(
            type(value) is type(literal) and value == literal
            for literal in arguments
        ):
            raise ValueError("literal value is invalid")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _decode_domain(value, candidate, codec, persisted=persisted)
            except (TypeError, ValueError, KeyError):
                continue
            except AIError as error:
                if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                    raise
                continue
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value is None:
        if target is type(None):
            return None
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if origin in (list,):
        item_type = arguments[0] if arguments else Any
        if not isinstance(value, list):
            raise TypeError("list value is invalid")
        return [
            _decode_domain(item, item_type, codec, persisted=persisted)
            for item in value
        ]
    if origin is tuple:
        items = _unwrap_tagged_list(value, "$tuple")
        if arguments and arguments[-1] is Ellipsis:
            return tuple(
                _decode_domain(item, arguments[0], codec, persisted=persisted)
                for item in items
            )
        if len(items) != len(arguments):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(
            _decode_domain(item, item_type, codec, persisted=persisted)
            for item, item_type in zip(items, arguments, strict=True)
        )
    if target is set or origin is set:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            "GA v1 does not support set values",
        )
    if target is frozenset or origin is frozenset:
        item_type = arguments[0] if arguments else Any
        return _decode_frozenset_items(
            value, item_type, codec, persisted=persisted
        )
    if isinstance(origin, type) and issubclass(origin, Mapping):
        key_type, item_type = arguments if len(arguments) == 2 else (Any, Any)
        return _decode_mapping_items(
            value, key_type, item_type, codec, persisted=persisted
        )
    if target is type(None):
        if value is not None:
            raise TypeError("none value is not null")
        return None
    if isinstance(target, type) and issubclass(target, Enum):
        return _decode_enum(value, target, codec)
    if target is datetime:
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_exact_keys(value, frozenset({"$datetime"}))
        raw = value["$datetime"]
        if not isinstance(raw, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            result = datetime.fromisoformat(raw)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if result.tzinfo is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if result.isoformat() != raw:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "GA v1 datetime wire is not canonical",
            )
        return result
    if target is bytes:
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_exact_keys(value, frozenset({"$bytes"}))
        raw = value["$bytes"]
        if not isinstance(raw, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            result = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        canonical = base64.b64encode(result).decode("ascii")
        if canonical != raw:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "GA v1 bytes wire is not canonical",
            )
        return result
    if isinstance(target, type) and target in codec.external_schema_types:
        return _decode_external(value, target, codec)
    if isinstance(target, type) and is_dataclass(target):
        return _decode_dataclass(value, target, codec, persisted=persisted)
    if target is float:
        if not isinstance(value, float):
            raise TypeError("scalar value has the wrong type")
        return _require_finite_wire_float(value)
    if target in (str, bool, int):
        if not isinstance(value, target) or target is int and isinstance(value, bool):
            raise TypeError("scalar value has the wrong type")
        return value
    return _decode_any(value, codec, persisted=persisted)


def _require_finite_wire_float(value: float) -> float:
    if not math.isfinite(value):
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            "GA v1 wire contains a non-finite float",
        )
    return value


def _decode_enum(
    value: object,
    target: type[Enum],
    codec: _VersionCodec,
) -> Enum:
    expected_wire_id = codec.enum_wire_ids.get(target)
    if expected_wire_id is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_exact_keys(value, frozenset({"$enum", "value"}))
    wire_id = value["$enum"]
    if not isinstance(wire_id, str) or wire_id != expected_wire_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw = value["value"]
    if raw is not None and type(raw) not in (str, bool, int, float):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if type(raw) is float:
        _require_finite_wire_float(raw)
    try:
        result = target(raw)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    if type(result.value) is not type(raw) or result.value != raw:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return result


def _decode_dataclass(
    value: object,
    target: type,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> object:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    keys = frozenset(value)
    if persisted:
        if keys not in {
            frozenset({"$dataclass", "fields"}),
            frozenset({"$dataclass", "schema", "fields"}),
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    else:
        _require_exact_keys(value, frozenset({"$dataclass", "fields"}))

    wire_id = value.get("$dataclass")
    if not isinstance(wire_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    expected_target = codec.domain_types.get(wire_id)
    if expected_target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if expected_target is not target:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    if persisted:
        schema = value.get("schema", CURRENT_DATA_VERSION)
        if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if schema != CURRENT_DATA_VERSION:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    decoder = codec.dataclass_decoders.get(wire_id)
    if decoder is not None:
        try:
            return decoder(raw_fields, codec, persisted)
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    try:
        hints = get_type_hints(target)
    except (NameError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    declared_fields = tuple(fields(target))
    declared_names = {field.name for field in declared_fields}
    unknown_fields = tuple(
        sorted(str(name) for name in set(raw_fields) - declared_names)
    )
    defaulted_fields: list[str] = []
    kwargs: dict[str, object] = {}
    post_init_fields: dict[str, object] = {}
    for field in declared_fields:
        if field.name not in raw_fields:
            if field.init and (
                field.default is MISSING
                and field.default_factory is MISSING
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if field.init:
                defaulted_fields.append(field.name)
            continue
        try:
            decoded = _decode_domain(
                raw_fields[field.name],
                hints.get(field.name, Any),
                codec,
                persisted=persisted,
            )
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if field.init:
            kwargs[field.name] = decoded
        else:
            post_init_fields[field.name] = decoded
    if unknown_fields or defaulted_fields:
        _logger.debug(
            "Runtime generic dataclass decode tolerated shape change: "
            "wire_id=%s ignored_fields=%s defaulted_fields=%s",
            wire_id,
            unknown_fields,
            tuple(defaulted_fields),
        )
    try:
        result = target(**kwargs)
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    for field_name, expected in post_init_fields.items():
        try:
            actual = attrgetter(field_name)(result)
        except AttributeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if actual != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_any(
    value: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> object:
    if isinstance(value, Mapping):
        matched = set(value.keys()).intersection(_RESERVED_V1_TAGS)
        if len(matched) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if len(matched) == 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        tag = next(iter(matched))
        if tag == "$datetime":
            return _decode_domain(value, datetime, codec)
        if tag == "$bytes":
            return _decode_domain(value, bytes, codec)
        if tag == "$enum":
            name = value["$enum"]
            if not isinstance(name, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return _decode_enum(value, _enum_type(name, codec), codec)
        if tag == "$dataclass":
            name = value["$dataclass"]
            if not isinstance(name, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target = codec.domain_types.get(name)
            if target is None:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            return _decode_dataclass(value, target, codec, persisted=persisted)
        if tag == "$mapping":
            return _decode_mapping_items(
                value, Any, Any, codec, persisted=persisted
            )
        if tag == "$tuple":
            return tuple(
                _decode_any(item, codec, persisted=persisted)
                for item in _unwrap_tagged_list(value, "$tuple")
            )
        return _decode_frozenset_items(
            value, Any, codec, persisted=persisted
        )
    if isinstance(value, list):
        return [
            _decode_any(item, codec, persisted=persisted) for item in value
        ]
    if isinstance(value, float):
        return _require_finite_wire_float(value)
    return value


def _enum_type(wire_id: str, codec: _VersionCodec) -> type[Enum]:
    try:
        return codec.enum_types[wire_id]
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error


_RESERVED_V1_TAGS = frozenset(
    {
        "$datetime",
        "$bytes",
        "$enum",
        "$dataclass",
        "$mapping",
        "$tuple",
        "$frozenset",
    }
)


def _canonical_wire_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(cast(JsonValue, value))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _decode_frozenset_items(
    value: object,
    item_type: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> frozenset[object]:
    items = _unwrap_tagged_list(value, "$frozenset")
    decoded_items: list[object] = []
    previous_wire: bytes | None = None
    for item in items:
        decoded_item = _decode_domain(
            item, item_type, codec, persisted=persisted
        )
        item_wire = _canonical_wire_bytes(item)
        if previous_wire is not None and previous_wire >= item_wire:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_wire = item_wire
        decoded_items.append(decoded_item)
    try:
        result = frozenset(decoded_items)
    except TypeError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(result) != len(decoded_items):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_mapping_items(
    value: object,
    key_type: object,
    item_type: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> dict[object, object]:
    pairs = _unwrap_tagged_list(value, "$mapping")
    result: dict[object, object] = {}
    previous_wire: bytes | None = None
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key_wire = _canonical_wire_bytes(pair[0])
        if previous_wire is not None and previous_wire >= key_wire:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_wire = key_wire
        decoded_key = _decode_domain(
            pair[0], key_type, codec, persisted=persisted
        )
        try:
            hash(decoded_key)
        except TypeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if decoded_key in result:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result[decoded_key] = _decode_domain(
            pair[1], item_type, codec, persisted=persisted
        )
    return result


def _string(value: Mapping[str, object], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("value must be a string or null")  # noqa: TRY004
    return value


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")  # noqa: TRY004
    return result


def _bool(value: Mapping[str, object], key: str) -> bool:
    result = value[key]
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")  # noqa: TRY004
    return result


def _digest_wire(value: str) -> bytes:
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("digest must be canonical lowercase hex") from error
    if len(result) != 32 or result.hex() != value:
        raise ValueError("digest must be canonical lowercase hex for 32 bytes")
    return result


def _optional_digest(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("digest must be a string or null")  # noqa: TRY004
    return _digest_wire(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string or null")  # noqa: TRY004
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if result.isoformat() != value:
        raise ValueError("timestamp must use canonical isoformat")
    return result


def _encode_step_envelope(value: object) -> dict[str, JsonValue]:
    return encode_envelope(
        {
            "type": wire_type_id(value),
            "payload": _encode_persisted_domain(value),
        }
    )


def _decode_step_envelope(value: Mapping[str, JsonValue]) -> object:
    envelope = decode_envelope(value)
    _require_exact_keys(envelope.value, frozenset({"type", "payload"}))
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    kind = envelope.value.get("type")
    if not isinstance(kind, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    targets = {
        "run_record": RunRecord,
        "step_event": StepEvent,
        "tool_effect": ToolEffectRecord,
        "stored_step_snapshot": StoredStepSnapshot,
    }
    target = targets.get(kind)
    if target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    expected = _codec_wire_type_id(target, codec)
    if kind != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    payload = envelope.value["payload"]
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_domain(payload, target, codec, persisted=True)


def _validate_v1_codec_definition() -> None:
    if CURRENT_DATA_VERSION != 1 or set(_VERSION_CODECS) != {1}:
        raise RuntimeError("GA v1 codec registry is invalid")
    if _CURRENT_CODEC is not _VERSION_CODECS[1]:
        raise RuntimeError("GA v1 current codec is invalid")
    wire_ids = tuple(wire_id for wire_id, _target in _V1_WIRE_TYPES)
    enum_wire_ids = tuple(wire_id for wire_id, _target in _V1_ENUM_WIRE_TYPES)
    if len(wire_ids) != len(set(wire_ids)):
        raise RuntimeError("GA v1 wire ids are not unique")
    if len(enum_wire_ids) != len(set(enum_wire_ids)):
        raise RuntimeError("GA v1 enum wire ids are not unique")
    if set(_CURRENT_CODEC.domain_types) != set(wire_ids):
        raise RuntimeError("GA v1 domain type registry is incomplete")
    if set(_CURRENT_CODEC.wire_ids.values()) != set(wire_ids):
        raise RuntimeError("GA v1 domain wire-id registry is incomplete")
    if set(_CURRENT_CODEC.enum_types) != set(enum_wire_ids):
        raise RuntimeError("GA v1 enum type registry is incomplete")
    if set(_CURRENT_CODEC.enum_wire_ids.values()) != set(enum_wire_ids):
        raise RuntimeError("GA v1 enum wire-id registry is incomplete")
    custom_dataclasses = {"task_node"}
    if set(_V1_DATACLASS_ENCODERS) != custom_dataclasses:
        raise RuntimeError("GA v1 dataclass encoder mapping is invalid")
    if set(_V1_DATACLASS_DECODERS) != custom_dataclasses:
        raise RuntimeError("GA v1 dataclass decoder mapping is invalid")
    if not set(_V1_DATACLASS_ENCODERS).issubset(set(wire_ids)):
        raise RuntimeError("GA v1 dataclass encoder mapping contains an unknown type")
    if not set(_V1_DATACLASS_DECODERS).issubset(set(wire_ids)):
        raise RuntimeError("GA v1 dataclass decoder mapping contains an unknown type")
    task_node_fields = tuple(field.name for field in fields(TaskNode))
    if task_node_fields != ("node_id", "dependencies", "budget_cost", "_input"):
        raise RuntimeError("GA v1 task_node source contract changed")


_validate_v1_codec_definition()


__all__ = [
    "CURRENT_DATA_VERSION",
    "CanonicalEnvelope",
    "_decode_enveloped_domain",
    "_decode_step_envelope",
    "canonical_digest",
    "decode_alias",
    "decode_domain",
    "decode_envelope",
    "decode_fact",
    "decode_operation",
    "decode_record",
    "encode_alias",
    "encode_domain",
    "encode_envelope",
    "encode_fact",
    "encode_operation",
    "encode_record",
    "iter_runtime_object_refs",
    "parse_envelope",
    "wire_type_id",
]
