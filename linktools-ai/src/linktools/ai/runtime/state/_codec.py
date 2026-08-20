#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical versioned codecs for Runtime persistence values."""

import base64
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
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

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
    ExecutionEventRecord,
    ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
    ExecutionRecord,
    ExecutionRunSealHead,
    ExecutionStartClaim,
    ExecutionStartUnknownCommit,
    ExecutionCancelRequestCommit,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExecutionEventAppend,
    ExternalCallRecord,
    HistoryQuality,
    IdempotencyRecord,
    InlineContextBlock,
    LoadedContextMessage,
    LoadedModelContext,
    MemoryRecord,
    RecoveryActiveRecord,
    RecoveryAdmissionRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RecoveryConversationIntent,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
    RecoveryIntegrityReport,
    RecoveryStateRecord,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    ResultRecord,
    RuntimePayloadRef,
    SessionRecord,
    SessionForkResultRecord,
    StoredStepSnapshot,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptOwnerDomain,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
    ToolOperationAdmission,
)
from ._plan import RuntimeDomain
from ._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    validate_record_identity,
)

CURRENT_DATA_VERSION = 2
DomainT = TypeVar("DomainT")

_V2_WIRE_TYPES: tuple[tuple[str, type[object]], ...] = (
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
_V2_WIRE_IDS = MappingProxyType(
    {target: wire_id for wire_id, target in _V2_WIRE_TYPES}
)
_V2_DOMAIN_TYPES = MappingProxyType(
    {wire_id: target for wire_id, target in _V2_WIRE_TYPES}
)

_V2_ENUM_WIRE_TYPES: tuple[tuple[str, type[Enum]], ...] = (
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
_V2_ENUM_WIRE_IDS = MappingProxyType(
    {target: wire_id for wire_id, target in _V2_ENUM_WIRE_TYPES}
)
_V2_ENUM_TYPES = MappingProxyType(
    {wire_id: target for wire_id, target in _V2_ENUM_WIRE_TYPES}
)

# Preserve the public v2 manifest names for callers.  Codec decisions below
# use the version-owned mappings through _VersionCodec.
_WIRE_TYPES = _V2_WIRE_TYPES
_WIRE_IDS = _V2_WIRE_IDS
_DOMAIN_TYPES = _V2_DOMAIN_TYPES
_ENUM_WIRE_TYPES = _V2_ENUM_WIRE_TYPES
_ENUM_WIRE_IDS = _V2_ENUM_WIRE_IDS
_ENUM_TYPES = _V2_ENUM_TYPES


@dataclass(frozen=True, slots=True)
class _VersionCodec:
    version: int
    wire_ids: Mapping[type[object], str]
    domain_types: Mapping[str, type[object]]
    enum_wire_ids: Mapping[type[Enum], str]
    enum_types: Mapping[str, type[Enum]]


_V2_CODEC = _VersionCodec(
    2,
    _V2_WIRE_IDS,
    _V2_DOMAIN_TYPES,
    _V2_ENUM_WIRE_IDS,
    _V2_ENUM_TYPES,
)
_VERSION_CODECS: Mapping[int, _VersionCodec] = MappingProxyType(
    {
        2: _V2_CODEC,
    }
)
_CURRENT_CODEC = _VERSION_CODECS[CURRENT_DATA_VERSION]


@dataclass(frozen=True, slots=True)
class CanonicalEnvelope:
    version: int
    value: Mapping[str, JsonValue]


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
    lease = value.get("lease")
    if not isinstance(lease, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    data = value.get("data")
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        record = StoredRecord(
            bytes.fromhex(_string(value, "key")),
            bytes.fromhex(_string(value, "partition")),
            _optional_digest(value.get("scope")),
            _optional_digest(value.get("parent")),
            _string(value, "kind"),
            _string(value, "sort"),
            _optional_string(value.get("state")),
            _integer(value, "storage_version"),
            _optional_string(lease.get("owner")),
            _integer(lease, "fence"),
            _optional_datetime(lease.get("expires_at")),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        validate_record_identity(record)
    except ValueError as error:
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
    data = value.get("data")
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredFact(
            bytes.fromhex(_string(value, "stream")),
            _integer(value, "sequence"),
            bytes.fromhex(_string(value, "owner")),
            _string(value, "kind"),
            _optional_digest(value.get("subject")),
            _optional_string(value.get("state")),
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
    data = value.get("data")
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredOperation(
            bytes.fromhex(_string(value, "key")),
            bytes.fromhex(_string(value, "stream")),
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
    try:
        return StoredAlias(
            bytes.fromhex(_string(value, "alias")),
            bytes.fromhex(_string(value, "record")),
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
    if target is ContinuableSnapshot:
        return "continuable_snapshot"
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
    if target is ContinuableSnapshot:
        return "continuable_snapshot"
    if isinstance(target, type) and issubclass(target, Enum):
        try:
            return codec.enum_wire_ids[target]
        except KeyError as error:
            raise TypeError(f"unsupported enum type: {target.__name__}") from error
    try:
        return codec.wire_ids[target]
    except KeyError as error:
        raise TypeError(f"unsupported domain type: {target.__name__}") from error


def encode_domain(value: DomainT) -> JsonValue:
    """Encode one domain value into the shared canonical JSON representation."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, TaskNode):
        return {
            "$dataclass": _CURRENT_CODEC.wire_ids[TaskNode],
            "fields": {
                "node_id": encode_domain(value.node_id),
                "dependencies": encode_domain(value.dependencies),
                "input": encode_domain(value.input),
                "budget_cost": encode_domain(value.budget_cost),
            },
        }
    if isinstance(value, Enum):
        wire_id = _CURRENT_CODEC.enum_wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"unsupported enum type: {type(value).__name__}")
        return {"$enum": wire_id, "value": value.value}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("domain timestamp must be timezone-aware")
        return {"$datetime": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value):
        if any(field.name.startswith("_") for field in fields(value)):
            raise TypeError("private dataclass fields require an explicit codec")
        wire_id = _CURRENT_CODEC.wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"unsupported dataclass type: {type(value).__name__}")
        return {
            "$dataclass": wire_id,
            "fields": {
                field.name: encode_domain(attrgetter(field.name)(value))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            "$mapping": [[encode_domain(key), encode_domain(item)] for key, item in value.items()]
        }
    if isinstance(value, tuple):
        return {"$tuple": [encode_domain(item) for item in value]}
    if isinstance(value, list):
        return [encode_domain(item) for item in value]
    if isinstance(value, frozenset):
        return {"$frozenset": [encode_domain(item) for item in value]}
    raise TypeError(f"unsupported domain value: {type(value).__name__}")


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
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    expected_type = _codec_wire_type_id(target, codec)
    actual_type = envelope.value.get("type")
    payload = envelope.value.get("payload")
    if actual_type != expected_type or payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if payload_transform is not None:
        payload = payload_transform(payload)
    return _decode_domain(payload, target, codec)


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
            decoded = _decode_domain(value, RuntimePayloadRef, codec)
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[StoredPayload]:
            decoded = _decode_domain(value, StoredPayload, codec)
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[ObjectRef]:
            decoded = _decode_domain(value, ObjectRef, codec)
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
            try:
                decoded = StoredPayload.from_json(value)
            except (TypeError, ValueError):
                decoded = None
            if decoded is not None:
                yield from _iter_runtime_object_refs(decoded, domain, codec)
                return
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
    if (
        isinstance(value, Mapping)
        and value.get("$enum") == codec.enum_wire_ids[RuntimeDomain]
    ):
        enum_value = value.get("value")
        if isinstance(enum_value, str):
            try:
                return RuntimeDomain(enum_value)
            except ValueError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if isinstance(value, str):
        try:
            return RuntimeDomain(value)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_domain(
    value: object,
    target: object,
    codec: _VersionCodec,
) -> object:
    if target is Any or target is object:
        return _decode_any(value, codec)
    origin = get_origin(target)
    arguments = get_args(target)
    if origin is Literal:
        if value not in arguments:
            raise ValueError("literal value is invalid")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _decode_domain(value, candidate, codec)
            except (TypeError, ValueError, KeyError, AIError):
                continue
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value is None:
        return None
    if origin in (list,):
        item_type = arguments[0] if arguments else Any
        if not isinstance(value, list):
            raise TypeError("list value is invalid")
        return [_decode_domain(item, item_type, codec) for item in value]
    if origin in (tuple,):
        if arguments and arguments[-1] is Ellipsis:
            return tuple(
                _decode_domain(item, arguments[0], codec)
                for item in _unwrap_sequence(value)
            )
        return tuple(
            _decode_domain(item, item_type, codec)
            for item, item_type in zip(_unwrap_sequence(value), arguments)
        )
    if origin in (set, frozenset):
        item_type = arguments[0] if arguments else Any
        result = {
            _decode_domain(item, item_type, codec)
            for item in _unwrap_sequence(value)
        }
        return frozenset(result) if origin is frozenset else result
    if isinstance(origin, type) and issubclass(origin, Mapping):
        key_type, item_type = arguments if len(arguments) == 2 else (Any, Any)
        pairs = value.get("$mapping") if isinstance(value, Mapping) else None
        if pairs is None:
            if not isinstance(value, Mapping):
                raise TypeError("mapping value is invalid")
            return {
                _decode_domain(key, key_type, codec): _decode_domain(
                    item,
                    item_type,
                    codec,
                )
                for key, item in value.items()
            }
        return {
            _decode_domain(pair[0], key_type, codec): _decode_domain(
                pair[1],
                item_type,
                codec,
            )
            for pair in pairs
        }
    if target is type(None):
        if value is not None:
            raise TypeError("none value is not null")
        return None
    if isinstance(target, type) and issubclass(target, Enum):
        raw = value.get("value") if isinstance(value, Mapping) and "$enum" in value else value
        return target(raw)
    if target is datetime:
        raw = value.get("$datetime") if isinstance(value, Mapping) else value
        result = datetime.fromisoformat(str(raw))
        if result.tzinfo is None:
            raise ValueError("domain timestamp must be timezone-aware")
        return result
    if target is bytes:
        raw = value.get("$bytes") if isinstance(value, Mapping) else value
        return base64.b64decode(str(raw), validate=True)
    if isinstance(target, type) and is_dataclass(target):
        return _decode_dataclass(value, target, codec)
    if target in (str, bool, int, float):
        if not isinstance(value, target) or target is int and isinstance(value, bool):
            raise TypeError("scalar value has the wrong type")
        return value
    return _decode_any(value, codec)


def _decode_dataclass(
    value: object,
    target: type,
    codec: _VersionCodec,
) -> object:
    if (
        not isinstance(value, Mapping)
        or value.get("$dataclass") != codec.wire_ids.get(target)
    ):
        raise TypeError("dataclass envelope is invalid")
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise TypeError("dataclass fields are invalid")
    hints = get_type_hints(target)
    if target is TaskNode:
        return target(
            str(_decode_domain(raw_fields["node_id"], hints["node_id"], codec)),
            tuple(
                _decode_domain(
                    raw_fields["dependencies"],
                    hints["dependencies"],
                    codec,
                )
            ),
            input=_decode_domain(
                raw_fields["input"],
                hints.get("input", Any),
                codec,
            ),
            budget_cost=int(
                _decode_domain(raw_fields["budget_cost"], hints["budget_cost"], codec)
            ),
        )
    kwargs = {}
    for field in fields(target):
        if not field.init:
            continue
        if field.name not in raw_fields:
            if field.default is not MISSING:
                kwargs[field.name] = field.default
                continue
            if field.default_factory is not MISSING:
                kwargs[field.name] = field.default_factory()
                continue
            raise KeyError(field.name)
        kwargs[field.name] = _decode_domain(
            raw_fields[field.name],
            hints.get(field.name, Any),
            codec,
        )
    return target(**kwargs)


def _decode_any(value: object, codec: _VersionCodec) -> object:
    if isinstance(value, Mapping):
        if "$datetime" in value:
            return _decode_domain(value, datetime, codec)
        if "$bytes" in value:
            return _decode_domain(value, bytes, codec)
        if "$enum" in value:
            name = value.get("$enum")
            target = _enum_type(str(name), codec)
            return target(value.get("value"))
        if "$dataclass" in value:
            target = codec.domain_types.get(str(value.get("$dataclass")))
            if target is None:
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            return _decode_dataclass(value, target, codec)
        if "$mapping" in value:
            return {
                _decode_any(pair[0], codec): _decode_any(pair[1], codec)
                for pair in value["$mapping"]
            }
        if "$tuple" in value:
            return tuple(_decode_any(item, codec) for item in value["$tuple"])
        if "$frozenset" in value:
            return frozenset(
                _decode_any(item, codec) for item in value["$frozenset"]
            )
        return {
            str(key): _decode_any(item, codec) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_any(item, codec) for item in value]
    return value


def _enum_type(wire_id: str, codec: _VersionCodec) -> type[Enum]:
    try:
        return codec.enum_types[wire_id]
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error


def _unwrap_sequence(value: object) -> list[object]:
    if isinstance(value, Mapping) and "$tuple" in value:
        value = value["$tuple"]
    if not isinstance(value, list):
        raise TypeError("sequence value is invalid")
    return value


def _string(value: Mapping[str, object], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("value must be a string or null")
    return value


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _bool(value: Mapping[str, object], key: str) -> bool:
    result = value[key]
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result


def _optional_digest(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("digest must be a string or null")
    result = bytes.fromhex(value)
    if len(result) != 32:
        raise ValueError("digest must contain 32 bytes")
    return result


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string or null")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return result


def _encode_step_envelope(value: object) -> dict[str, JsonValue]:
    if isinstance(value, ContinuableSnapshot):
        return encode_envelope(
            {
                "type": "continuable_snapshot",
                "run_id": value.run_id,
                "step_index": value.step_index,
                "messages": base64.b64encode(
                    ModelMessagesTypeAdapter.dump_json(value.messages)
                ).decode("ascii"),
                "conversation_id": value.conversation_id,
                "parent_run_id": value.parent_run_id,
                "agent_name": value.agent_name,
                "timestamp": value.timestamp.isoformat(),
                "state": value.state,
            }
        )
    return encode_envelope(
        {
            "type": wire_type_id(value),
            "payload": encode_domain(value),
        }
    )


def _decode_step_envelope(value: Mapping[str, JsonValue]) -> object:
    envelope = decode_envelope(value)
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    kind = envelope.value.get("type")
    if kind == "continuable_snapshot":
        try:
            encoded_messages = envelope.value["messages"]
            if not isinstance(encoded_messages, str):
                raise ValueError("step messages are invalid")
            messages = ModelMessagesTypeAdapter.validate_json(
                base64.b64decode(encoded_messages, validate=True)
            )
            timestamp = envelope.value["timestamp"]
            state = envelope.value["state"]
            if not isinstance(timestamp, str) or not isinstance(state, str):
                raise ValueError("step metadata is invalid")
            return ContinuableSnapshot(
                run_id=str(envelope.value["run_id"]),
                step_index=int(envelope.value["step_index"]),
                messages=messages,
                conversation_id=envelope.value.get("conversation_id"),
                parent_run_id=envelope.value.get("parent_run_id"),
                agent_name=envelope.value.get("agent_name"),
                timestamp=datetime.fromisoformat(timestamp),
                state=state,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
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
    payload = envelope.value.get("payload")
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_domain(payload, target, codec)


__all__ = [
    "CanonicalEnvelope",
    "CURRENT_DATA_VERSION",
    "canonical_digest",
    "decode_domain",
    "_decode_enveloped_domain",
    "_decode_step_envelope",
    "decode_alias",
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
