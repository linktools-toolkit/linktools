#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical versioned codecs for Runtime persistence values."""

import base64
import binascii
import hashlib
import json
import math
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from operator import attrgetter
from types import MappingProxyType
from typing import (
    Any,
    ForwardRef,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

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

_V1_ENUM_VALUES: Mapping[str, frozenset[JsonValue]] = MappingProxyType(
    {
        "approval_decision": frozenset(("APPROVE", "DENY")),
        "approval_status": frozenset(
            ("PENDING", "APPROVED", "DENIED", "CANCELLED", "EXPIRED")
        ),
        "evaluation_status": frozenset(
            ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")
        ),
        "execution_event_type": frozenset(
            (
                "EXECUTION_CREATED",
                "EXECUTION_STARTED",
                "EXECUTION_START_UNKNOWN",
                "APPROVAL_REQUESTED",
                "APPROVAL_DECIDED",
                "EXTERNAL_REQUESTED",
                "EXTERNAL_SUPPLIED",
                "CANCEL_REQUESTED",
                "EXECUTION_SUCCEEDED",
                "EXECUTION_FAILED",
                "EXECUTION_CANCELLED",
                "ASSISTANT_PART_COMPLETED",
                "TOOL_CALL_STARTED",
                "TOOL_CALL_FINISHED",
            )
        ),
        "execution_history_state": frozenset(("open", "sealed")),
        "execution_lineage_kind": frozenset(
            ("RUN", "SESSION_RESUME", "RETRY", "FORK", "SUBAGENT")
        ),
        "execution_status": frozenset(
            (
                "PENDING_START",
                "STARTED",
                "FINALIZING",
                "START_UNKNOWN",
                "WAITING_APPROVAL",
                "WAITING_EXTERNAL",
                "CANCELLING",
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
            )
        ),
        "external_call_status": frozenset(
            ("PENDING", "SUPPLIED", "CANCELLED", "EXPIRED")
        ),
        "history_quality": frozenset(("complete", "conservative")),
        "idempotency_status": frozenset(
            ("RESERVED", "STARTED", "START_UNKNOWN", "COMPLETED", "FAILED", "CANCELLED")
        ),
        "operation_kind": frozenset(
            (
                "EXECUTION_START",
                "MODEL",
                "TOOL",
                "APPROVAL",
                "EXTERNAL",
                "BUDGET",
                "RESULT",
                "EVENT",
                "EXECUTION_CANCEL",
                "TASK_CANCEL",
                "SESSION_CREATE",
                "SESSION_FORK",
                "SESSION_UPDATE",
                "SESSION_CLOSE",
                "MEMORY_WRITE",
                "MEMORY_DELETE",
                "TASK_NODE",
                "DOWNLOAD_GRANT",
            )
        ),
        "operation_status": frozenset(
            (
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "EFFECT_UNKNOWN",
                "COMPACTED",
            )
        ),
        "resource_kind": frozenset(
            (
                "SESSION",
                "EXECUTION",
                "TASK_GRAPH",
                "EVALUATION",
                "APPROVAL",
                "EXTERNAL_CALL",
                "ARTIFACT",
                "MEMORY",
                "TOOL_OPERATION",
                "DOWNLOAD_GRANT",
            )
        ),
        "runtime_domain": frozenset(
            ("conversation", "execution", "memory", "artifact", "task", "evaluation", "recovery")
        ),
        "recovery_checkpoint_state": frozenset(
            ("admitted", "active", "waiting", "handoff", "completed")
        ),
        "recovery_handoff_phase": frozenset(
            ("none", "prepared", "conversation_resolved", "execution_committed", "completed")
        ),
        "session_status": frozenset(("OPEN", "CLOSING", "CLOSED", "CLEANUP_REQUIRED")),
        "stop_reason": frozenset(
            ("END_TURN", "REFUSAL", "TURN_LIMIT", "OUTPUT_VALIDATION_FAILED", "CANCELLED", "ERROR")
        ),
        "task_status": frozenset(
            ("PENDING", "READY", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "BLOCKED")
        ),
        "tool_operation_status": frozenset(
            ("PENDING", "CLAIMED", "COMPLETED", "FAILED", "CANCELLED", "EFFECT_UNKNOWN")
        ),
        "transcript_origin": frozenset(("raw", "unknown")),
        "transcript_owner_domain": frozenset(("conversation", "execution", "recovery")),
        "transcript_seek_dimension": frozenset(
            ("message", "session_history_item", "execution_transcript_item")
        ),
    }
)

_V1_SCHEMA_FINGERPRINTS: Mapping[str, str] = MappingProxyType(
    {
        "agent_attempt_claim": "6c70466a08d20f57baf058d8da2a7d2ab3cd738494e45d82111e791fc8beaac2",
        "approval_record": "14d31c99c7e60edebe193b1349c54e5c9a342829c4a8538651d7e9b1249a158d",
        "artifact_record": "9e8c5e9b10ebe4d75605a7dafcb11c7b442694485c5aae21138cb81339992a93",
        "context_projection": "8f77d052bed206e72f13faab7fcb3471cf11358c3b8ac65ffd96d5c13e229bc2",
        "conversation_cursor": "85bc775ad1c18bebf3a9b54ca634d874350f906c36f61c0f1254bff1b3b18888",
        "conversation_history": "e02add0b6455024f22d2599fa4479a5f4ee89fb4af399e70e9e6dfee119eae76",
        "conversation_history_index_node": "0db81616aa96a4e8a78f818334ddbd1b42f3023acb4e1209de8057f76a729729",
        "conversation_history_segment": "62cc29ba7ddb357641b4de91f9168a70ce4f518043444482ee16fefe57a336a5",
        "evaluation_record": "67ac6e9191111abc33d611d2bfd628a4190fa6794c28b8f15f8b878ba46ad345",
        "execution_cancel_request_commit": "1a5f2667e2909eb1a94b0446483ea19cc105131a67d606854bb6e3d11541fa7a",
        "execution_event": "c7d10fa9a15092e7e29c503b938f12358b836ee0d578bd5faff0f0951d3bedf8",
        "execution_event_append": "2db5fe9712b5b6b99fbdf014e3d34f9d20bdcaf419c1f1c616547962d0de5d8d",
        "execution_history_head": "403fad67908ca97614648aa16b681fcecaa205f9ba0847f77ce5dbf666a86b68",
        "execution_history_seal": "dc5a7181d8ee8e23b75390bae449da48eca31f1286159212bb1960d5aad5a0bd",
        "execution_record": "c8fbccbc839df165ee7e8e24c7930c078aeda8029f505c55065041b43e9ac681",
        "execution_run_seal_head": "458b4c2b6722e27546f0948ff87888a07740355c65809c2aff716a7215bcdb0c",
        "execution_start_claim": "a615f3846374e9f5a3e0391188af3644af2d68da55d73980a9f0208fab9f1746",
        "execution_start_reservation": "4c4f6f6bfd6d2d18761bfc14e41facdbdbab354bb006df5a908c00b0be0dbf3e",
        "execution_start_reservation_result": "58e496a28617436532bf22056a34259134d724d9b8c56748e55b7091f8504c7f",
        "execution_start_unknown_commit": "e1e7da1699fdfe2f637225b40a25133205e46c454a56933a68080fc70edfedbf",
        "execution_terminal_commit": "9c9412849f0762ea748066b50f0cfd0d31f2a47dd3155ade8b39dfbca126db51",
        "execution_terminal_commit_result": "4f76947b8c1255a05d0328a7dc9282755a8a7b06ef7f9e5b5cf00e556fd60b1a",
        "external_call_record": "5ebc4362a741d84203885179353de1947ac443212af17f8858008207776fcf1b",
        "idempotency_record": "86071c1bbbd2ed8496e981471f27c4cb3ac68afbdb3d39c645d975403ead58fa",
        "inline_context_block": "e473a751b3324cc5408e9264a1e55de9600bb0a312f0f3b7b7f8fc47afd593a6",
        "loaded_context_message": "14215094b9ec79d9c889e336cf558661635f53590fd60a245de8d34f90007272",
        "loaded_model_context": "676b20054fe7f2e9912d9e7f746e58c65ea9baa4c17d26d269f79862667ec8ec",
        "memory_record": "75b1e858f839ea2cf2fe861c1754cfbc123171a0a1b3a85dd00bc127b8766a58",
        "object_ref": "802ac36862e30f34a601e1db02eaeba787489f66fcce226ee3b81c4ffff55e25",
        "operation_ledger_input": "3f6ebe50100e1235cb5715e45ef05f7e4bdaa79a3ffe5bcc7e79f23eeefdd757",
        "operation_ledger_record": "dd220a701845fc3c1af57f1465733f4caad444dacdad198fc86dc0fdccb926c1",
        "principal": "8ec0c3ae89b05258a228bd01e7941a0d43d35d001f874bf2f8ddd366ef0de78f",
        "recovery_active": "28b8227b09c406d537425b6928b1bbf33008d8c1ef046d3e14c5055ecb3ed664",
        "recovery_admission": "8b1607f342e2d64aeff7947c66514a9c9d5ec8a7609c59c3e4fb0c5c5429b813",
        "recovery_checkpoint": "5e1fb5347b99bfd3fef6dd14e20ec17c65f4ba9dc0ced5cfc2853e3b228b1abc",
        "recovery_conversation_intent": "92dc7357513f07e4724dfe5a6229d4bcbfe34efb984f002538b486d2b0cb4605",
        "recovery_execution_input": "eb0303dd4cdc6295ab841b881073985a9413bb3983e2bd87c7e14dd8a5089c63",
        "recovery_idempotency_input": "74aba8491b8849d02b3930b7745ddb03914df19ee75ceee466c1fcc53965089b",
        "recovery_integrity_report": "287e89f61adc722da8d2365c7ddd8ae7fa96eac37c58bd530c8f860cd48d750d",
        "recovery_state": "533697b347535207c474066973e2cd32da55ae8259c5e5610c80a74657ddaa7e",
        "recovery_terminal_handoff": "dc9cc606232780c624a2b3026e1a2f42e9290dc842749e3fa17e5fdbfb2b8a04",
        "recovery_terminal_outcome": "70249559979a766098cb25acd5977b40665f56dd563b02d4a1892694978fe9ba",
        "resource_ref": "bb37a07e4b78ebe2552774733eb90fc5aa488fd1e262604c578ca87347aaec0d",
        "result_record": "88ba85b7804e77022de1c625ecba090bbeaee815ec355f4dc8787cef95992afe",
        "run_record": "07005d406627ef2a843fc41601c6149336d52d4a5a3b81635fdd1ba58f5c1e9f",
        "runtime_payload_ref": "e681eb4a80417ccdae9ece1f4679dd0601a02dfc5cb89989e2c021c70f63a20e",
        "session_fork_result": "b3a7401efb44d7d9671cc1505dc52d7cb2e04edf2e05d19da2a7f371e201e91b",
        "session_record": "d29d8d76c025ee18d7e93f2315e699618470e096b728e23673f28cad61961ca2",
        "step_event": "c970e92b12e5b3f0fa72afac8e955061c98927dec5805e2acce6e39fc82828e5",
        "stored_payload": "f893cfe67f1722ed1605f28758a868307ef6c253e610a9d2d1853ba422aba7f7",
        "stored_step_snapshot": "b054107b4077a2cf9e948aadb63dae4ae50b8bc73fe72c5908610c332ee70886",
        "task_graph": "6f40954f940848ff244e524865e1a99641a0e9e753f9c0717802bba685595496",
        "task_graph_limits": "2fb7edd03491fffa44c89ea2d13dec396cb0ce345d98832184ac4b720a654585",
        "task_graph_view": "319d351683d8358140409c091ed104559fdf87be156e64e1a5c8ba21d4165fce",
        "task_lease": "b82387e9b2458764179dce21711ccbb184fd5476b3c56de8ae18c8a510657ce1",
        "task_node": "ef43d8cf7d1481b5b50543619d19bf0992bc2ccb2cb8f655c0dffa93e51303a1",
        "task_node_view": "b50443b36f48b177407471ae984f36be27ed944d852647ba2408a954a0e3750b",
        "task_terminal": "fd0d78bd5136f775f0787310fceb2f9ab145423af6071da76ab5effb1b27b211",
        "tool_effect": "2cc6f38e2379596e811db4d877b69bf99bbb83bc0291879116174d8e81683b5f",
        "tool_operation": "7e3639943ad64c296f47b97fe79b3c699d7e1da9487eec4083411d3cd7c7eba7",
        "tool_operation_admission": "86aae6aa8d2bdca393eac38824a00f40e401a1cec8da5bed6d27f4ba254e2642",
        "transcript_chunk": "40b078307e74dc2e3cc0f166aec7c0235fc1354714b3bc05e3458a9b835de1f4",
        "transcript_head": "d8a905a8ed9b406a85b5248d27c96ae325264e80b70f397fbf7c56967b2cdddc",
        "transcript_message_ref": "b7c6c85b153a6da85e5e93d99a20eee1482bd705506aa512e4a41625056fff28",
        "transcript_seek": "faa2c40bb57e757a9291b3d1c124c5a30989fc4fa90e43bbadad25d740c5cbe8",
        "transcript_span_ref": "163f19705426e2c34f8a4e00ed80340a9dd737b450e17b47b4624356fd25e9ff",
        "usage_metrics": "6b537d6b25be39549ab8e1440d13b431c28f734f3f857fabebe67075a9134b00",
    }
)


DataclassEncoder = Callable[
    [object, "_VersionCodec"],
    Mapping[str, JsonValue],
]
DataclassDecoder = Callable[
    [Mapping[str, object], "_VersionCodec"],
    object,
]


@dataclass(frozen=True, slots=True)
class _VersionCodec:
    version: int
    wire_ids: Mapping[type[object], str]
    domain_types: Mapping[str, type[object]]
    enum_wire_ids: Mapping[type[Enum], str]
    enum_types: Mapping[str, type[Enum]]
    schema_fingerprints: Mapping[str, str]
    dataclass_encoders: Mapping[str, DataclassEncoder]
    dataclass_decoders: Mapping[str, DataclassDecoder]
    enum_values: Mapping[str, frozenset[JsonValue]]
    external_schema_types: Mapping[type[object], JsonValue]


_EXECUTION_V1_LEGACY_FIELDS = frozenset(
    {
        "execution_id",
        "tenant_id",
        "session_id",
        "binding_digest",
        "parent_execution_id",
        "root_execution_id",
        "source_execution_id",
        "base_execution_id",
        "lineage_kind",
        "status",
        "revision",
        "event_sequence",
        "agent_run_sequence",
        "error_code",
        "safe_error_details",
        "created_at",
        "updated_at",
        "memory_scope",
        "conversation_step_run_id",
        "result",
    }
)
_EXECUTION_V1_CURRENT_FIELDS = _EXECUTION_V1_LEGACY_FIELDS | frozenset(
    {"planning", "thinking", "binding"}
)
_RECOVERY_EXECUTION_V1_LEGACY_FIELDS = frozenset(
    {
        "user_prompt",
        "principal_id",
        "principal_kind",
        "session_id",
        "memory_scope",
        "agent_id",
        "binding_digest",
        "lineage_kind",
        "parent_execution_id",
        "root_execution_id",
        "source_execution_id",
        "base_execution_id",
        "conversation_step_run_id",
        "idempotency",
    }
)
_RECOVERY_EXECUTION_V1_CURRENT_FIELDS = _RECOVERY_EXECUTION_V1_LEGACY_FIELDS | frozenset(
    {"planning", "thinking", "binding"}
)


def _encode_v1_extended_record(
    value: object,
    legacy_fields: frozenset[str],
    codec: "_VersionCodec",
) -> Mapping[str, JsonValue]:
    planning = attrgetter("planning")(value)
    thinking = attrgetter("thinking")(value)
    binding = attrgetter("binding")(value)
    if not isinstance(planning, bool) or not isinstance(thinking, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if binding is None:
        if planning or thinking:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {
            name: _encode_domain(attrgetter(name)(value), codec)
            for name in legacy_fields
        }
    if not isinstance(binding, AgentBindingSnapshot):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    encoded = {
        name: _encode_domain(attrgetter(name)(value), codec)
        for name in legacy_fields
    }
    encoded["planning"] = planning
    encoded["thinking"] = thinking
    encoded["binding"] = binding.to_payload()
    return encoded


def _decode_v1_extended_record_fields(
    raw_fields: Mapping[str, object],
    target: type[object],
    legacy_fields: frozenset[str],
    current_fields: frozenset[str],
    codec: "_VersionCodec",
) -> dict[str, object]:
    actual = frozenset(raw_fields)
    if actual == legacy_fields:
        planning = False
        thinking = False
        binding = None
    elif actual == current_fields:
        planning = _decode_domain(raw_fields["planning"], bool, codec)
        thinking = _decode_domain(raw_fields["thinking"], bool, codec)
        binding = AgentBindingSnapshot.from_payload(raw_fields["binding"])
    else:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    hints = get_type_hints(target)
    decoded = {
        name: _decode_domain(raw_fields[name], hints[name], codec)
        for name in legacy_fields
    }
    decoded.update(
        planning=planning,
        thinking=thinking,
        binding=binding,
    )
    return decoded


def _encode_v1_execution_record(
    value: object,
    codec: "_VersionCodec",
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ExecutionRecord):
        raise TypeError("V1 execution_record encoder received the wrong type")
    return _encode_v1_extended_record(value, _EXECUTION_V1_LEGACY_FIELDS, codec)


def _decode_v1_execution_record(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
) -> ExecutionRecord:
    decoded = _decode_v1_extended_record_fields(
        raw_fields,
        ExecutionRecord,
        _EXECUTION_V1_LEGACY_FIELDS,
        _EXECUTION_V1_CURRENT_FIELDS,
        codec,
    )
    try:
        return ExecutionRecord(**decoded)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _encode_v1_recovery_execution_input(
    value: object,
    codec: "_VersionCodec",
) -> Mapping[str, JsonValue]:
    if not isinstance(value, RecoveryExecutionInput):
        raise TypeError("V1 recovery_execution_input encoder received the wrong type")
    return _encode_v1_extended_record(
        value,
        _RECOVERY_EXECUTION_V1_LEGACY_FIELDS,
        codec,
    )


def _decode_v1_recovery_execution_input(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
) -> RecoveryExecutionInput:
    decoded = _decode_v1_extended_record_fields(
        raw_fields,
        RecoveryExecutionInput,
        _RECOVERY_EXECUTION_V1_LEGACY_FIELDS,
        _RECOVERY_EXECUTION_V1_CURRENT_FIELDS,
        codec,
    )
    try:
        return RecoveryExecutionInput(**decoded)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _encode_v1_task_node(
    value: object,
    codec: "_VersionCodec",
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskNode):
        raise TypeError("V1 task_node encoder received the wrong type")
    return {
        "node_id": _encode_domain(value.node_id, codec),
        "dependencies": _encode_domain(value.dependencies, codec),
        "input": _encode_domain(value.input, codec),
        "budget_cost": _encode_domain(value.budget_cost, codec),
    }


def _decode_v1_task_node(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
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
        str(_decode_domain(raw_fields["node_id"], str, codec)),
        tuple(
            _decode_domain(
                raw_fields["dependencies"],
                tuple[str, ...],
                codec,
            )
        ),
        input=_decode_domain(
            raw_fields["input"],
            Any,
            codec,
        ),
        budget_cost=int(
            _decode_domain(raw_fields["budget_cost"], int, codec)
        ),
    )


_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {
        "execution_record": _encode_v1_execution_record,
        "recovery_execution_input": _encode_v1_recovery_execution_input,
        "task_node": _encode_v1_task_node,
    }
)
_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(
    {
        "execution_record": _decode_v1_execution_record,
        "recovery_execution_input": _decode_v1_recovery_execution_input,
        "task_node": _decode_v1_task_node,
    }
)


_V1_EXTERNAL_SCHEMA_TYPES: Mapping[type[object], JsonValue] = MappingProxyType(
    {
        IdempotencyTerminalUpdate: (
            "linktools.ai.runtime.state.IdempotencyTerminalUpdate"
        ),
        OperationTerminalUpdate: (
            "linktools.ai.runtime.state.OperationTerminalUpdate"
        ),
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
    schema_fingerprints=_V1_SCHEMA_FINGERPRINTS,
    dataclass_encoders=_V1_DATACLASS_ENCODERS,
    dataclass_decoders=_V1_DATACLASS_DECODERS,
    enum_values=_V1_ENUM_VALUES,
    external_schema_types=_V1_EXTERNAL_SCHEMA_TYPES,
)
_VERSION_CODECS: Mapping[int, _VersionCodec] = MappingProxyType(
    {
        1: _V1_CODEC,
    }
)
_CURRENT_CODEC = _VERSION_CODECS[CURRENT_DATA_VERSION]


def _dataclass_schema_descriptor(
    target: type[object],
    codec: _VersionCodec,
) -> JsonValue:
    if not is_dataclass(target):
        raise TypeError(f"schema target is not a dataclass: {target!r}")
    if target not in codec.wire_ids:
        raise TypeError(f"schema target is not a V1 dataclass: {target!r}")
    try:
        hints = get_type_hints(target)
    except (NameError, TypeError) as error:
        raise TypeError(f"schema annotations are unresolved: {target!r}") from error
    descriptors: list[JsonValue] = []
    for field in fields(target):
        annotation = hints.get(field.name)
        if annotation is None:
            raise TypeError(f"schema annotation is missing: {target!r}.{field.name}")
        descriptors.append(
            {
                "name": field.name,
                "init": field.init,
                "type": _schema_type_descriptor(annotation, codec),
            }
        )
    return {"fields": descriptors}


def _schema_type_descriptor(
    annotation: object,
    codec: _VersionCodec,
) -> JsonValue:
    if isinstance(annotation, ForwardRef):
        if annotation.__forward_arg__ == "JsonValue":
            return "json_value"
        raise TypeError(f"schema annotation is unresolved: {annotation!r}")
    if annotation is Any or annotation is object:
        return "any"
    if annotation is None or annotation is type(None):
        return "none"
    primitive_names = {
        str: "str",
        bool: "bool",
        int: "int",
        float: "float",
        bytes: "bytes",
        datetime: "datetime",
    }
    if annotation in primitive_names:
        return primitive_names[annotation]
    external_descriptor = codec.external_schema_types.get(annotation)
    if external_descriptor is not None:
        return {"external": external_descriptor}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        wire_id = codec.enum_wire_ids.get(annotation)
        if wire_id is None:
            raise TypeError(f"schema enum is not in the V1 codec: {annotation!r}")
        return {"enum": wire_id}
    if isinstance(annotation, type) and is_dataclass(annotation):
        wire_id = codec.wire_ids.get(annotation)
        if wire_id is not None:
            return {"dataclass": wire_id}
        raise TypeError(f"schema dataclass is not in the V1 codec: {annotation!r}")

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return {
            "literal": [_schema_literal(value, codec) for value in arguments]
        }
    if origin in (Union, types.UnionType):
        values = [_schema_type_descriptor(value, codec) for value in arguments]
        return {"union": sorted(values, key=canonical_json_bytes)}
    if origin is list:
        if len(arguments) != 1:
            raise TypeError(f"schema list annotation is incomplete: {annotation!r}")
        return {"list": _schema_type_descriptor(arguments[0], codec)}
    if origin is tuple:
        if not arguments:
            raise TypeError(f"schema tuple annotation is incomplete: {annotation!r}")
        if arguments[-1] is Ellipsis:
            if len(arguments) != 2:
                raise TypeError(f"schema tuple annotation is invalid: {annotation!r}")
            return {"tuple_var": _schema_type_descriptor(arguments[0], codec)}
        return {
            "tuple": [
                _schema_type_descriptor(value, codec) for value in arguments
            ]
        }
    if origin is frozenset:
        if len(arguments) != 1:
            raise TypeError(
                f"schema frozenset annotation is incomplete: {annotation!r}"
            )
        return {"frozenset": _schema_type_descriptor(arguments[0], codec)}
    if origin is dict or origin is Mapping:
        if len(arguments) != 2:
            raise TypeError(
                f"schema mapping annotation is incomplete: {annotation!r}"
            )
        return {
            "mapping": [
                _schema_type_descriptor(arguments[0], codec),
                _schema_type_descriptor(arguments[1], codec),
            ]
        }
    raise TypeError(f"schema annotation is not representable: {annotation!r}")


def _schema_literal(value: object, codec: _VersionCodec) -> JsonValue:
    if isinstance(value, Enum):
        wire_id = codec.enum_wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"schema literal enum is not in V1: {value!r}")
        return {"$enum": wire_id, "value": value.value}
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError("GA v1 schema literal requires finite floats")
        return value
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"schema literal is not canonical: {value!r}")


def _dataclass_schema_fingerprint(
    target: type[object],
    codec: _VersionCodec,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_dataclass_schema_descriptor(target, codec))
    ).hexdigest()


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


def _encode_domain(value: object, codec: _VersionCodec) -> JsonValue:
    if isinstance(value, Enum):
        wire_id = codec.enum_wire_ids.get(type(value))
        if wire_id is None:
            raise TypeError(f"unsupported enum type: {type(value).__name__}")
        allowed = codec.enum_values.get(wire_id)
        if allowed is None:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        try:
            frozen = value.value in allowed
        except TypeError:
            frozen = False
        if not frozen:
            raise AIError(
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                "current enum cannot be written as GA v1",
            )
        return {"$enum": wire_id, "value": value.value}
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
            encoded_fields = encoder(value, codec)
        else:
            expected_fingerprint = codec.schema_fingerprints.get(wire_id)
            if expected_fingerprint is None:
                raise AIError(
                    ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                    "current dataclass schema cannot be written as GA v1",
                )
            try:
                actual_fingerprint = _dataclass_schema_fingerprint(type(value), codec)
            except (TypeError, ValueError) as error:
                raise AIError(
                    ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                    "current dataclass schema cannot be written as GA v1",
                ) from error
            if actual_fingerprint != expected_fingerprint:
                raise AIError(
                    ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                    "current dataclass schema cannot be written as GA v1",
                )
            if any(field.name.startswith("_") for field in fields(value)):
                raise TypeError("private dataclass fields require an explicit codec")
            encoded_fields = {
                field.name: _encode_domain(attrgetter(field.name)(value), codec)
                for field in fields(value)
            }
        wire: dict[str, JsonValue] = {
            "$dataclass": wire_id,
            "fields": dict(encoded_fields),
        }
        try:
            _decode_dataclass(wire, type(value), codec)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                raise
            raise TypeError(
                f"{type(value).__name__} does not match its declared GA v1 schema"
            ) from error
        except (TypeError, ValueError, KeyError) as error:
            raise TypeError(
                f"{type(value).__name__} does not match its declared GA v1 schema"
            ) from error
        return wire
    if isinstance(value, Mapping):
        encoded_pairs: list[tuple[bytes, JsonValue, JsonValue]] = []
        for key, item in value.items():
            encoded_key = _encode_domain(key, codec)
            encoded_item = _encode_domain(item, codec)
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
        return {"$tuple": [_encode_domain(item, codec) for item in value]}
    if isinstance(value, list):
        return [_encode_domain(item, codec) for item in value]
    if isinstance(value, frozenset):
        encoded_items = [
            (canonical_json_bytes(encoded_item), encoded_item)
            for item in value
            for encoded_item in (_encode_domain(item, codec),)
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
) -> object:
    if target is Any or target is object:
        return _decode_any(value, codec)
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
                return _decode_domain(value, candidate, codec)
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
        return [_decode_domain(item, item_type, codec) for item in value]
    if origin is tuple:
        items = _unwrap_tagged_list(value, "$tuple")
        if arguments and arguments[-1] is Ellipsis:
            return tuple(
                _decode_domain(item, arguments[0], codec)
                for item in items
            )
        if len(items) != len(arguments):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(
            _decode_domain(item, item_type, codec)
            for item, item_type in zip(items, arguments, strict=True)
        )
    if target is set or origin is set:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            "GA v1 does not support set values",
        )
    if target is frozenset or origin is frozenset:
        item_type = arguments[0] if arguments else Any
        return _decode_frozenset_items(value, item_type, codec)
    if isinstance(origin, type) and issubclass(origin, Mapping):
        key_type, item_type = arguments if len(arguments) == 2 else (Any, Any)
        return _decode_mapping_items(value, key_type, item_type, codec)
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
        return _decode_dataclass(value, target, codec)
    if target is float:
        if not isinstance(value, float):
            raise TypeError("scalar value has the wrong type")
        return _require_finite_wire_float(value)
    if target in (str, bool, int):
        if not isinstance(value, target) or target is int and isinstance(value, bool):
            raise TypeError("scalar value has the wrong type")
        return value
    return _decode_any(value, codec)


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
    allowed = codec.enum_values.get(expected_wire_id)
    if allowed is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    try:
        valid = raw in allowed
    except TypeError:
        valid = False
    if not valid:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return target(raw)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error


def _decode_dataclass(
    value: object,
    target: type,
    codec: _VersionCodec,
) -> object:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
    decoder = codec.dataclass_decoders.get(wire_id)
    if decoder is not None:
        return decoder(raw_fields, codec)
    expected_fingerprint = codec.schema_fingerprints.get(wire_id)
    if expected_fingerprint is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    try:
        actual_fingerprint = _dataclass_schema_fingerprint(target, codec)
    except TypeError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    if actual_fingerprint != expected_fingerprint:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    hints = get_type_hints(target)
    declared_fields = tuple(fields(target))
    if set(raw_fields.keys()) != {field.name for field in declared_fields}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decoded_fields: dict[str, object] = {}
    for field in declared_fields:
        decoded_fields[field.name] = _decode_domain(
            raw_fields[field.name],
            hints.get(field.name, Any),
            codec,
        )
    kwargs: dict[str, object] = {}
    post_init_fields: dict[str, object] = {}
    for field in declared_fields:
        if field.init:
            kwargs[field.name] = decoded_fields[field.name]
        else:
            post_init_fields[field.name] = decoded_fields[field.name]
    try:
        result = target(**kwargs)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    for field_name, expected in post_init_fields.items():
        try:
            actual = attrgetter(field_name)(result)
        except AttributeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if actual != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_any(value: object, codec: _VersionCodec) -> object:
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
            return _decode_dataclass(value, target, codec)
        if tag == "$mapping":
            return _decode_mapping_items(value, Any, Any, codec)
        if tag == "$tuple":
            return tuple(
                _decode_any(item, codec)
                for item in _unwrap_tagged_list(value, "$tuple")
            )
        return _decode_frozenset_items(value, Any, codec)
    if isinstance(value, list):
        return [_decode_any(item, codec) for item in value]
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
) -> frozenset[object]:
    items = _unwrap_tagged_list(value, "$frozenset")
    decoded_items: list[object] = []
    previous_wire: bytes | None = None
    for item in items:
        decoded_item = _decode_domain(item, item_type, codec)
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
        decoded_key = _decode_domain(pair[0], key_type, codec)
        try:
            hash(decoded_key)
        except TypeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if decoded_key in result:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result[decoded_key] = _decode_domain(pair[1], item_type, codec)
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
        raise ValueError("digest must be a string or null")
    return _digest_wire(value)


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string or null")
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
            "payload": encode_domain(value),
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
    return _decode_domain(payload, target, codec)


def _validate_v1_codec_definition() -> None:
    if CURRENT_DATA_VERSION != 1 or set(_VERSION_CODECS) != {1}:
        raise RuntimeError("GA v1 codec registry is invalid")
    if _CURRENT_CODEC is not _VERSION_CODECS[1]:
        raise RuntimeError("GA v1 current codec is invalid")
    wire_ids = tuple(wire_id for wire_id, _target in _V1_WIRE_TYPES)
    enum_wire_ids = tuple(wire_id for wire_id, _target in _V1_ENUM_WIRE_TYPES)
    dataclass_wire_ids = {
        wire_id
        for wire_id, target in _V1_WIRE_TYPES
        if is_dataclass(target)
    }
    enum_value_ids = set(_V1_ENUM_VALUES)
    if len(wire_ids) != len(set(wire_ids)):
        raise RuntimeError("GA v1 wire ids are not unique")
    if len(enum_wire_ids) != len(set(enum_wire_ids)):
        raise RuntimeError("GA v1 enum wire ids are not unique")
    if set(_V1_SCHEMA_FINGERPRINTS) != dataclass_wire_ids:
        raise RuntimeError(
            "GA v1 schema fingerprint manifest does not match domain types"
        )
    if set(_V1_ENUM_VALUES) != set(enum_wire_ids):
        raise RuntimeError("GA v1 enum value manifest does not match enum types")
    custom_dataclasses = {
        "execution_record",
        "recovery_execution_input",
        "task_node",
    }
    if set(_V1_DATACLASS_ENCODERS) != custom_dataclasses:
        raise RuntimeError("GA v1 dataclass encoder manifest is invalid")
    if set(_V1_DATACLASS_DECODERS) != custom_dataclasses:
        raise RuntimeError("GA v1 dataclass decoder manifest is invalid")
    if not set(_V1_DATACLASS_ENCODERS).issubset(dataclass_wire_ids):
        raise RuntimeError("GA v1 dataclass encoder manifest contains an unknown type")
    if not set(_V1_DATACLASS_DECODERS).issubset(dataclass_wire_ids):
        raise RuntimeError("GA v1 dataclass decoder manifest contains an unknown type")
    if enum_value_ids != set(enum_wire_ids):
        raise RuntimeError("GA v1 enum value manifest is incomplete")
    try:
        _schema_type_descriptor(set[str], _V1_CODEC)
    except TypeError:
        pass
    else:
        raise RuntimeError("GA v1 schema descriptor unexpectedly supports set")


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
