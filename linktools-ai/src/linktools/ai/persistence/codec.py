#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical versioned codecs for Runtime persistence values."""

from ..storage import StorageRetentionMode
from ._archive import ArchiveRoot, ArchiveSlice, decode_archive_scope, validate_archive_content_identity, validate_archive_group_id
from ._message_root import NativeUserInputReference, NativeMessageRoot, NativeMessageSlice, NativeTextReference, NativeToolResultReference, NativeToolArgumentsReference

import base64
import binascii
import json
import math
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
from ..steps import NativeStepSource
from ..execution import ExecutionMetadataRecord, ExecutionResultRecord, ExecutionRetentionRecord, ExecutionTimelineRecord, ExecutionView
from datetime import datetime
from enum import Enum
from functools import lru_cache
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

from linktools.core import environ
from pydantic_ai.messages import ModelRequest, ModelResponse

from ..asset import AssetVersionRef
from ..spec import CapabilityRef, validate_logical_id
from ..core import validate_resource_id
from ..evaluation import (
    CandidateContract,
    CandidateSlotRef,
    CandidateSummary,
    CaseContract,
    CaseRef,
    ComparisonReadCutoff,
    ComparisonReport,
    ContractDifference,
    DatasetContract,
    DatasetRef,
    DimensionBound,
    DimensionContract,
    EvaluationManifest,
    EvaluationPolicy,
    EvaluationReadCutoff,
    EvaluationReport,
    EvidenceBundle,
    EvidencePolicy,
    EvidenceRef,
    ExecutionSubjectRef,
    ExecutionTargetEvidence,
    FailureCount,
    GatePolicy,
    GraphInputContract,
    GraphOutputItem,
    GraphSubjectRef,
    GraphTargetContract,
    GraphTargetEvidence,
    InlineValue,
    LabelProvenance,
    PairedDimensionSummary,
    PairedScoreRow,
    ScoreAttemptView,
    ScoreComparisonSelection,
    ScoreSelection,
    ScorerContract,
    ScorerSummary,
    SlotDispositionView,
    TargetTrialRef,
    TaskCaseInput,
    TrialPlan,
    TrialView,
    ScoreBundle,
    EvidenceAttachmentRef,
    ModelUsage,
)
from ._evaluation_records import (
    EvaluationLaunchIntent, EvaluationTrialEvidence, EvaluationSlotDisposition, EvaluationHumanDecision,
    EvaluationDatasetRecord, EvaluationCaseRecord, EvaluationTombstone, EvaluationContentTombstone, EvaluationCleanupRecord,
)

from ..agent import AgentInputCaptureRef, AgentBindingContract
from ..core import (
    BudgetUsage,
    RunBudget,
    RUNTIME_OBJECT_STORE_ID,
    ApprovalDecision,
    ApprovalStatus,
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
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..storage import ObjectRef, StoredPayload
from ..task import (
    TaskBindingContract,
    TaskExpanderRef,
    TaskRef,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphSubmission,
    TaskSubmissionRef,
    TaskGraphLimits,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
    TaskResultRef,
    TaskNodeResultRef,
    TaskInvocationInputRef,
    TaskInvocationInputContract,
    TaskDependencyCapture,
    TaskDependencyState,
    TaskGraphTemplate,
    TaskGraphTemplateRef,
    TaskGraphCaptureRef,
    TaskTerminalRecord,
)
from ..model.messages import decode_model_messages, encode_model_messages
from ..task import AgentTaskNode
from .contracts import (
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
    ExecutionHistorySealRecord,
    ExecutionRecord,
    ExecutionControlRecord,
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
    ArchivedContextBlock,
    MemoryRecord,
    ModelClientBodyRef,
    ModelClientRequestRef,
    ModelClientResponseRef,
    ModelInteractionRecord,
    StreamAttemptRecord,
    OperationTerminalUpdate,
    PendingDeferredCall,
    PendingToolContinuation,
    ExecutionRecoveryDetail,
    ExecutionRecoveryState,
    RecoveryConversationIntent,
    RecoveryHandoffPhase,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    RepositoryInstructionBarrier,
    ResultRecord,
    RuntimePayloadRef,
    SessionRecord,
    SessionTurnCommit,
    StoredAgentRunCheckpoint,
    StoredUserInput,
    TaskPreparedInputRecord,
    ToolOperationAdmission,
    ToolOperationRecord,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptMessageRef,
    TranscriptOrigin,
    TranscriptOwnerDomain,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
)
from ._budget_records import BudgetModelReservation, BudgetToolReservation
from ._plan import PersistenceDomain
from ._step_contracts import (
    AgentRunRecord,
    StepEvent,
)
from .state import (
    STREAM_KEY_MAX_BYTES,
    OPERATION_KEY_MAX_BYTES,
    validate_stream_key,
    RELATION_KEY_MAX_BYTES,
    validate_relation_key,
    validate_record_key,
    validate_parent_key,
    RECORD_KEY_MAX_BYTES,
    PARENT_KEY_MAX_BYTES,
    StoredFact,
    StoredOperation,
    StoredRecord,
    validate_sort_key,
)

CURRENT_DATA_VERSION = 1
_TASK_NODE_WIRE_ID = "task_node"
_TASK_NODE_VIEW_WIRE_ID = "task_node_view"
DomainT = TypeVar("DomainT")
_logger = environ.get_logger("ai.runtime.state.codec")

_V1_WIRE_TYPES: tuple[tuple[str, type[object]], ...] = (
    ("run_budget", RunBudget),
    ("budget_usage", BudgetUsage),
    ("budget_model_reservation", BudgetModelReservation),
    ("budget_tool_reservation", BudgetToolReservation),
    ("evaluation_candidate_slot_ref", CandidateSlotRef),
    ("evaluation_score_selection", ScoreSelection),
    ("evaluation_score_comparison_selection", ScoreComparisonSelection),
    ("evaluation_dimension_bound", DimensionBound),
    ("evaluation_gate_policy", GatePolicy),
    ("evaluation_read_cutoff", EvaluationReadCutoff),
    ("evaluation_comparison_read_cutoff", ComparisonReadCutoff),
    ("evaluation_contract_difference", ContractDifference),
    ("evaluation_failure_count", FailureCount),
    ("evaluation_candidate_summary", CandidateSummary),
    ("evaluation_scorer_summary", ScorerSummary),
    ("evaluation_report", EvaluationReport),
    ("evaluation_paired_score_row", PairedScoreRow),
    ("evaluation_paired_dimension_summary", PairedDimensionSummary),
    ("evaluation_comparison_report", ComparisonReport),
    ("evaluation_slot_disposition_view", SlotDispositionView),
    ("evaluation_trial_view", TrialView),
    ("evaluation_score_attempt_view", ScoreAttemptView),
    ("evaluation_execution_subject_ref", ExecutionSubjectRef),
    ("evaluation_graph_subject_ref", GraphSubjectRef),
    ("evaluation_evidence_ref", EvidenceRef),
    ("evaluation_execution_target_evidence", ExecutionTargetEvidence),
    ("evaluation_graph_output_item", GraphOutputItem),
    ("evaluation_graph_target_evidence", GraphTargetEvidence),
    ("evaluation_evidence_bundle", EvidenceBundle),
    ("evaluation_case_ref", CaseRef),
    ("evaluation_dataset_ref", DatasetRef),
    ("evaluation_inline_value", InlineValue),
    ("evaluation_label_provenance", LabelProvenance),
    ("evaluation_task_case_input", TaskCaseInput),
    ("evaluation_graph_input_contract", GraphInputContract),
    ("evaluation_dimension_contract", DimensionContract),
    ("evaluation_evidence_policy", EvidencePolicy),
    ("evaluation_policy", EvaluationPolicy),
    ("evaluation_case_contract", CaseContract),
    ("evaluation_dataset_contract", DatasetContract),
    ("evaluation_graph_target_contract", GraphTargetContract),
    ("evaluation_candidate_contract", CandidateContract),
    ("evaluation_scorer_contract", ScorerContract),
    ("evaluation_target_trial_ref", TargetTrialRef),
    ("evaluation_trial_plan", TrialPlan),
    ("evaluation_manifest", EvaluationManifest),
    ("evaluation_launch_intent", EvaluationLaunchIntent),
    ("evaluation_trial_evidence", EvaluationTrialEvidence),
    ("evaluation_slot_disposition", EvaluationSlotDisposition),
    ("approval_record", ApprovalRecord),
    ("agent_attempt_claim", AgentAttemptClaim),
    ("artifact_record", ArtifactRecord),
    ("conversation_cursor", ConversationCursor),
    ("conversation_history_index_node", ConversationHistoryIndexNodeRecord),
    ("conversation_history", ConversationHistoryRecord),
    ("conversation_history_segment", ConversationHistorySegmentRef),
    ("context_projection", ContextProjection),
    ("error_diagnostics", ErrorDiagnostics),
    ("evaluation_record", EvaluationRecord),
    ("evaluation_tombstone", EvaluationTombstone),
    ("evaluation_content_tombstone", EvaluationContentTombstone),
    ("evaluation_cleanup_record", EvaluationCleanupRecord),
    ("evaluation_attachment_ref", EvidenceAttachmentRef),
    ("evaluation_dataset_record", EvaluationDatasetRecord),
    ("evaluation_case_record", EvaluationCaseRecord),
    ("evaluation_model_usage", ModelUsage),
    ("evaluation_human_decision", EvaluationHumanDecision),
    ("execution_event", ExecutionEventRecord),
    ("execution_history_seal", ExecutionHistorySealRecord),
    ("execution_record", ExecutionRecord),
    ("native_step_source", NativeStepSource),
    ("execution_run_seal_head", ExecutionRunSealHead),
    ("model_client_body_ref", ModelClientBodyRef),
    ("model_client_request_ref", ModelClientRequestRef),
    ("model_client_response_ref", ModelClientResponseRef),
    ("model_interaction", ModelInteractionRecord),
    ("stream_attempt", StreamAttemptRecord),
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
    ("pending_deferred_call", PendingDeferredCall),
    ("pending_tool_continuation", PendingToolContinuation),
    ("execution_recovery_detail", ExecutionRecoveryDetail),
    ("recovery_conversation_intent", RecoveryConversationIntent),
    ("recovery_terminal_handoff", RecoveryTerminalHandoff),
    ("recovery_terminal_outcome", RecoveryTerminalOutcome),
    ("repository_instruction_barrier", RepositoryInstructionBarrier),
    ("resource_ref", ResourceRef),
    ("result_record", ResultRecord),
    ("session_record", SessionRecord),
    ("session_turn_commit", SessionTurnCommit),
    ("stored_agent_run_checkpoint", StoredAgentRunCheckpoint),
    ("object_ref", ObjectRef),
    ("stored_payload", StoredPayload),
    ("history_quality", HistoryQuality),
    ("inline_context_block", InlineContextBlock),
    ("archived_context_block", ArchivedContextBlock),
    ("runtime_payload_ref", RuntimePayloadRef),
    ("stored_user_input", StoredUserInput),
    ("task_binding_contract", TaskBindingContract),
    ("archive_root", ArchiveRoot),
    ("archive_slice", ArchiveSlice),
    ("native_message_root", NativeMessageRoot),
    ("native_message_slice", NativeMessageSlice),
    ("transcript_chunk", TranscriptChunk),
    ("transcript_head", TranscriptHeadRecord),
    ("transcript_message_ref", TranscriptMessageRef),
    ("transcript_origin", TranscriptOrigin),
    ("transcript_owner_domain", TranscriptOwnerDomain),
    ("transcript_seek_dimension", TranscriptSeekDimension),
    ("transcript_seek", TranscriptSeekRecord),
    ("transcript_span_ref", TranscriptSpanRef),
    ("capability_ref", CapabilityRef),
    ("tool_operation_admission", ToolOperationAdmission),
    ("runtime_domain", PersistenceDomain),
    ("task_graph", TaskGraph),
    ("task_graph_admission", TaskGraphAdmission),
    ("task_graph_submission", TaskGraphSubmission),
    ("task_submission_ref", TaskSubmissionRef),
    ("task_graph_limits", TaskGraphLimits),
    ("task_graph_view", TaskGraphView),
    ("task_lease", TaskLease),
    (_TASK_NODE_WIRE_ID, TaskNode),
    ("task_ref", TaskRef),
    ("task_expander_ref", TaskExpanderRef),
    (_TASK_NODE_VIEW_WIRE_ID, TaskNodeView),
    ("task_result", TaskResultRecord),
    ("task_result_ref", TaskResultRef),
    ("task_node_result_ref", TaskNodeResultRef),
    ("task_invocation_input_ref", TaskInvocationInputRef),
    ("task_invocation_input_contract", TaskInvocationInputContract),
    ("task_dependency_capture", TaskDependencyCapture),
    ("task_dependency_state", TaskDependencyState),
    ("task_graph_template", TaskGraphTemplate),
    ("task_graph_template_ref", TaskGraphTemplateRef),
    ("task_graph_capture_ref", TaskGraphCaptureRef),
    ("agent_input_capture_ref", AgentInputCaptureRef),
    ("task_prepared_input", TaskPreparedInputRecord),
    ("task_terminal", TaskTerminalRecord),
    ("tool_operation", ToolOperationRecord),
    ("usage_metrics", UsageMetrics),
    ("agent_run_record", AgentRunRecord),
    ("step_event", StepEvent),
)
_V1_WIRE_IDS = MappingProxyType(
    {**{target: wire_id for wire_id, target in _V1_WIRE_TYPES}, AgentTaskNode: _TASK_NODE_WIRE_ID}
)
_V1_DOMAIN_TYPES = MappingProxyType(
    {wire_id: target for wire_id, target in _V1_WIRE_TYPES}
)

_V1_ENUM_WIRE_TYPES: tuple[tuple[str, type[Enum]], ...] = (
    ("approval_decision", ApprovalDecision),
    ("approval_status", ApprovalStatus),
    ("execution_event_type", ExecutionEventType),
    ("execution_lineage_kind", ExecutionLineageKind),
    ("execution_status", ExecutionStatus),
    ("external_call_status", ExternalCallStatus),
    ("history_quality", HistoryQuality),
    ("idempotency_status", IdempotencyStatus),
    ("operation_kind", OperationKind),
    ("operation_status", OperationStatus),
    ("resource_kind", ResourceKind),
    ("runtime_domain", PersistenceDomain),
    ("runtime_retention_mode", StorageRetentionMode),
    ("execution_recovery_state", ExecutionRecoveryState),
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
    external_schema_types: frozenset[type[object]]


def _encode_v1_task_node_fields(
    value: TaskNode,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    fields: dict[str, JsonValue] = {
        "node_id": _encode_domain(value.node_id, codec, persisted=persisted),
        "task": _encode_domain(value.task, codec, persisted=persisted),
        "dependencies": _encode_domain(
            value.dependencies, codec, persisted=persisted
        ),
        "input": _encode_domain(value.input, codec, persisted=persisted),
        "input_semantics": "agent-task-input-v1" if isinstance(value, AgentTaskNode) else "json-v1",
        "budget_cost": _encode_domain(
            value.budget_cost, codec, persisted=persisted
        ),
        "expander": _encode_domain(value.expander, codec, persisted=persisted),
        "dependency_policy": value.dependency_policy,
        "failure_policy": value.failure_policy,
    }
    if value.output_type is not None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value.input_refs:
        fields["input_refs"] = [
            [
                name,
                {"node_id": reference.node_id} if isinstance(reference, TaskNodeResultRef) else {
                    "namespace": reference.namespace,
                    "tenant_id": reference.tenant_id,
                    "graph_id": reference.graph_id,
                    "node_id": reference.node_id,
                    "result_digest": reference.result_digest,
                },
            ]
            for name, reference in sorted(value.input_refs.items())
        ]
    if value.original_input is not None:
        fields["original_input"] = _encode_domain(value.original_input, codec, persisted=persisted)
    if value.input_capture is not None:
        fields["input_capture"] = _encode_domain(value.input_capture, codec, persisted=persisted)
    if value.timeout_seconds is not None:
        fields["timeout_seconds"] = value.timeout_seconds
    if value.max_attempts != 1:
        fields["max_attempts"] = value.max_attempts
    if value.retry_delay_seconds != 0:
        fields["retry_delay_seconds"] = value.retry_delay_seconds
    if value.output_contract is not None:
        fields["output_contract"] = dict(value.output_contract)
    if value.effect_policy != "none":
        fields["effect_policy"] = value.effect_policy
    if value.reconcile:
        fields["reconcile"] = True
    return fields


def _encode_v1_task_node(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskNode):
        raise TypeError("V1 task_node encoder received the wrong type")
    if isinstance(value, AgentTaskNode):
        value.to_mapping()
    return _encode_v1_task_node_fields(value, codec, persisted)


def _decode_v1_task_node(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> TaskNode:
    required = frozenset(
        {
            "node_id",
            "task",
            "dependencies",
            "input",
            "input_semantics",
            "budget_cost",
            "expander",
            "dependency_policy",
            "failure_policy",
        }
    )
    keys = set(raw_fields)
    if not required.issubset(keys):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_refs = raw_fields.get("input_refs", [])
    if not isinstance(raw_refs, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    input_refs: dict[str, TaskResultRef | TaskNodeResultRef] = {}
    for item in raw_refs:
        if not isinstance(item, list) or len(item) != 2:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        name, raw_reference = item
        if not isinstance(name, str) or not isinstance(raw_reference, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(raw_reference) == {"node_id"}:
            if name in input_refs:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                input_refs[name] = TaskNodeResultRef(raw_reference["node_id"])
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            continue
        _require_required_keys(
            raw_reference,
            frozenset(
                {"namespace", "tenant_id", "graph_id", "node_id", "result_digest"}
            ),
        )
        try:
            if name in input_refs:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            input_refs[name] = TaskResultRef(
                cast(str, _decode_domain(raw_reference["namespace"], str, codec)),
                cast(str, _decode_domain(raw_reference["tenant_id"], str, codec)),
                cast(str, _decode_domain(raw_reference["graph_id"], str, codec)),
                cast(str, _decode_domain(raw_reference["node_id"], str, codec)),
                cast(str, _decode_domain(raw_reference["result_digest"], str, codec)),
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    timeout_seconds = raw_fields.get("timeout_seconds")
    if timeout_seconds is not None and type(timeout_seconds) is not float:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    max_attempts = raw_fields.get("max_attempts", 1)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    retry_delay_seconds = raw_fields.get("retry_delay_seconds", 0.0)
    if type(retry_delay_seconds) is not float:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    output_contract = raw_fields.get("output_contract")
    if output_contract is not None and not isinstance(output_contract, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    optional = frozenset(
        {
            "input_refs",
            "input_capture",
            "original_input",
            "timeout_seconds",
            "max_attempts",
            "retry_delay_seconds",
            "output_contract",
            "effect_policy",
            "reconcile",
        }
    )
    if keys - required - optional:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    effect_policy = raw_fields.get("effect_policy", "none")
    reconcile = raw_fields.get("reconcile", False)
    dependency_policy = raw_fields["dependency_policy"]
    failure_policy = raw_fields["failure_policy"]
    if (
        not isinstance(effect_policy, str)
        or not isinstance(reconcile, bool)
        or not isinstance(dependency_policy, str)
        or not isinstance(failure_policy, str)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    semantics = raw_fields["input_semantics"]
    if not isinstance(semantics, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if semantics not in {"json-v1", "agent-task-input-v1"}:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    node_type = AgentTaskNode if semantics == "agent-task-input-v1" else TaskNode
    node = node_type.from_resolved(
        cast(str, _decode_domain(raw_fields["node_id"], str, codec, persisted=persisted)),
        tuple(
            _decode_domain(
                raw_fields["dependencies"],
                tuple[str, ...],
                codec,
                persisted=persisted,
            )
        ),
        task=cast(
            TaskRef | None,
            _decode_domain(
                raw_fields["task"],
                TaskRef | None,
                codec,
                persisted=persisted,
            ),
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
        expander=cast(
            TaskExpanderRef | None,
            _decode_domain(
                raw_fields["expander"],
                TaskExpanderRef | None,
                codec,
                persisted=persisted,
            ),
        ),
        input_refs=input_refs,
        input_capture=_decode_domain(raw_fields.get("input_capture"), TaskInvocationInputRef | None, codec, persisted=persisted),
        original_input=_decode_domain(raw_fields.get("original_input"), Mapping[str, JsonValue] | None, codec, persisted=persisted),
        timeout_seconds=cast("float | None", timeout_seconds),
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        output_contract=cast(
            Mapping[str, JsonValue] | None,
            output_contract,
        ),
        effect_policy=effect_policy,
        reconcile=reconcile,
        dependency_policy=dependency_policy,
        failure_policy=failure_policy,
    )
    if isinstance(node, AgentTaskNode):
        node.to_mapping()
    return node


def _encode_v1_task_graph_view(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskGraphView):
        raise TypeError("V1 task_graph_view encoder received the wrong type")
    fields: dict[str, JsonValue] = {
        "graph_id": _encode_domain(value.graph_id, codec, persisted=persisted),
        "status": _encode_domain(value.status, codec, persisted=persisted),
    }
    if not persisted:
        fields["nodes"] = _encode_domain(value.nodes, codec, persisted=persisted)
    return fields


def _decode_v1_task_graph_view(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> TaskGraphView:
    expected = frozenset({"graph_id", "status"}) if persisted else frozenset(
        {"graph_id", "status", "nodes"}
    )
    _require_required_keys(raw_fields, expected)
    nodes = () if persisted else tuple(
        _decode_domain(raw_fields["nodes"], tuple[TaskNode, ...], codec, persisted=persisted)
    )
    return TaskGraphView(
        cast(str, _decode_domain(raw_fields["graph_id"], str, codec, persisted=persisted)),
        cast(
            TaskStatus,
            _decode_domain(raw_fields["status"], TaskStatus, codec, persisted=persisted),
        ),
        nodes,
    )


def _encode_v1_task_node_view(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, TaskNodeView):
        raise TypeError("V1 task_node_view encoder received the wrong type")
    fields: dict[str, JsonValue] = {
        "graph_id": _encode_domain(value.graph_id, codec, persisted=persisted),
        "node_id": _encode_domain(value.node_id, codec, persisted=persisted),
        "status": _encode_domain(value.status, codec, persisted=persisted),
        "execution_id": _encode_domain(value.execution_id, codec, persisted=persisted),
        "result_digest": _encode_domain(value.result_digest, codec, persisted=persisted),
        "error_code": _encode_domain(value.error_code, codec, persisted=persisted),
        "error_digest": _encode_domain(value.error_digest, codec, persisted=persisted),
        "next_attempt_at": _encode_domain(
            value.next_attempt_at, codec, persisted=persisted
        ),
        "occupies_concurrency": _encode_domain(
            value.occupies_concurrency, codec, persisted=persisted
        ),
        "error_origin": _encode_domain(
            value.error_origin, codec, persisted=persisted
        ),
        "safe_error_details": _encode_domain(
            dict(value.safe_error_details), codec, persisted=persisted
        ),
    }
    if not persisted:
        fields.update(
            dependencies=_encode_domain(
                value.dependencies,
                codec,
                persisted=persisted,
            ),
            owner=_encode_domain(value.owner, codec, persisted=persisted),
            fence=_encode_domain(value.fence, codec, persisted=persisted),
            lease_expires_at=_encode_domain(
                value.lease_expires_at,
                codec,
                persisted=persisted,
            ),
        )
    return fields


def _decode_v1_task_node_view(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> TaskNodeView:
    if persisted:
        # Lease ownership and resolved dependencies are outside this durable projection.
        raw_fields = {
            **raw_fields,
            "dependencies": {"$tuple": []},
            "owner": None,
            "fence": 1,
            "lease_expires_at": None,
        }
    return cast(TaskNodeView, _decode_dataclass_fields(
        raw_fields, TaskNodeView, codec, persisted=persisted,
    ))


def _decode_v1_task_result(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> TaskResultRecord:
    keys = frozenset(raw_fields)
    base = frozenset({"graph_id", "node_id", "result_digest"})
    if keys != base | {"execution_id"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    execution_id = cast(
        str,
        _decode_domain(
            raw_fields["execution_id"],
            str,
            codec,
            persisted=persisted,
        ),
    )
    return TaskResultRecord(
        cast(
            str,
            _decode_domain(
                raw_fields["graph_id"],
                str,
                codec,
                persisted=persisted,
            ),
        ),
        cast(
            str,
            _decode_domain(
                raw_fields["node_id"],
                str,
                codec,
                persisted=persisted,
            ),
        ),
        cast(
            str,
            _decode_domain(
                raw_fields["result_digest"],
                str,
                codec,
                persisted=persisted,
            ),
        ),
        execution_id,
    )


def _encode_v1_object_ref(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ObjectRef):
        raise TypeError("V1 object_ref encoder received the wrong type")
    encoded: dict[str, JsonValue] = {
        "key": _encode_domain(value.key, codec, persisted=persisted),
        "digest": _encode_domain(value.digest, codec, persisted=persisted),
        "size": _encode_domain(value.size, codec, persisted=persisted),
    }
    if not persisted:
        encoded["store_id"] = _encode_domain(
            value.store_id, codec, persisted=persisted
        )
    return encoded


def _decode_v1_object_ref(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> ObjectRef:
    expected = (
        frozenset({"key", "digest", "size"})
        if persisted
        else frozenset({"store_id", "key", "digest", "size"})
    )
    _require_required_keys(raw_fields, expected)
    store_id = (
        RUNTIME_OBJECT_STORE_ID
        if persisted
        else cast(
            str,
            _decode_domain(
                raw_fields["store_id"], str, codec, persisted=persisted
            ),
        )
    )
    return ObjectRef(
        store_id,
        cast(str, _decode_domain(raw_fields["key"], str, codec, persisted=persisted)),
        cast(
            str,
            _decode_domain(raw_fields["digest"], str, codec, persisted=persisted),
        ),
        cast(int, _decode_domain(raw_fields["size"], int, codec, persisted=persisted)),
    )


def _encode_v1_stored_payload(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, StoredPayload):
        raise TypeError("V1 stored_payload encoder received the wrong type")
    return {
        "kind": _encode_domain(value.kind, codec, persisted=persisted),
        "encoding": _encode_domain(value.encoding, codec, persisted=persisted),
        "digest": _encode_domain(value.digest, codec, persisted=persisted),
        "size": _encode_domain(value.size, codec, persisted=persisted),
        "value": _encode_domain(value.value, codec, persisted=persisted),
        "ref": _encode_domain(value.ref, codec, persisted=persisted),
    }


def _decode_v1_stored_payload(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> StoredPayload:
    _require_required_keys(
        raw_fields, frozenset({"kind", "encoding", "digest", "size", "value", "ref"}),
    )
    return StoredPayload(
        kind=cast(str, _decode_domain(raw_fields["kind"], str, codec, persisted=persisted)),
        encoding=cast(str | None, _decode_domain(
            raw_fields["encoding"], str | None, codec, persisted=persisted,
        )),
        digest=cast(str, _decode_domain(raw_fields["digest"], str, codec, persisted=persisted)),
        size=cast(int, _decode_domain(raw_fields["size"], int, codec, persisted=persisted)),
        value=cast(JsonValue, _decode_domain(raw_fields["value"], JsonValue, codec, persisted=persisted)),
        ref=cast(ObjectRef | None, _decode_domain(
            raw_fields["ref"], ObjectRef | None, codec, persisted=persisted,
        )),
    )


def _encode_v1_stored_user_input(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, StoredUserInput):
        raise TypeError("V1 stored_user_input encoder received the wrong type")
    encoded: dict[str, JsonValue] = {
        "codec": _encode_domain(value.codec, codec, persisted=persisted),
        "payload": _encode_domain(value.payload, codec, persisted=persisted),
    }
    if value.view is not None:
        encoded["view"] = _encode_domain(value.view, codec, persisted=persisted)
    return encoded


def _decode_v1_stored_user_input(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> StoredUserInput:
    _require_required_keys(raw_fields, frozenset({"codec", "payload"}))
    codec_name = _decode_domain(
        raw_fields["codec"], str, codec, persisted=persisted
    )
    if codec_name not in {"text", "user-content-v1", "task-input-v1"}:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    payload = _decode_domain(
        raw_fields["payload"], StoredPayload, codec, persisted=persisted
    )
    view = _decode_domain(
        raw_fields.get("view"),
        Mapping[str, JsonValue] | None,
        codec,
        persisted=persisted,
    )
    try:
        return StoredUserInput(
            codec_name,
            payload,
            cast(Mapping[str, JsonValue] | None, view),
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


_EXECUTION_OUTCOME_FIELDS = {
    "terminal_status": "status",
    "error_code": "error_code",
    "safe_error_details": "safe_error_details",
    "error_diagnostics": "error_diagnostics",
}
_RESULT_OUTCOME_FIELDS = {
    "output": "output", "stop_reason": "stop_reason", "usage": "usage",
    "result_created_at": "created_at",
}


def _execution_outcome_source(raw_fields: Mapping[str, object]) -> dict[str, object]:
    result = raw_fields.get("result")
    if not isinstance(result, Mapping) or result.get("$dataclass") != "result_record":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result_fields = result.get("fields")
    if not isinstance(result_fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return {
            **{name: raw_fields[source] for name, source in _EXECUTION_OUTCOME_FIELDS.items()},
            **{name: result_fields[source] for name, source in _RESULT_OUTCOME_FIELDS.items()},
        }
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _execution_outcome_fields(raw_fields: Mapping[str, object]) -> Mapping[str, object] | None:
    recovery = raw_fields.get("recovery")
    if not isinstance(recovery, Mapping):
        return None
    recovery_fields = recovery.get("fields")
    handoff = recovery_fields.get("terminal_handoff") if isinstance(recovery_fields, Mapping) else None
    handoff_fields = handoff.get("fields") if isinstance(handoff, Mapping) else None
    outcome = handoff_fields.get("outcome") if isinstance(handoff_fields, Mapping) else None
    outcome_fields = outcome.get("fields") if isinstance(outcome, Mapping) else None
    return outcome_fields if isinstance(outcome_fields, Mapping) else None


def _replace_execution_outcome_fields(
    raw_fields: Mapping[str, object], outcome_fields: Mapping[str, object],
) -> dict[str, object]:
    # Copy only the typed owner path; caller-owned wire values remain unchanged.
    result = dict(raw_fields)
    recovery = dict(result["recovery"])
    result["recovery"] = recovery
    recovery_fields = dict(recovery["fields"])
    recovery["fields"] = recovery_fields
    handoff = dict(recovery_fields["terminal_handoff"])
    recovery_fields["terminal_handoff"] = handoff
    handoff_fields = dict(handoff["fields"])
    handoff["fields"] = handoff_fields
    handoff_fields["outcome"] = {**handoff_fields["outcome"], "fields": dict(outcome_fields)}
    return result


def _encode_v1_execution_record(
    value: object, codec: "_VersionCodec", persisted: bool,
    *, binding: JsonValue | None = None,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ExecutionRecord):
        raise TypeError("V1 execution_record encoder received the wrong type")
    encoded = {
        name: binding if name == "binding" and binding is not None else
        _encode_domain(getattr(value, name), codec, persisted=persisted)
        for name in _dataclass_fields(ExecutionRecord)
        if not (persisted and (name == "repository_instructions" and value.input_context is not None
                              or name in {"memory_scope", "tool_response_ref"} and getattr(value, name) is None))
    }
    recovery = value.recovery
    handoff = None if recovery is None else recovery.terminal_handoff
    # Until the terminal result exists, recovery owns its complete outcome.
    # Checkpoint-owned objects must keep their separate retention source.
    if (persisted and value.result is not None and handoff is not None
            and handoff.outcome.object_source_domain in {None, PersistenceDomain.EXECUTION}):
        outcome = _execution_outcome_fields(encoded)
        # Whole templates already share fixed output/diagnostic owners. This
        # older inline projection is valid only when both inline sources exist;
        # a Result reference intentionally has no stop_reason/usage fields.
        result_fields = encoded["result"]["fields"]
        if outcome is None or "content" in result_fields:
            return cast(Mapping[str, JsonValue], encoded)
        source = _execution_outcome_source(encoded)
        if all(
            canonical_json_bytes(outcome.get(name)) == canonical_json_bytes(item)
            for name, item in source.items()
        ):
            compact = {name: item for name, item in outcome.items() if name not in source}
            compact["execution_result"] = True
            encoded = _replace_execution_outcome_fields(encoded, compact)
    return cast(Mapping[str, JsonValue], encoded)


def decode_execution_terminal_details(payload: StoredPayload | None) -> Mapping[str, JsonValue]:
    """Decode the typed empty-diagnostics marker or its verified content owner."""
    if payload is None:
        return {"safe_error_details": _encode_persisted_domain({}), "error_diagnostics": None}
    if not isinstance(payload, StoredPayload) or payload.kind != "inline" or payload.encoding != "json":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    details = payload.decode()
    if not isinstance(details, Mapping) or set(details) != {"safe_error_details", "error_diagnostics"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return details


def _execution_terminal_fields(raw_fields: Mapping[str, object], codec: "_VersionCodec", *, persisted: bool) -> Mapping[str, object]:
    if "terminal_details" not in raw_fields:
        return raw_fields
    if "safe_error_details" in raw_fields or "error_diagnostics" in raw_fields:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    payload = _decode_domain(raw_fields["terminal_details"], StoredPayload | None, codec, persisted=persisted)
    details = decode_execution_terminal_details(payload)
    return {**{name: item for name, item in raw_fields.items() if name != "terminal_details"}, **details}


def _decode_v1_execution_record(
    raw_fields: Mapping[str, object], codec: "_VersionCodec", persisted: bool,
) -> ExecutionRecord:
    if "runtime_policy" in raw_fields or "runtime_overrides" in raw_fields:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_fields = _execution_terminal_fields(raw_fields, codec, persisted=persisted)
    outcome = _execution_outcome_fields(raw_fields)
    if outcome is not None and "execution_result" in outcome:
        if not persisted or outcome["execution_result"] is not True:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = _execution_outcome_source(raw_fields)
        source_domain = _decode_domain(
            outcome.get("object_source_domain"), PersistenceDomain | None, codec, persisted=persisted,
        )
        if (set(outcome).intersection(source)
                or source_domain not in {None, PersistenceDomain.EXECUTION}):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raw_fields = _replace_execution_outcome_fields(raw_fields, {
            **{name: item for name, item in outcome.items() if name != "execution_result"}, **source,
        })
    return cast(ExecutionRecord, _decode_dataclass_fields(
        raw_fields, ExecutionRecord, codec, persisted=persisted,
    ))


def _encode_v1_execution_start_reservation(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ExecutionStartReservation):
        raise TypeError("V1 execution_start_reservation encoder received the wrong type")
    # Budget admission has its own durable owner outside this projection.
    return {
        "execution": _encode_domain(value.execution, codec, persisted=persisted),
        "idempotency": _encode_domain(value.idempotency, codec, persisted=persisted),
    }


def _decode_v1_execution_start_reservation(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> ExecutionStartReservation:
    _require_required_keys(raw_fields, frozenset({"execution", "idempotency"}))
    return ExecutionStartReservation(
        cast(ExecutionRecord, _decode_domain(
            raw_fields["execution"], ExecutionRecord, codec, persisted=persisted,
        )),
        cast(IdempotencyRecord, _decode_domain(
            raw_fields["idempotency"], IdempotencyRecord, codec, persisted=persisted,
        )),
    )


def _encode_v1_slot_disposition_view(
    value: object,
    codec: "_VersionCodec",
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, SlotDispositionView):
        raise TypeError("V1 evaluation_slot_disposition_view encoder received the wrong type")
    return {
        "kind": value.kind,
        "reason_code": value.reason_code,
        "terminal": value.terminal,
        "retryable": value.retryable,
        "created_at": _encode_domain(value.created_at, codec, persisted=persisted),
    }


def _decode_v1_slot_disposition_view(
    raw_fields: Mapping[str, object],
    codec: "_VersionCodec",
    persisted: bool,
) -> SlotDispositionView:
    _require_required_keys(
        raw_fields, frozenset({"kind", "reason_code", "terminal", "retryable", "created_at"}),
    )
    return SlotDispositionView(
        kind=_decode_domain(raw_fields["kind"], str, codec, persisted=persisted),
        reason_code=_decode_domain(raw_fields["reason_code"], str, codec, persisted=persisted),
        terminal=_decode_domain(raw_fields["terminal"], bool, codec, persisted=persisted),
        retryable=_decode_domain(raw_fields["retryable"], bool, codec, persisted=persisted),
        created_at=_decode_domain(raw_fields["created_at"], datetime, codec, persisted=persisted),
    )


def _encode_v1_error_diagnostics(
    value: object, codec: "_VersionCodec", persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ErrorDiagnostics):
        raise TypeError("V1 error_diagnostics encoder received the wrong type")
    encoded = value.to_payload()
    if value.exception_chain is not None:
        encoded["exception_chain"] = _encode_domain(value.exception_chain, codec, persisted=persisted)
    return encoded


def _encode_v1_model_client_request_ref(
    value: object, codec: "_VersionCodec", persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ModelClientRequestRef):
        raise TypeError("V1 model_client_request_ref encoder received the wrong type")
    return {
        field.name: _encode_domain(getattr(value, field.name), codec, persisted=persisted)
        for field in fields(value)
        if field.name != "diagnostics" or value.diagnostics is not None
    }


def _encode_v1_model_interaction(
    value: object, codec: "_VersionCodec", persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, ModelInteractionRecord):
        raise TypeError("V1 model_interaction encoder received the wrong type")
    return {
        field.name: _encode_domain(getattr(value, field.name), codec, persisted=persisted)
        for field in fields(value)
        if field.name != "stream_attempts" or value.stream_attempts is not None
    }


_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(
    {
        "error_diagnostics": _encode_v1_error_diagnostics,
        "model_interaction": _encode_v1_model_interaction,
        "model_client_request_ref": _encode_v1_model_client_request_ref,
        "object_ref": _encode_v1_object_ref,
        "stored_payload": _encode_v1_stored_payload,
        "stored_user_input": _encode_v1_stored_user_input,
        "task_graph_view": _encode_v1_task_graph_view,
        _TASK_NODE_WIRE_ID: _encode_v1_task_node,
        _TASK_NODE_VIEW_WIRE_ID: _encode_v1_task_node_view,
        "execution_record": _encode_v1_execution_record,
        "execution_start_reservation": _encode_v1_execution_start_reservation,
        "evaluation_slot_disposition_view": _encode_v1_slot_disposition_view,
    }
)
_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(
    {
        "object_ref": _decode_v1_object_ref,
        "stored_payload": _decode_v1_stored_payload,
        "stored_user_input": _decode_v1_stored_user_input,
        "task_graph_view": _decode_v1_task_graph_view,
        _TASK_NODE_WIRE_ID: _decode_v1_task_node,
        _TASK_NODE_VIEW_WIRE_ID: _decode_v1_task_node_view,
        "task_result": _decode_v1_task_result,
        "execution_record": _decode_v1_execution_record,
        "execution_start_reservation": _decode_v1_execution_start_reservation,
        "evaluation_slot_disposition_view": _decode_v1_slot_disposition_view,
    }
)

_V1_EXTERNAL_SCHEMA_TYPES: frozenset[type[object]] = frozenset(
    {
        IdempotencyTerminalUpdate,
        OperationTerminalUpdate,
        AgentBindingContract,
        AssetVersionRef,
        ScoreBundle,
        ModelRequest,
        ModelResponse,
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
_VERSION_CODECS: Mapping[int, _VersionCodec] = MappingProxyType({1: _V1_CODEC})
_CURRENT_CODEC = _VERSION_CODECS[CURRENT_DATA_VERSION]
_JSON_VALUE_FORWARD_REF = ForwardRef("JsonValue")


@dataclass(frozen=True, slots=True)
class CanonicalEnvelope:
    version: int
    value: Mapping[str, JsonValue]


def _require_required_keys(
    value: Mapping[str, object],
    required: frozenset[str],
) -> None:
    if not required.issubset(value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _unwrap_tagged_list(value: object, tag: str) -> list[object]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_required_keys(value, frozenset({tag}))
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
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            "canonical data must be an object",
        )
    _require_required_keys(value, frozenset({"v", "value"}))
    version = value.get("v")
    payload = value.get("value")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            "canonical data version is invalid",
        )
    if not isinstance(payload, Mapping):
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            "canonical data value is invalid",
        )
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
        "key": record.key.hex(),
        "scope": None if record.scope_key is None else record.scope_key.hex(),
        "parent": None if record.parent_key is None else record.parent_key.hex(),
        "subject": record.subject_id,
        "kind": record.kind,
        "sort": base64.b64encode(record.sort_key).decode("ascii"),
        "state": record.state,
        "storage_revision": record.storage_revision,
        "lease": {
            "owner": record.lease_owner,
            "fence": record.lease_fence,
            "expires_at": None
            if record.lease_expires_at is None
            else record.lease_expires_at.isoformat(),
        },
        "data": dict(record.data),
        "payload_bytes": None if record.payload_bytes is None else base64.b64encode(record.payload_bytes).decode("ascii"),
    }


def _decode_record_sort(value: str) -> bytes:
    raw = base64.b64decode(value, validate=True)
    if base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("record sort encoding is not canonical")
    return validate_sort_key(raw)


def decode_record(value: Mapping[str, JsonValue]) -> StoredRecord:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    current_keys = frozenset(
        {
            "key",
            "scope",
            "parent",
            "subject",
            "kind",
            "sort",
            "state",
            "storage_revision",
            "lease",
            "data",
        }
    )
    _require_required_keys(value, current_keys)
    lease = value["lease"]
    if not isinstance(lease, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_required_keys(lease, frozenset({"owner", "fence", "expires_at"}))
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        record = StoredRecord(
            _record_key_wire(_string(value, "key")),
            _optional_relation_key(value["scope"]),
            _optional_parent_key(value["parent"]),
            _string(value, "kind"),
            _decode_record_sort(_string(value, "sort")),
            _optional_string(value["state"]),
            _integer(value, "storage_revision"),
            _optional_string(lease["owner"]),
            _integer(lease, "fence"),
            _optional_datetime(lease["expires_at"]),
            dict(data),
            subject_id=_optional_string(value["subject"]),
            payload_bytes=None if value.get("payload_bytes") is None else base64.b64decode(value["payload_bytes"], validate=True),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return record


def encode_fact(fact: StoredFact) -> dict[str, JsonValue]:
    return {
        "stream": fact.stream_key.hex(),
        "sequence": fact.sequence,
        "owner": fact.owner_key.hex(),
        "kind": fact.kind,
        "subject": fact.subject_id,
        "state": fact.state,
        "data": dict(fact.data),
    }


def decode_fact(value: Mapping[str, JsonValue]) -> StoredFact:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_required_keys(
        value,
        frozenset(
            {"stream", "sequence", "owner", "kind", "subject", "state", "data"}
        ),
    )
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredFact(
            _stream_key_wire(_string(value, "stream")),
            _integer(value, "sequence"),
            _record_key_wire(_string(value, "owner")),
            _string(value, "kind"),
            _optional_string(value["subject"]),
            _optional_string(value["state"]),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def encode_operation(operation: StoredOperation) -> dict[str, JsonValue]:
    return {
        "key": operation.key.hex(),
        "stream": operation.stream_key.hex(),
        "sequence": operation.sequence,
        "state": operation.state,
        "compactable": operation.compactable,
        "data": dict(operation.data),
    }


def decode_operation(value: Mapping[str, JsonValue]) -> StoredOperation:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_required_keys(
        value,
        frozenset({"key", "stream", "sequence", "state", "compactable", "data"}),
    )
    data = value["data"]
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return StoredOperation(
            _operation_key_wire(_string(value, "key")),
            _stream_key_wire(_string(value, "stream")),
            _integer(value, "sequence"),
            _string(value, "state"),
            _bool(value, "compactable"),
            dict(data),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def wire_type_id(value: DomainT | type[DomainT]) -> str:
    """Return the explicit stable protocol id for one persisted type."""
    return _codec_wire_type_id(value, _CURRENT_CODEC)


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
    if isinstance(value, (AgentBindingContract, AssetVersionRef)):
        return value.to_payload()
    if isinstance(value, ScoreBundle):
        return _encode_domain(value.to_mapping(), codec)
    if isinstance(value, (IdempotencyTerminalUpdate, OperationTerminalUpdate)):
        return {
            name: _encode_domain(attrgetter(name)(value), codec)
            for name in _dataclass_fields(type(value))
        }
    if isinstance(value, (ModelRequest, ModelResponse)):
        raw = encode_model_messages((value,))
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise RuntimeError(
                "model message codec returned an invalid single-message value"
            )
        return cast(JsonValue, decoded[0])
    raise TypeError(f"unsupported external type: {type(value).__name__}")


def _decode_external(
    value: object,
    target: type[object],
    codec: _VersionCodec,
    *,
    persisted: bool,
) -> object:
    if target not in codec.external_schema_types:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if target in (AgentBindingContract, AssetVersionRef, ScoreBundle):
        try:
            if target is ScoreBundle:
                payload = cast(
                    dict[str, JsonValue],
                    _decode_domain(value, dict[str, JsonValue], codec, persisted=persisted),
                )
                return ScoreBundle.from_mapping(payload)
            return target.from_payload(value)
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
    if target in (IdempotencyTerminalUpdate, OperationTerminalUpdate):
        declared = _dataclass_fields(target)
        _require_required_keys(value, frozenset(declared))
        return target(**{
            name: _decode_domain(value[name], field_type, codec)
            for name, (field_type, _init) in declared.items()
        })
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
        raw_value = _encode_enum_value(value.value)
        return {"$enum": wire_id, "value": raw_value}
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Runtime v1 wire requires finite floats")
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
            declared = _dataclass_fields(type(value))
            try:
                encoded_fields = {
                    field_name: _encode_domain(
                        attrgetter(field_name)(value),
                        codec,
                        persisted=persisted,
                    )
                    for field_name in declared
                }
            except AttributeError as error:
                raise TypeError(
                    f"Runtime v{codec.version} dataclass implementation no longer matches {wire_id}"
                ) from error
        wire: dict[str, JsonValue] = {
            "$dataclass": wire_id,
            "fields": dict(encoded_fields),
        }
        if persisted:
            wire = {
                "$dataclass": wire_id,
                "schema": codec.version,
                "fields": dict(encoded_fields),
            }
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
                _encode_domain(item, codec, persisted=persisted)
                for item in value
            ]
        }
    if isinstance(value, list):
        return [
            _encode_domain(item, codec, persisted=persisted)
            for item in value
        ]
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
    try:
        return _encode_domain(value, _CURRENT_CODEC, persisted=True)
    except Exception as error:
        _logger.error(
            "Runtime persistence encode failed: type=%s",
            type(value).__name__,
        )
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            retryable=False,
            safe_details={
                "phase": "persistence_encode",
                "domain_type": type(value).__name__,
            },
        ) from error


def decode_domain(value: JsonValue, target: type[DomainT]) -> DomainT:
    """Decode a canonical value using the declared domain type."""
    return _decode_domain(value, target, _CURRENT_CODEC)  # type: ignore[return-value]


def _execution_projection_fields(value: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], _VersionCodec]:
    envelope = decode_envelope(value)
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    payload = envelope.value.get("payload")
    if (envelope.value.get("type") != "execution_record" or not isinstance(payload, Mapping)
            or payload.get("$dataclass") != "execution_record"):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _require_required_keys(payload, frozenset({"$dataclass", "schema", "fields"}))
    if type(payload["schema"]) is not int or payload["schema"] != envelope.version:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    source = payload["fields"]
    if not isinstance(source, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return source, codec


def _execution_projection_field(source: Mapping[str, JsonValue], name: str, codec: _VersionCodec) -> object:
    if name in EXECUTION_POLICY_FIELDS and "runtime_policy" in source:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if name in {"safe_error_details", "error_diagnostics"}:
        source = _execution_terminal_fields(source, codec, persisted=True)
    target, _init = _dataclass_fields(ExecutionRecord)[name]
    if name in source:
        try:
            return _decode_domain(source[name], target, codec, persisted=True)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    declaration = next(field for field in fields(ExecutionRecord) if field.name == name)
    if declaration.default is not MISSING:
        return declaration.default
    if declaration.default_factory is not MISSING:
        return declaration.default_factory()
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def decode_execution_retention(value: Mapping[str, JsonValue]) -> ExecutionRetentionRecord:
    """Read only closure authority; immutable content remains opaque."""
    source, codec = _execution_projection_fields(value)
    try:
        recovery = source.get("recovery")
        handoff = False
        if recovery is not None:
            if (not isinstance(recovery, Mapping) or recovery.get("$dataclass") != "execution_recovery_detail"
                    or recovery.get("schema") != 1 or not isinstance(recovery.get("fields"), Mapping)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw = recovery["fields"]
            declaration = next(field for field in fields(ExecutionRecoveryDetail) if field.name == "handoff_phase")
            phase = (_decode_domain(raw["handoff_phase"], RecoveryHandoffPhase, codec, persisted=True)
                     if "handoff_phase" in raw else declaration.default)
            terminal = raw.get("terminal_handoff")
            pending = raw.get("pending_tools")
            if ((phase is RecoveryHandoffPhase.NONE and terminal is not None)
                    or (phase is RecoveryHandoffPhase.PREPARED and terminal is None)
                    or (phase is not RecoveryHandoffPhase.NONE and pending is not None)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            handoff = phase is RecoveryHandoffPhase.PREPARED
        result = ExecutionRetentionRecord(**{
            field.name: _execution_projection_field(source, field.name, codec)
            for field in fields(ExecutionRetentionRecord) if field.name != "recovery_handoff"
        }, recovery_handoff=handoff)
        if result.revision < 0 or not result.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def decode_execution_result(value: Mapping[str, JsonValue], *, codec: _VersionCodec | None = None) -> ExecutionResultRecord:
    """Decode result fields using the canonical owner's omission semantics."""
    source, stored_codec = _execution_projection_fields(value)
    try:
        return ExecutionResultRecord(**{
            field.name: _execution_projection_field(source, field.name, codec or stored_codec)
            for field in fields(ExecutionResultRecord)
        })
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def decode_execution_header(value: Mapping[str, JsonValue], *, tenant_id: str) -> ResourceRef:
    """Decode the execution's authorization identity without loading content."""
    source, codec = _execution_projection_fields(value)
    identity = _execution_projection_field(source, "execution_id", codec)
    if not isinstance(identity, str) or not identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return ResourceRef(ResourceKind.EXECUTION, identity, tenant_id)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def decode_execution_producer_authority(
    value: Mapping[str, JsonValue],
) -> tuple[int, ExecutionStatus, str | None]:
    """Validate dispatch authority without borrowing Runtime policy content."""
    source, codec = _execution_projection_fields(value)
    payload = decode_envelope(value).value["payload"]
    policy = runtime_policy_reference(payload)
    if policy is not None:
        overrides = source.get("runtime_overrides", {})
        if (set(source).intersection(EXECUTION_POLICY_FIELDS) or not isinstance(overrides, Mapping)
                or set(overrides) - set(EXECUTION_POLICY_FIELDS)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for name in overrides:
            _execution_projection_field(overrides, name, codec)
    generation = _execution_projection_field(source, "producer_generation", codec)
    claim = _execution_projection_field(source, "producer_claim_id", codec)
    status = _execution_projection_field(source, "status", codec)
    seal = _execution_projection_field(source, "history_seal_digest", codec)
    if (generation < 0 or claim is not None and (not claim or generation < 1)
            or seal is not None and (len(seal) != 64 or any(char not in "0123456789abcdef" for char in seal))):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return generation, status, seal


def decode_execution_producer_control(
    value: Mapping[str, JsonValue],
) -> tuple[int, ExecutionStatus, bool, str | None]:
    """Validate history retention after its fixed Runtime policy is resolved."""
    source, codec = _execution_projection_fields(value)
    generation, status, seal = decode_execution_producer_authority(value)
    durable_trace = _execution_projection_field(source, "durable_trace", codec)
    if seal is not None and not durable_trace:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return generation, status, durable_trace, seal


def decode_execution_control(value: Mapping[str, JsonValue]) -> ExecutionControlRecord:
    """Project the fields consumed by producer and history publication guards."""
    source, codec = _execution_projection_fields(value)
    generation, status, seal = decode_execution_producer_authority(value)
    names = ("execution_id", "producer_claim_id", "budget_scope_id", "retention_closed")
    values = {name: _execution_projection_field(source, name, codec) for name in names}
    try:
        validate_resource_id(values["execution_id"])
        budget_scope = values["budget_scope_id"]
        if budget_scope is not None and (not isinstance(budget_scope, str) or not budget_scope.strip()):
            raise ValueError("execution budget scope is invalid")
        if not isinstance(values["retention_closed"], bool):
            raise TypeError("execution retention_closed must be bool")
        return ExecutionControlRecord(**values, status=status, producer_generation=generation,
                                      history_seal_digest=seal)
    except (AIError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _validate_execution_view(view: ExecutionView, summary: Mapping[str, JsonValue]) -> None:
    # Preserve the owner's invariants for fields consumed by this projection.
    if summary["kind"] == "agent":
        if view.task_attempt != 0 or view.task_deadline_at is not None or view.task_next_attempt_at is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    else:
        if (view.session_id is not None or view.lineage_kind is not ExecutionLineageKind.RUN
                or view.parent_execution_id is not None or view.parent_invocation_id is not None
                or view.agent_run_seq != 0
                or view.task_attempt < 0 or view.task_attempt > summary["max_attempts"]):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if view.lineage_kind is ExecutionLineageKind.SUBAGENT:
        if not view.parent_execution_id or not view.parent_invocation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif view.parent_execution_id is not None or view.parent_invocation_id is not None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def decode_execution_view(value: Mapping[str, JsonValue], *, tenant_id: str) -> tuple[ResourceRef, ExecutionView]:
    """Decode state from the verified admitted binding, without input or output."""
    source, codec = _execution_projection_fields(value)
    names = ("execution_id", "status", "lineage_kind", "parent_execution_id", "root_execution_id",
             "parent_invocation_id", "session_id", "task_attempt", "task_deadline_at",
             "task_next_attempt_at", "event_seq", "started_at", "agent_run_seq", "durable_trace", "created_at")
    values = {name: _execution_projection_field(source, name, codec) for name in names}
    summary = _execution_binding_summary(_execution_projection_field(source, "binding", codec))
    principal_id = _execution_projection_field(source, "principal_id", codec)
    if not isinstance(principal_id, str) or not principal_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result = source.get("result")
    terminal_at = None
    if result is not None:
        if (not isinstance(result, Mapping) or result.get("$dataclass") != "result_record"
                or not isinstance(result.get("fields"), Mapping)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if type(result.get("schema")) is not int or result["schema"] != codec.version:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        result_fields = result["fields"]
        if "created_at" not in result_fields:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        terminal_at = _decode_domain(result_fields["created_at"], datetime, codec, persisted=True)
        if terminal_at.tzinfo is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        view = ExecutionView(**values,
            agent_id=summary["id"] if summary["kind"] == "agent" else None,
            task_id=summary["id"] if summary["kind"] == "task" else None,
            binding_kind=summary["kind"],
            terminal_at=terminal_at)
        _validate_execution_view(view, summary)
        return ResourceRef(ResourceKind.EXECUTION, view.execution_id, tenant_id, principal_id), view
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def decode_execution_metadata(value: Mapping[str, JsonValue], *, tenant_id: str) -> ExecutionMetadataRecord:
    """Project diagnostic metadata from the verified admitted binding."""
    header, view = decode_execution_view(value, tenant_id=tenant_id)
    source, codec = _execution_projection_fields(value)
    summary = _execution_binding_summary(_execution_projection_field(source, "binding", codec))
    names = ("updated_at", "error_code", "safe_error_details")
    values = {name: _execution_projection_field(source, name, codec) for name in names}
    refs = (None if summary["kind"] == "task" else
            _decode_domain(summary["capability_refs"], tuple[CapabilityRef, ...], codec, persisted=True))
    if values["updated_at"].tzinfo is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ExecutionMetadataRecord(view=view, principal_id=header.owner_principal_id,
                                   capability_refs=refs, binding_digest=summary["binding_digest"], **values)


_PAYLOAD_PATHS = (
    ("stored_user_input",),
    ("terminal_details",),
    ("result",),
    ("result", "fields", "output"),
    ("recovery", "fields", "pending_tools"),
    ("recovery", "fields", "terminal_handoff"),
    ("recovery", "fields", "terminal_handoff", "fields", "output"),
    ("recovery", "fields", "terminal_handoff", "fields", "terminal_details"),
)


def execution_payload_slots(value: object) -> tuple[tuple[tuple[str, ...], Mapping], ...]:
    """Select declared payload owners, without entering application JSON."""
    if not isinstance(value, Mapping) or value.get("$dataclass") != "execution_record":
        return ()
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result = []
    for path in _PAYLOAD_PATHS:
        item = fields
        for name in path:
            item = item.get(name) if isinstance(item, Mapping) else None
        if item is not None:
            expected = {"stored_user_input": "stored_user_input", "pending_tools": "pending_tool_continuation",
                        "terminal_handoff": "recovery_terminal_handoff", "result": "result_record"}.get(path[-1], "stored_payload")
            if not isinstance(item, Mapping) or item.get("$dataclass") != expected:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if native_text_reference(item) is not None and path not in {("result", "fields", "output"), ("recovery", "fields", "terminal_handoff", "fields", "output")}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if native_user_input_reference(item) is not None and path != ("stored_user_input",):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result.append((path, item))
            if path == ("recovery", "fields", "pending_tools") and "content" in item["fields"]:
                arguments = item["fields"].get("arguments")
                if not isinstance(arguments, Mapping) or set(arguments) != {str(index) for index in range(len(arguments))}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for index in range(len(arguments)):
                    argument = arguments[str(index)]
                    if not isinstance(argument, Mapping) or argument.get("$dataclass") != "stored_payload":
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if native_text_reference(argument) is not None or native_user_input_reference(argument) is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    result.append(((*path, "fields", "arguments", str(index)), argument))
    return tuple(result)


def native_user_input_reference(value: Mapping) -> NativeUserInputReference | None:
    """Recognize a projection only within the declared accepted-input slot."""
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = fields.get("content")
    if not isinstance(content, Mapping) or "$native_user_input" not in content:
        return None
    _validate_dataclass_wrapper(value, StoredUserInput, _CURRENT_CODEC, persisted=True)
    if set(fields) != {"content"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return NativeUserInputReference.from_json(content)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def validate_native_user_input(reference: NativeUserInputReference, value: object) -> StoredUserInput:
    from ..model import plain_text_input_content

    if (not isinstance(reference, NativeUserInputReference) or not isinstance(value, StoredUserInput)
            or plain_text_input_content(value) is None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def native_text_reference(value: Mapping) -> NativeTextReference | None:
    """Recognize a projection only inside its declared result payload slot."""
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = fields.get("content")
    if not isinstance(content, Mapping) or "$native_text" not in content:
        return None
    _validate_dataclass_wrapper(value, StoredPayload, _CURRENT_CODEC, persisted=True)
    if set(fields) != {"content"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return NativeTextReference.from_json(content)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def validate_native_text_payload(reference: NativeTextReference, payload: object) -> StoredPayload:
    if (not isinstance(payload, StoredPayload) or payload.kind != "inline" or payload.encoding != "json"
            or not isinstance(reference, NativeTextReference)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = payload.decode()
    if ((reference.projection == "text" and not isinstance(value, str))
            or (reference.projection == "assistant-text-output" and
                (not isinstance(value, dict) or set(value) != {"text"} or not isinstance(value["text"], str)))
            or (reference.projection == "tool-call-output" and not isinstance(value, dict))):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return payload


def payload_reference(value: Mapping) -> tuple[str, str, str] | None:
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if "content" not in fields:
        return None
    target = {"stored_user_input": StoredUserInput, "stored_payload": StoredPayload,
              "pending_tool_continuation": PendingToolContinuation,
              "recovery_terminal_handoff": RecoveryTerminalHandoff, "result_record": ResultRecord}.get(value.get("$dataclass"))
    if target is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _validate_dataclass_wrapper(value, target, _CURRENT_CODEC, persisted=True)
    if native_user_input_reference(value) is not None or native_text_reference(value) is not None:
        return None
    reference = execution_payload_reference(fields["content"])
    expected = ({"content", "output", "terminal_details"} if target is RecoveryTerminalHandoff else
                {"content", "output"} if target is ResultRecord else
                {"content", "arguments"} if target is PendingToolContinuation else {"content"})
    if reference is None or set(fields) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return reference


def map_execution_payload_slots(value: Mapping, mapper: Callable[[Mapping], JsonValue]) -> dict:
    fields = dict(value["fields"])
    # Child reference slots must be rewritten before replacing their whole
    # typed parent; a restored handoff no longer has the wrapper-only output slot.
    for path, payload in reversed(execution_payload_slots(value)):
        current = fields
        for name in path[:-1]:
            current[name] = dict(current[name])
            current = current[name]
        current[path[-1]] = mapper(current[path[-1]])
    return {**value, "fields": fields}


EXECUTION_POLICY_FIELDS = (
    "mode", "planning", "thinking", "tool_effects", "session_context",
    "restart_checkpoints", "durable_trace", "conversation",
)


def runtime_policy_reference(value: object) -> tuple[str, str, str] | None:
    """Read the fixed policy owner only from its declared execution slot."""
    if not isinstance(value, Mapping) or value.get("$dataclass") != "execution_record":
        return None
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reference = fields.get("runtime_policy")
    if reference is None:
        if "runtime_overrides" in fields:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None
    if not isinstance(reference, Mapping) or "$runtime_policy" not in reference:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _execution_content_reference(reference, "$runtime_policy")


def encode_execution_content(
    value: ExecutionRecord, payloads: Mapping[int, JsonValue],
    *, binding: JsonValue | None = None, policy: JsonValue | None = None,
    policy_overrides: Mapping[str, JsonValue] | None = None,
    terminal_details: JsonValue | None = None,
) -> JsonValue:
    """Encode references only in the declared input and result owner slots."""
    def input_encoder(item: object, codec: _VersionCodec, persisted: bool) -> Mapping[str, JsonValue]:
        if id(item) in payloads:
            return payloads[id(item)]["fields"]
        return _encode_v1_stored_user_input(item, codec, persisted)
    def output_encoder(item: object, codec: _VersionCodec, persisted: bool) -> Mapping[str, JsonValue]:
        return {name: (payloads[id(getattr(item, name))]
            if name == "output" and id(getattr(item, name)) in payloads else
            _encode_domain(getattr(item, name), codec, persisted=persisted))
            for name in _dataclass_fields(type(item))}
    def recovery_content_encoder(item: object, codec: _VersionCodec, persisted: bool) -> Mapping[str, JsonValue]:
        if id(item) in payloads:
            return payloads[id(item)]["fields"]
        return {name: _encode_domain(getattr(item, name), codec, persisted=persisted)
                for name in _dataclass_fields(type(item))}
    def execution_encoder(item: object, codec: _VersionCodec, persisted: bool) -> Mapping[str, JsonValue]:
        encoded = dict(_encode_v1_execution_record(item, codec, persisted, binding=binding))
        if policy is not None:
            for name in EXECUTION_POLICY_FIELDS:
                encoded.pop(name)
            encoded["runtime_policy"] = policy
            if policy_overrides:
                encoded["runtime_overrides"] = dict(policy_overrides)
        if terminal_details is not None or not item.safe_error_details and item.error_diagnostics is None:
            encoded.pop("safe_error_details", None)
            encoded.pop("error_diagnostics", None)
            encoded["terminal_details"] = terminal_details
        return encoded
    codec = replace(_CURRENT_CODEC, dataclass_encoders={**_CURRENT_CODEC.dataclass_encoders,
        "execution_record": execution_encoder,
        "stored_user_input": input_encoder, "result_record": recovery_content_encoder,
        "recovery_terminal_outcome": output_encoder,
        "pending_tool_continuation": recovery_content_encoder,
        "recovery_terminal_handoff": recovery_content_encoder,
    })
    return _encode_domain(value, codec, persisted=True)


def execution_payload_reference(value: object) -> tuple[str, str, str] | None:
    """Recognize a storage reference only in a typed StoredPayload slot."""
    if not isinstance(value, Mapping) or "$execution_payload" not in value:
        return None
    return _execution_content_reference(value, "$execution_payload")


def _restore_pending_arguments(template: Mapping, arguments: Mapping) -> dict:
    if not isinstance(arguments, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fields_value = dict(template["fields"])
    index = 0
    for name in ("approvals", "calls"):
        sequence = fields_value.get(name)
        if not isinstance(sequence, Mapping) or set(sequence) != {"$tuple"} or not isinstance(sequence["$tuple"], list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        restored = []
        for call in sequence["$tuple"]:
            if not isinstance(call, Mapping) or call.get("$dataclass") != "pending_deferred_call" or str(index) not in arguments:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            restored.append({**call, "fields": {**call["fields"], "arguments_payload": arguments[str(index)]}})
            index += 1
        fields_value[name] = {"$tuple": restored}
    if set(arguments) != {str(position) for position in range(index)}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return {**template, "fields": fields_value}


def _execution_payload_codec(codec: _VersionCodec, payloads: Mapping) -> _VersionCodec:
    def decoder(original: DataclassDecoder) -> DataclassDecoder:
        def decode_payload(raw: Mapping[str, object], current: _VersionCodec, persisted: bool) -> object:
            content = raw.get("content")
            if isinstance(content, Mapping) and "$native_user_input" in content:
                if original is not _decode_v1_stored_user_input:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    reference = NativeUserInputReference.from_json(content)
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            elif isinstance(content, Mapping) and "$native_text" in content:
                if original is not _decode_v1_stored_payload:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                try:
                    reference = NativeTextReference.from_json(content)
                except (TypeError, ValueError, KeyError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            else:
                reference = execution_payload_reference(content)
            if reference is None:
                return original(raw, current, persisted)
            if set(raw) not in ({"content"}, {"content", "output"}, {"content", "arguments"}, {"content", "output", "terminal_details"}) or reference not in payloads:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            selected = payloads[reference]
            if isinstance(selected, Mapping) and selected.get("$dataclass") == "pending_tool_continuation":
                if set(raw) != {"content", "arguments"}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                restored = _restore_pending_arguments(selected, raw["arguments"])
                return _decode_domain(restored, PendingToolContinuation, current, persisted=True)
            if isinstance(selected, Mapping) and selected.get("$dataclass") in {"result_record", "recovery_terminal_handoff"}:
                expected = {"content", "output", "terminal_details"} if selected["$dataclass"] == "recovery_terminal_handoff" else {"content", "output"}
                if set(raw) != expected:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                output = _decode_domain(raw["output"], StoredPayload | None, current, persisted=True)
                restored = dict(selected)
                fields_value = dict(restored["fields"])
                if selected["$dataclass"] == "result_record":
                    fields_value["output"] = _encode_domain(output, current, persisted=True)
                    target = ResultRecord
                else:
                    outcome = dict(fields_value["outcome"])
                    details_payload = _decode_domain(raw["terminal_details"], StoredPayload | None, current, persisted=True)
                    details = decode_execution_terminal_details(details_payload)
                    outcome["fields"] = {**outcome["fields"], **details, "output": _encode_domain(output, current, persisted=True)}
                    fields_value["outcome"] = outcome
                    target = RecoveryTerminalHandoff
                restored["fields"] = fields_value
                return _decode_domain(restored, target, current, persisted=True)
            if isinstance(reference, NativeUserInputReference):
                return validate_native_user_input(reference, selected)
            return validate_native_text_payload(reference, selected) if isinstance(reference, NativeTextReference) else selected
        return decode_payload
    current = replace(codec, dataclass_decoders={**codec.dataclass_decoders,
        "result_record": decoder(lambda raw, current, persisted:
            _decode_dataclass_fields(raw, ResultRecord, current, persisted=persisted)),
        "stored_payload": decoder(_decode_v1_stored_payload),
        "stored_user_input": decoder(_decode_v1_stored_user_input),
        "pending_tool_continuation": decoder(lambda raw, current, persisted:
            _decode_dataclass_fields(raw, PendingToolContinuation, current, persisted=persisted)),
        "recovery_terminal_handoff": decoder(lambda raw, current, persisted:
            _decode_dataclass_fields(raw, RecoveryTerminalHandoff, current, persisted=persisted)),
    })
    return current


def decode_execution_payload(value: Mapping, payloads: Mapping) -> object:
    targets = {"stored_user_input": StoredUserInput, "stored_payload": StoredPayload,
               "pending_tool_continuation": PendingToolContinuation,
               "recovery_terminal_handoff": RecoveryTerminalHandoff, "result_record": ResultRecord}
    target = targets.get(value.get("$dataclass"))
    if target is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_domain(value, target, _execution_payload_codec(_CURRENT_CODEC, payloads), persisted=True)


def decode_execution_content(
    value: Mapping[str, JsonValue], payloads: Mapping[tuple[str, str, str] | NativeTextReference | NativeUserInputReference, StoredPayload | StoredUserInput],
    *, timeline: bool = False, result: bool = False,
) -> ExecutionRecord | ExecutionTimelineRecord | ExecutionResultRecord:
    """Inject verified immutable bodies without encoding or decoding them again."""
    envelope = decode_envelope(value)
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    current = _execution_payload_codec(codec, payloads)
    if result:
        return decode_execution_result(value, codec=current)
    if timeline:
        return decode_execution_timeline(value, codec=current)
    if envelope.value.get("type") != "execution_record":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _decode_domain(envelope.value.get("payload"), ExecutionRecord, current, persisted=True)


def decode_execution_timeline(value: Mapping[str, JsonValue], *, codec: _VersionCodec | None = None) -> ExecutionTimelineRecord:
    """Validate the envelope and strictly decode only the read-only timeline fields."""
    envelope = decode_envelope(value)
    try:
        if envelope.version not in _VERSION_CODECS or envelope.value.get("type") != "execution_record":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload = envelope.value["payload"]
        if not isinstance(payload, Mapping) or payload.get("$dataclass") != "execution_record":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_required_keys(payload, frozenset({"$dataclass", "schema", "fields"}))
        if type(payload["schema"]) is not int or payload["schema"] != envelope.version:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        source = payload["fields"]
        if not isinstance(source, Mapping) or "runtime_policy" in source or "runtime_overrides" in source:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        codec = codec or _VERSION_CODECS[envelope.version]
        source = _execution_terminal_fields(source, codec, persisted=True)
        values = {}
        for name, (target, _init) in _dataclass_fields(ExecutionTimelineRecord).items():
            if name in source:
                values[name] = _decode_domain(source[name], target, codec, persisted=True)
                continue
            # The projection inherits omission semantics from its canonical owner,
            # not a separately maintained set of timeline defaults.
            declaration = next(field for field in fields(ExecutionRecord) if field.name == name)
            if declaration.default is not MISSING:
                values[name] = declaration.default
            elif declaration.default_factory is not MISSING:
                values[name] = declaration.default_factory()
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ExecutionTimelineRecord(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def encode_session_turn_commit(value: SessionTurnCommit) -> Mapping[str, JsonValue]:
    """Encode the current flat conversation commit row."""
    encoded = cast(Mapping[str, JsonValue], encode_domain(value))
    return {"version": 1, **cast(Mapping[str, JsonValue], encoded["fields"])}


def decode_session_turn_commit(value: object) -> SessionTurnCommit:
    """Decode terminal turn metadata with the same contract as domain values."""
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return decode_domain({
        "$dataclass": "session_turn_commit",
        "fields": {key: item for key, item in value.items() if key != "version"},
    }, SessionTurnCommit)


def _decode_enveloped_domain(
    value: Mapping[str, JsonValue],
    target: type[DomainT],
    *,
    payload_transform: Callable[[JsonValue], JsonValue] | None = None,
) -> DomainT:
    """Decode persisted domain data without losing its envelope version."""
    envelope = decode_envelope(value)
    _require_required_keys(envelope.value, frozenset({"type", "payload"}))
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


def _execution_binding_summary(binding: AgentBindingContract | TaskBindingContract) -> dict[str, JsonValue]:
    """Derive read-only view fields from the complete admitted contract."""
    if isinstance(binding, AgentBindingContract):
        return {"kind": "agent", "id": binding.agent_spec.id,
                "revision": binding.agent_spec.revision, "max_attempts": None,
                "capability_refs": _encode_persisted_domain(binding.capability_refs), "binding_digest": binding.binding_digest}
    return {"kind": "task", "id": binding.id, "revision": binding.revision,
            "max_attempts": binding.max_attempts, "capability_refs": None, "binding_digest": binding.binding_digest}


def _execution_content_reference(value: Mapping, marker: str) -> tuple[str, str, str]:
    expected = {marker, "archive_scope", "content_scope"}
    if set(value) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    identity, scope, content_scope = value[marker], value["archive_scope"], value["content_scope"]
    validate_archive_group_id(scope)
    decode_archive_scope(content_scope)
    validate_archive_content_identity(identity)
    return scope, content_scope, identity


TOOL_PAYLOAD_CODEC = "execution-payload"
_TOOL_PAYLOAD_SLOTS = ("arguments_payload", "result_payload", "error_payload")


def tool_payload_slots(value: object) -> Iterator[tuple[str, Mapping]]:
    if not isinstance(value, Mapping) or value.get("$dataclass") != "tool_operation":
        return
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for name in _TOOL_PAYLOAD_SLOTS:
        payload = fields.get(name)
        if payload is not None:
            if not isinstance(payload, Mapping) or payload.get("$dataclass") != "stored_payload":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if native_tool_result_reference(payload) is not None and name != "result_payload":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if native_tool_arguments_reference(payload) is not None and name != "arguments_payload":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            yield name, payload


def native_tool_result_reference(payload: Mapping) -> NativeToolResultReference | None:
    """Recognize only the declared immutable tool receipt projection wrapper."""
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = fields.get("content")
    if not isinstance(content, Mapping) or "$native_tool_result" not in content:
        return None
    _validate_dataclass_wrapper(payload, StoredPayload, _CURRENT_CODEC, persisted=True)
    if set(fields) != {"content"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return NativeToolResultReference.from_json(content)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def native_tool_arguments_reference(payload: Mapping) -> NativeToolArgumentsReference | None:
    """Recognize only the declared immutable tool argument projection wrapper."""
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = fields.get("content")
    if not isinstance(content, Mapping) or "$native_tool_arguments" not in content:
        return None
    _validate_dataclass_wrapper(payload, StoredPayload, _CURRENT_CODEC, persisted=True)
    if set(fields) != {"content"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return NativeToolArgumentsReference.from_json(content)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def tool_payload_reference(payload: Mapping) -> tuple[str, str, str] | None:
    """Tool facts use the same canonical Step payload owner as continuations."""
    if native_tool_result_reference(payload) is not None or native_tool_arguments_reference(payload) is not None:
        return None
    return payload_reference(payload)


def map_tool_payload_slots(value: Mapping, mapper: Callable[[Mapping], JsonValue]) -> dict:
    slots = tuple(tool_payload_slots(value))
    if not slots:
        return dict(value)
    fields = dict(value["fields"])
    for name, payload in slots:
        fields[name] = mapper(payload)
    return {**value, "fields": fields}


def encode_agent_binding_content(binding: AgentBindingContract) -> bytes:
    """Use the canonical complete contract shared by execution and capture owners."""
    if not isinstance(binding, AgentBindingContract):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return canonical_json_bytes(_encode_persisted_domain(binding))


def decode_agent_binding_content(raw: bytes) -> AgentBindingContract:
    """Validate the persisted contract wrapper before restoring a capture slot."""
    try:
        return _decode_domain(json.loads(raw), AgentBindingContract, _CURRENT_CODEC, persisted=True)
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def agent_binding_content_summary(binding: AgentBindingContract) -> dict[str, JsonValue]:
    """Derive the existing execution binding view, never a second identity."""
    return _execution_binding_summary(binding)


def binding_reference(value: object) -> tuple[str, str, str] | None:
    """Recognize only the fixed complete-binding owner, without a derived copy."""
    if not isinstance(value, Mapping) or "$execution_binding" not in value:
        return None
    return _execution_content_reference(value, "$execution_binding")


def iter_runtime_object_refs(
    value: JsonValue,
    *,
    default_domain: PersistenceDomain,
) -> Iterator[tuple[PersistenceDomain, ObjectRef]]:
    """Yield object references without depending on a storage backend."""
    yield from _iter_runtime_object_refs(value, default_domain, _CURRENT_CODEC)


def iter_runtime_object_dependencies(
    reference: ObjectRef,
    payload: bytes,
    *,
    default_domain: PersistenceDomain,
) -> Iterator[tuple[PersistenceDomain, ObjectRef]]:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(manifest, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    if reference.key.startswith("v1/asset-snapshot/"):
        if (
            manifest.get("kind") != "asset-snapshot"
            or manifest.get("format_version") != 1
            or not isinstance(manifest.get("entries"), list)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for item in cast(list[object], manifest["entries"]):
            if not isinstance(item, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            content = item.get("content")
            if not isinstance(content, Mapping) or not {
                "key",
                "digest",
                "size",
            }.issubset(content):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            size = content["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                nested = ObjectRef(
                    reference.store_id,
                    cast(str, content["key"]),
                    cast(str, content["digest"]),
                    size,
                )
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            yield default_domain, nested
        return


def _iter_runtime_object_refs(
    value: object,
    domain: PersistenceDomain,
    codec: _VersionCodec,
) -> Iterator[tuple[PersistenceDomain, ObjectRef]]:
    if isinstance(value, StoredPayload):
        for reference in value.object_refs():
            yield domain, reference
        return
    if isinstance(value, ObjectRef):
        yield domain, value
        return
    if isinstance(value, RuntimePayloadRef):
        source_domain = value.source_domain or domain
        yield from _iter_runtime_object_refs(value.payload, source_domain, codec)
        return
    if isinstance(value, AgentBindingContract):
        return
    if isinstance(value, Mapping):
        dataclass_name = value.get("$dataclass")
        if dataclass_name == codec.wire_ids.get(TaskNode):
            fields = value.get("fields")
            if not isinstance(fields, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            task_input = _decode_domain(fields.get("input"), object, codec, persisted="schema" in value)
            prompt = task_input.get("prompt") if isinstance(task_input, Mapping) else None
            if (fields.get("input_semantics") == "agent-task-input-v1"
                    and isinstance(prompt, Mapping) and prompt.get("kind") == "stored-user-content-v1"):
                stored = cast(StoredUserInput, _decode_domain(prompt.get("value"), StoredUserInput, codec))
                yield from _iter_runtime_object_refs(stored.payload, PersistenceDomain.TASK, codec)
        if dataclass_name == codec.wire_ids.get(ToolOperationRecord):
            fields_value = map_tool_payload_slots(value,
                lambda payload: None if tool_payload_reference(payload) is not None or native_tool_result_reference(payload) is not None or native_tool_arguments_reference(payload) is not None else payload)["fields"]
            for item in fields_value.values():
                yield from _iter_runtime_object_refs(item, domain, codec)
            return
        if dataclass_name == codec.wire_ids.get(ExecutionRecord):
            fields_value = value.get("fields")
            if not isinstance(fields_value, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            def skip_content(payload: Mapping) -> JsonValue:
                return None if payload_reference(payload) is not None or native_text_reference(payload) is not None or native_user_input_reference(payload) is not None else payload
            fields_value = map_execution_payload_slots(value, skip_content)["fields"]
            for key, item in fields_value.items():
                if (
                    key == "binding"
                    and isinstance(item, Mapping)
                    and "$dataclass" not in item
                ):
                    if binding_reference(item) is not None:
                        # The shared binding owns no object references. Its
                        # content row is retained by the archive closure walker.
                        continue
                    binding = _decode_external(
                        item,
                        AgentBindingContract,
                        codec,
                        persisted=True,
                    )
                    if not isinstance(binding, AgentBindingContract):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    continue
                yield from _iter_runtime_object_refs(item, domain, codec)
            return
        if dataclass_name == codec.wire_ids.get(PendingDeferredCall):
            fields_value = value.get("fields")
            if not isinstance(fields_value, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for key, item in fields_value.items():
                yield from _iter_runtime_object_refs(
                    item, PersistenceDomain.CHECKPOINT if key == "arguments_payload" else domain, codec,
                )
            return
        if dataclass_name == codec.wire_ids[RuntimePayloadRef]:
            decoded = _decode_domain(
                value,
                RuntimePayloadRef,
                codec,
                persisted="schema" in value,
            )
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[StoredUserInput]:
            decoded = _decode_domain(value, StoredUserInput, codec, persisted="schema" in value)
            yield from _iter_runtime_object_refs(decoded.payload, domain, codec)
            return
        if dataclass_name == codec.wire_ids[StoredPayload]:
            decoded = _decode_domain(value, StoredPayload, codec, persisted="schema" in value)
            yield from _iter_runtime_object_refs(decoded, domain, codec)
            return
        if dataclass_name == codec.wire_ids[ObjectRef]:
            decoded = _decode_domain(value, ObjectRef, codec, persisted="schema" in value)
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
    default: PersistenceDomain,
    codec: _VersionCodec,
) -> PersistenceDomain:
    if value is None:
        return default
    try:
        result = _decode_domain(value, PersistenceDomain, codec)
    except AIError:
        raise
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(result, PersistenceDomain):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_domain(
    value: object,
    target: object,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> object:
    if target is str or target is bool or target is int or target is float:
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(value, target) or (
            target is int and isinstance(value, bool)
        ):
            raise TypeError("scalar value has the wrong type")
        if target is float:
            return _require_finite_wire_float(value)
        return value
    if target is Any or target is object:
        return _decode_any(value, codec, persisted=persisted)
    if target == "JsonValue":
        return _decode_domain(value, JsonValue, codec, persisted=persisted)
    if isinstance(target, ForwardRef):
        if target == _JSON_VALUE_FORWARD_REF:
            return _decode_domain(value, JsonValue, codec, persisted=persisted)
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    origin = get_origin(target)
    arguments = get_args(target)
    if origin is Literal:
        for literal in arguments:
            if isinstance(literal, float) and not math.isfinite(literal):
                raise TypeError("Runtime v1 schema literal requires finite floats")
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
        diagnostic_error: AIError | None = None
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _decode_domain(
                    value,
                    candidate,
                    codec,
                    persisted=persisted,
                )
            except (TypeError, ValueError, KeyError):
                continue
            except AIError as error:
                if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                    raise
                if error.safe_details.get("missing_fields"):
                    diagnostic_error = error
                continue
        if diagnostic_error is not None:
            raise diagnostic_error
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
                _decode_domain(
                    item,
                    arguments[0],
                    codec,
                    persisted=persisted,
                )
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
            "Runtime v1 does not support set values",
        )
    if target is frozenset or origin is frozenset:
        item_type = arguments[0] if arguments else Any
        return _decode_frozenset_items(
            value,
            item_type,
            codec,
            persisted=persisted,
        )
    if isinstance(origin, type) and issubclass(origin, Mapping):
        key_type, item_type = arguments if len(arguments) == 2 else (Any, Any)
        return _decode_mapping_items(
            value,
            key_type,
            item_type,
            codec,
            persisted=persisted,
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
        _require_required_keys(value, frozenset({"$datetime"}))
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
                "Runtime v1 datetime wire is not canonical",
            )
        return result
    if target is bytes:
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_required_keys(value, frozenset({"$bytes"}))
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
                "Runtime v1 bytes wire is not canonical",
            )
        return result
    if isinstance(target, type) and target in codec.external_schema_types:
        return _decode_external(
            value,
            target,
            codec,
            persisted=persisted,
        )
    if isinstance(target, type) and is_dataclass(target):
        return _decode_dataclass(value, target, codec, persisted=persisted)
    return _decode_any(value, codec, persisted=persisted)


def _require_finite_wire_float(value: float) -> float:
    if not math.isfinite(value):
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            "Runtime v1 wire contains a non-finite float",
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
    _require_required_keys(value, frozenset({"$enum", "value"}))
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


@lru_cache(maxsize=None)
def _dataclass_fields(
    target: type[object],
) -> Mapping[str, tuple[object, bool]]:
    # Registered wire classes have stable declarations; payloads remain uncached.
    hints = get_type_hints(target)
    return MappingProxyType({
        field.name: (
            hints.get(field.name, Any),
            field.init,
        )
        for field in fields(target)
    })


def _validate_dataclass_wrapper(
    value: object,
    target: type,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> tuple[str, Mapping[str, object]]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if persisted:
        _require_required_keys(
            value, frozenset({"$dataclass", "schema", "fields"})
        )
    else:
        _require_required_keys(value, frozenset({"$dataclass", "fields"}))

    wire_id = value.get("$dataclass")
    if not isinstance(wire_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    expected_target = codec.domain_types.get(wire_id)
    if expected_target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if codec.wire_ids.get(target) != wire_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    if persisted:
        schema = value["schema"]
        if isinstance(schema, bool) or not isinstance(schema, int) or schema < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if schema != codec.version:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    return wire_id, raw_fields


def _decode_dataclass(
    value: object,
    target: type,
    codec: _VersionCodec,
    *,
    persisted: bool = False,
) -> object:
    wire_id, raw_fields = _validate_dataclass_wrapper(value, target, codec, persisted=persisted)
    decoder = codec.dataclass_decoders.get(wire_id)
    if decoder is not None:
        try:
            decoded = decoder(raw_fields, codec, persisted)
            if not isinstance(decoded, target):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return decoded
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    return _decode_dataclass_fields(raw_fields, target, codec, persisted=persisted)


def _decode_dataclass_fields(
    raw_fields: Mapping[str, object],
    target: type,
    codec: _VersionCodec,
    *,
    persisted: bool,
) -> object:
    try:
        declared = _dataclass_fields(target)
    except (NameError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    missing_fields = sorted(
        field.name for field in fields(target) if field.name not in raw_fields
        and (not field.init or field.default is MISSING and field.default_factory is MISSING)
    )
    if missing_fields:
        raise AIError(
            ErrorCode.STORAGE_INTEGRITY_ERROR,
            safe_details={"wire_type": _codec_wire_type_id(target, codec), "missing_fields": missing_fields},
        )
    kwargs: dict[str, object] = {}
    post_init_fields: dict[str, object] = {}
    for field_name, (field_type, field_init) in declared.items():
        if field_name not in raw_fields:
            continue
        try:
            decoded = _decode_domain(
                raw_fields[field_name],
                field_type,
                codec,
                persisted=persisted,
            )
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if field_init:
            kwargs[field_name] = decoded
        else:
            post_init_fields[field_name] = decoded
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
        matched = set(value.keys()).intersection(_RESERVED_WIRE_TAGS)
        if len(matched) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if len(matched) == 0:
            if any(
                isinstance(key, str) and key.startswith("$")
                for key in value
            ):
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
                value,
                Any,
                Any,
                codec,
                persisted=persisted,
            )
        if tag == "$tuple":
            return tuple(
                _decode_any(item, codec, persisted=persisted)
                for item in _unwrap_tagged_list(value, "$tuple")
            )
        return _decode_frozenset_items(
            value,
            Any,
            codec,
            persisted=persisted,
        )
    if isinstance(value, list):
        return [
            _decode_any(item, codec, persisted=persisted)
            for item in value
        ]
    if isinstance(value, float):
        return _require_finite_wire_float(value)
    return value


def _enum_type(wire_id: str, codec: _VersionCodec) -> type[Enum]:
    try:
        return codec.enum_types[wire_id]
    except KeyError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error


_RESERVED_WIRE_TAGS = frozenset(
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
            item,
            item_type,
            codec,
            persisted=persisted,
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
            pair[0],
            key_type,
            codec,
            persisted=persisted,
        )
        try:
            hash(decoded_key)
        except TypeError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if decoded_key in result:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result[decoded_key] = _decode_domain(
            pair[1],
            item_type,
            codec,
            persisted=persisted,
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


def _operation_key_wire(value: str) -> bytes:
    if not 2 <= len(value) <= 2 * OPERATION_KEY_MAX_BYTES:
        raise ValueError("operation key encoding length is invalid")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("operation key must be canonical lowercase hex")
    return result


def _stream_key_wire(value: str) -> bytes:
    if not 2 <= len(value) <= 2 * STREAM_KEY_MAX_BYTES:
        raise ValueError("stream key encoding length is invalid")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("stream key encoding is not canonical")
    return validate_stream_key(result)


def _record_key_wire(value: str) -> bytes:
    if not 2 <= len(value) <= 2 * RECORD_KEY_MAX_BYTES:
        raise ValueError("record key encoding length is invalid")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("record key encoding is not canonical")
    return validate_record_key(result)


def _optional_parent_key(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 2 <= len(value) <= 2 * PARENT_KEY_MAX_BYTES:
        raise ValueError("parent key encoding length is invalid")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("parent key encoding is not canonical")
    return validate_parent_key(result)


def _optional_relation_key(value: object) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 2 <= len(value) <= 2 * RELATION_KEY_MAX_BYTES:
        raise ValueError("relation key must be bounded canonical lowercase hex or null")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("relation key must be canonical lowercase hex")
    return validate_relation_key(result)


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
    _require_required_keys(envelope.value, frozenset({"type", "payload"}))
    codec = _VERSION_CODECS.get(envelope.version)
    if codec is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    kind = envelope.value.get("type")
    if not isinstance(kind, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    targets = {
        "agent_run_record": AgentRunRecord,
        "step_event": StepEvent,
        "stored_agent_run_checkpoint": StoredAgentRunCheckpoint,
        "model_interaction": ModelInteractionRecord,
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
        raise RuntimeError("Runtime v1 codec registry is invalid")
    if _CURRENT_CODEC is not _VERSION_CODECS[1]:
        raise RuntimeError("Runtime v1 current codec is invalid")
    wire_ids = tuple(wire_id for wire_id, _target in _V1_WIRE_TYPES)
    enum_wire_ids = tuple(wire_id for wire_id, _target in _V1_ENUM_WIRE_TYPES)
    if len(wire_ids) != len(set(wire_ids)):
        raise RuntimeError("Runtime v1 wire ids are not unique")
    if len(enum_wire_ids) != len(set(enum_wire_ids)):
        raise RuntimeError("Runtime v1 enum wire ids are not unique")
    domain_wire_ids = set(wire_ids)
    if set(_CURRENT_CODEC.domain_types) != domain_wire_ids:
        raise RuntimeError("Runtime v1 domain type registry is incomplete")
    if set(_CURRENT_CODEC.wire_ids.values()) != set(wire_ids):
        raise RuntimeError("Runtime v1 domain wire-id registry is incomplete")
    if set(_CURRENT_CODEC.enum_types) != set(enum_wire_ids):
        raise RuntimeError("Runtime v1 enum type registry is incomplete")
    if set(_CURRENT_CODEC.enum_wire_ids.values()) != set(enum_wire_ids):
        raise RuntimeError("Runtime v1 enum wire-id registry is incomplete")
    if not set(_V1_DATACLASS_ENCODERS).issubset(domain_wire_ids):
        raise RuntimeError(
            "Runtime v1 dataclass encoder mapping contains an unknown type"
        )
    if not set(_V1_DATACLASS_DECODERS).issubset(domain_wire_ids):
        raise RuntimeError(
            "Runtime v1 dataclass decoder mapping contains an unknown type"
        )


_validate_v1_codec_definition()


__all__ = ["native_tool_arguments_reference", "decode_execution_producer_authority", "decode_execution_terminal_details", "decode_execution_payload", "EXECUTION_POLICY_FIELDS", "runtime_policy_reference", "encode_agent_binding_content", "decode_agent_binding_content", "agent_binding_content_summary", "native_user_input_reference", "validate_native_user_input", "native_tool_result_reference", "validate_native_text_payload", "native_text_reference", "TOOL_PAYLOAD_CODEC", "tool_payload_slots", "tool_payload_reference", "map_tool_payload_slots", "decode_execution_metadata", 'decode_execution_control', 'decode_execution_result', 'decode_execution_retention', 'CURRENT_DATA_VERSION', 'CanonicalEnvelope', '_decode_enveloped_domain', '_decode_step_envelope', 'binding_reference', 'execution_payload_reference', 'execution_payload_slots', 'payload_reference', 'map_execution_payload_slots', 'encode_execution_content', 'decode_execution_content', 'decode_domain', 'decode_envelope', 'decode_fact', 'decode_operation', 'decode_record', 'decode_session_turn_commit', 'decode_execution_timeline', 'decode_execution_header', 'decode_execution_view', 'decode_execution_producer_control', 'encode_domain', 'encode_envelope', 'encode_fact', 'encode_operation', 'encode_record', 'encode_session_turn_commit', 'iter_runtime_object_dependencies', 'iter_runtime_object_refs', 'parse_envelope', 'wire_type_id']
