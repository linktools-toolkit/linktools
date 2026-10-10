#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate logical Runtime snapshot identities and cross-domain references."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import timezone
from typing import cast

from ...core import BudgetUsage, OperationLedgerInput, canonical_sha256
from ._budget_records import BudgetModelReservation, BudgetToolReservation
from ...errors import AIError, ErrorCode
from ...evaluation import EvidenceBundle, EvaluationReport, ComparisonReport
from ._evaluation_records import EvaluationCaseRecord, EvaluationDatasetRecord, EvaluationTombstone, EvaluationContentTombstone, EvaluationCleanupRecord
from ...task import (
    TaskGraphAdmission,
    TaskGraphSubmission,
    TaskSubmissionRef,
    TaskGraphView,
    TaskNode,
    TaskResultRef,
    TaskNodeView,
    TaskResultRecord,
)
from ._codec import _decode_enveloped_domain, decode_envelope
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ContextProjection,
    ConversationHistoryIndexNodeRecord,
    ConversationHistoryRecord,
    EvaluationRecord,
    ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord,
    ExecutionRecord,
    ExternalCallRecord,
    IdempotencyRecord,
    MemoryRecord,
    ModelInteractionRecord,
    RecoveryCheckpoint,
    SessionRecord,
    StoredAgentRunCheckpoint,
    TaskPreparedInputRecord,
    ToolOperationRecord,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptOrigin,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
    TranscriptSpanRef,
)
from ._plan import RuntimeDomain
from ._repository_common import (
    canonical_record_identity,
    project_record,
    record_state,
    restore_lease_fields,
)
from ._step_contracts import TOOL_ERROR_CODE_METADATA_KEY, AgentRunRecord, StepEvent
from ._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    alias_digest,
    operation_key,
    parent_digest,
    record_key_digest,
    scope_digest,
    sequence_key,
    sortable_identity,
    sortable_timestamp,
    stream_digest,
    subject_digest,
)


_ALLOWED_RECORD_KINDS = {
    RuntimeDomain.CONVERSATION: frozenset(
        {
            "session",
            "session_turn_commit",
            "conversation_history",
            "conversation_index_node",
            "transcript_head",
            "transcript_seek",
            "context_projection",
            "agent_run",
            "model_interaction",
        }
    ),
    RuntimeDomain.EXECUTION: frozenset(
        {
            "budget_scope",
            "budget_model",
            "budget_tool",
            "execution",
            "execution_history_head",
            "execution_history_seal",
            "idempotency",
            "transcript_head",
            "transcript_seek",
            "context_projection",
            "agent_run",
            "model_interaction",
            "history_association",
        }
    ),
    RuntimeDomain.MEMORY: frozenset({"memory"}),
    RuntimeDomain.ARTIFACT: frozenset({"artifact"}),
    RuntimeDomain.TASK: frozenset(
        {
            "task_graph",
            "task_admission",
            "task_submission",
            "task_submission_payload",
            "task_node_definition",
            "task_node_state",
            "task_result",
            "task_prepared_input",
        }
    ),
    RuntimeDomain.EVALUATION: frozenset({"evaluation", "idempotency", "evaluation_case", "evaluation_dataset", "evaluation_evidence", "evaluation_report", "evaluation_comparison", "evaluation_tombstone", "evaluation_content_tombstone", "evaluation_cleanup"}),
    RuntimeDomain.RECOVERY: frozenset(
        {
            "recovery_checkpoint",
            "approval",
            "external_call",
            "tool_operation",
            "transcript_head",
            "transcript_seek",
            "context_projection",
            "agent_run",
            "model_interaction",
        }
    ),
}

_RECORD_TYPES = {
    "budget_scope": BudgetUsage,
    "budget_model": BudgetModelReservation,
    "budget_tool": BudgetToolReservation,
    "session": SessionRecord,
    "conversation_history": ConversationHistoryRecord,
    "conversation_index_node": ConversationHistoryIndexNodeRecord,
    "transcript_head": TranscriptHeadRecord,
    "transcript_seek": TranscriptSeekRecord,
    "context_projection": ContextProjection,
    "execution": ExecutionRecord,
    "execution_history_head": ExecutionHistoryHeadRecord,
    "execution_history_seal": ExecutionHistorySealRecord,
    "idempotency": IdempotencyRecord,
    "memory": MemoryRecord,
    "artifact": ArtifactRecord,
    "evaluation": EvaluationRecord,
    "evaluation_tombstone": EvaluationTombstone,
    "evaluation_content_tombstone": EvaluationContentTombstone,
    "evaluation_cleanup": EvaluationCleanupRecord,
    "evaluation_case": EvaluationCaseRecord,
    "evaluation_dataset": EvaluationDatasetRecord,
    "evaluation_evidence": EvidenceBundle,
    "evaluation_report": EvaluationReport,
    "evaluation_comparison": ComparisonReport,
    "recovery_checkpoint": RecoveryCheckpoint,
    "approval": ApprovalRecord,
    "external_call": ExternalCallRecord,
    "tool_operation": ToolOperationRecord,
    "task_graph": TaskGraphView,
    "task_admission": TaskGraphAdmission,
    "task_submission": TaskSubmissionRef,
    "task_submission_payload": TaskGraphSubmission,
    "task_node_definition": TaskNode,
    "task_node_state": TaskNodeView,
    "task_result": TaskResultRecord,
    "task_prepared_input": TaskPreparedInputRecord,
    "agent_run": AgentRunRecord,
    "model_interaction": ModelInteractionRecord,
}


def canonical_snapshot_indexes(
    *,
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    records: tuple[StoredRecord, ...],
    facts: tuple[StoredFact, ...],
    operations: tuple[StoredOperation, ...],
) -> tuple[tuple[StoredAlias, ...], Mapping[bytes, int]]:
    """Rebuild derived snapshot indexes from their durable semantic owners."""
    _records_by_key, values = _decode_snapshot_records(domain, records)
    _validate_budget_projections(values)
    aliases = _canonical_aliases(
        namespace,
        tenant_id,
        domain,
        values,
    )
    sequences = _canonical_sequences(
        namespace,
        tenant_id,
        domain,
        facts,
        operations,
        values,
    )
    return aliases, sequences


def validate_snapshot_domain(
    *,
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    records: tuple[StoredRecord, ...],
    aliases: tuple[StoredAlias, ...],
    facts: tuple[StoredFact, ...],
    operations: tuple[StoredOperation, ...],
    sequences: Mapping[bytes, int],
) -> None:
    """Reject physical identities that cannot represent the decoded v1 facts."""
    records_by_key, values = _decode_snapshot_records(domain, records)

    _validate_budget_projections(values)

    graph_parents = _task_graph_parents(
        namespace,
        tenant_id,
        domain,
        values,
    )
    associations = _history_association_records(namespace, tenant_id, domain, facts, values)
    associations_by_owner: dict[bytes | None, dict[bytes, StoredRecord]] = {}
    for key, association in associations.items():
        associations_by_owner.setdefault(association.parent_digest, {})[key] = association
    event_counts: dict[bytes, int] = {}
    for fact in facts:
        if fact.kind == "step_event":
            event_counts[fact.owner_key_digest] = max(event_counts.get(fact.owner_key_digest, 0), fact.sequence)
    # Associations are derived indexes; older stores contain only their facts.
    # Any stored index must still agree exactly with its authoritative event.
    for record in records:
        if record.kind == "history_association":
            if record.sort_key == "coverage:event":
                _validate_history_association_coverage(
                    namespace, tenant_id, domain, record, records_by_key,
                    event_counts.get(record.parent_digest, 0), values,
                    associations_by_owner.get(record.parent_digest, {}),
                )
                continue
            expected = associations.get(record.key_digest)
            if expected is None or not _same_physical_identity(record, expected) or record.data != expected.data:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            continue
        expected = _expected_record(
            namespace,
            tenant_id,
            domain,
            record,
            values[record.key_digest],
            records_by_key,
            graph_parents,
        )
        if not _same_physical_identity(record, expected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    expected_aliases = _canonical_aliases(
        namespace,
        tenant_id,
        domain,
        values,
    )
    if aliases != expected_aliases:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    _validate_facts(
        namespace,
        tenant_id,
        domain,
        facts,
        records_by_key,
        values,
    )
    _validate_operations(namespace, tenant_id, domain, operations)
    expected_sequences = _canonical_sequences(
        namespace,
        tenant_id,
        domain,
        facts,
        operations,
        values,
    )
    if dict(sequences) != dict(expected_sequences):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def validate_snapshot_references(
    records: Mapping[RuntimeDomain, tuple[StoredRecord, ...]],
    facts: Mapping[RuntimeDomain, tuple[StoredFact, ...]],
) -> None:
    """Validate cross-domain references against the snapshot's durable owners."""
    transcript_heads: dict[tuple[RuntimeDomain, str], TranscriptHeadRecord] = {}
    interactions: list[ModelInteractionRecord] = []
    for domain, domain_records in records.items():
        for record in domain_records:
            if record.kind == "transcript_head":
                head = _decode_enveloped_domain(record.data, TranscriptHeadRecord)
                transcript_heads[domain, head.owner_id] = head
            elif record.kind == "model_interaction":
                interactions.append(_decode_enveloped_domain(record.data, ModelInteractionRecord))
    for domain_facts in facts.values():
        interactions.extend(
            _decode_enveloped_domain(fact.data, ModelInteractionRecord)
            for fact in domain_facts if fact.kind == "model_interaction"
        )
    for interaction in interactions:
        for context in (interaction.request_context, interaction.response_context):
            if context is None:
                continue
            for item in context.items:
                if not isinstance(item, TranscriptSpanRef):
                    continue
                head = transcript_heads.get((item.source_domain, item.owner_id))
                if head is None or item.end > head.message_count:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    budgets: dict[str, BudgetUsage] = {}
    for record in records.get(RuntimeDomain.EXECUTION, ()):
        if record.kind != "budget_scope":
            continue
        usage = _decode_enveloped_domain(record.data, BudgetUsage)
        if usage.scope_id in budgets:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        budgets[usage.scope_id] = usage

    for record in records.get(RuntimeDomain.TASK, ()):
        if record.kind == "task_admission":
            admission = _decode_enveloped_domain(record.data, TaskGraphAdmission)
        elif record.kind == "task_submission_payload":
            # Preparation captures a scope before persisting its payload;
            # read-only descriptions persist neither payload nor scope.
            admission = _decode_enveloped_domain(record.data, TaskGraphSubmission).admission
        else:
            continue
        if admission.budget_scope_id is None:
            continue
        usage = budgets.get(admission.budget_scope_id)
        if usage is None or usage.limits != admission.budget:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_snapshot_records(
    domain: RuntimeDomain,
    records: tuple[StoredRecord, ...],
) -> tuple[Mapping[bytes, StoredRecord], Mapping[bytes, object]]:
    records_by_key: dict[bytes, StoredRecord] = {}
    values: dict[bytes, object] = {}
    for record in records:
        if record.key_digest in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.kind not in _ALLOWED_RECORD_KINDS[domain]:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        records_by_key[record.key_digest] = record
        values[record.key_digest] = _decode_record(record)
    return records_by_key, values


def _canonical_aliases(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    values: Mapping[bytes, object],
) -> tuple[StoredAlias, ...]:
    if domain is not RuntimeDomain.RECOVERY:
        return ()
    aliases: dict[bytes, bytes] = {}
    for record_key, value in values.items():
        if not isinstance(value, ToolOperationRecord):
            continue
        digest = alias_digest(
            namespace,
            tenant_id,
            domain.value,
            "tool_call",
            [value.agent_run_id, value.tool_call_id],
        )
        current = aliases.get(digest)
        if current is not None and current != record_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        aliases[digest] = record_key
    return tuple(
        StoredAlias(alias, aliases[alias])
        for alias in sorted(aliases)
    )


def _decode_record(record: StoredRecord) -> object:
    if record.kind == "session_turn_commit":
        return _decode_session_turn_commit(record.data)
    if record.kind == "history_association":
        sequence = record.data.get("sequence")
        if set(record.data) != {"sequence"} or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return dict(record.data)
    target = _RECORD_TYPES.get(record.kind)
    if target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    transform = (
        (lambda payload: restore_lease_fields(payload, target))
        if target in {TaskNodeView, ToolOperationRecord}
        else None
    )
    value = _decode_enveloped_domain(
        record.data,
        target,
        payload_transform=transform,
    )
    if isinstance(value, (TaskNodeView, ToolOperationRecord)):
        try:
            value = replace(
                value,
                owner=record.lease_owner,
                fence=record.lease_fence,
                lease_expires_at=record.lease_expires_at,
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return value


def _history_association_records(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    facts: tuple[StoredFact, ...],
    values: Mapping[bytes, object],
) -> Mapping[bytes, StoredRecord]:
    if domain is not RuntimeDomain.EXECUTION:
        return {}
    result: dict[bytes, StoredRecord] = {}
    for fact in facts:
        if fact.kind != "step_event":
            continue
        run = values.get(fact.owner_key_digest)
        if not isinstance(run, AgentRunRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        event = _decode_enveloped_domain(fact.data, StepEvent)
        if event.agent_run_id != run.agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        locators = (
            [("trace", str(fact.sequence))]
            if event.event_type.startswith(("MODEL_REQUEST_", "TOOL_CALL_")) else []
        )
        request = event.metadata.get("linktools.ai.model_request_seq")
        message = event.metadata.get("linktools.ai.message_seq")
        if event.event_type.startswith("MODEL_REQUEST_") and request is not None:
            locators.append((f"request:{event.event_type}", request))
        if event.event_type == "MODEL_REQUEST_SUCCEEDED" and message is not None:
            locators.append(("response", message))
        if (event.event_type in {"TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED", "TOOL_CALL_FAILED"}
                and event.tool_call_id is not None
                and not (event.event_type == "TOOL_CALL_FAILED"
                         and event.metadata.get(TOOL_ERROR_CODE_METADATA_KEY) == ErrorCode.TOOL_EFFECT_UNKNOWN.value)):
            locators.append((event.event_type, event.tool_call_id))
        for family, identity in locators:
            key = record_key_digest(
                namespace, tenant_id, domain.value, "history_association",
                [run.agent_run_id, family, identity],
            )
            sort_key = (
                f"trace:{event.timestamp.astimezone(timezone.utc).isoformat(timespec='microseconds')}:{fact.sequence:020d}"
                if family == "trace" else f"{fact.sequence:020d}"
            )
            candidate = StoredRecord(
                key, None, fact.owner_key_digest, "history_association", sort_key,
                None, 1, None, 0, None, {"sequence": fact.sequence},
            )
            previous = result.get(key)
            if previous is not None and previous != candidate:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[key] = candidate
    return result


def _validate_history_association_coverage(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    record: StoredRecord,
    records: Mapping[bytes, StoredRecord],
    event_count: int,
    values: Mapping[bytes, object],
    associations: Mapping[bytes, StoredRecord],
) -> None:
    run = values.get(record.parent_digest)
    if domain is not RuntimeDomain.EXECUTION or not isinstance(run, AgentRunRecord):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    covered = cast(int, record.data["sequence"])
    expected = StoredRecord(
        record_key_digest(namespace, tenant_id, domain.value, "history_association",
                          [run.agent_run_id, "coverage", "event"]),
        None, record.parent_digest, "history_association", "coverage:event",
        None, record.storage_version, None, 0, None, {"sequence": covered},
    )
    if covered > event_count or record.storage_version < 1 or not _same_physical_identity(record, expected):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for key, association in associations.items():
        if association.data["sequence"] <= covered:
            stored = records.get(key)
            if stored is None or not _same_physical_identity(stored, association) or stored.data != association.data:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_session_turn_commit(value: object) -> Mapping[str, object]:
    expected = {
        "version",
        "session_id",
        "turn_seq",
        "execution_id",
        "start_message_index",
        "end_message_index",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if any(
        not isinstance(value[name], str) or not value[name]
        for name in ("session_id", "execution_id")
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    turn_seq = value["turn_seq"]
    start = value["start_message_index"]
    end = value["end_message_index"]
    if (
        isinstance(turn_seq, bool)
        or not isinstance(turn_seq, int)
        or turn_seq < 1
        or isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _task_graph_parents(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    values: Mapping[bytes, object],
) -> Mapping[bytes, str]:
    if domain is not RuntimeDomain.TASK:
        return {}
    result: dict[bytes, str] = {}
    for value in values.values():
        if not isinstance(value, (TaskGraphView, TaskGraphAdmission)):
            continue
        digest = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            "task_node_definition",
            "graph",
            value.graph_id,
        )
        previous = result.get(digest)
        if previous is not None and previous != value.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result[digest] = value.graph_id
    return result


def _expected_record(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    record: StoredRecord,
    value: object,
    records: Mapping[bytes, StoredRecord],
    graph_parents: Mapping[bytes, str],
) -> StoredRecord:
    kind = record.kind
    if kind == "session_turn_commit":
        return _expected_session_turn_commit(
            namespace,
            tenant_id,
            domain,
            record,
            value,
            records,
        )

    identity = _record_identity(kind, value, record, graph_parents)
    scope: bytes | None = None
    parent: bytes | None = None
    state = record_state(value)
    sort_key: str | None = None

    if isinstance(value, BudgetModelReservation):
        _require_anchor(namespace, tenant_id, domain, records, "budget_scope", value.scope_id)
        scope = scope_digest(namespace, tenant_id, domain.value, kind, "budget", value.scope_id)
        state = value.status
    elif isinstance(value, BudgetToolReservation):
        _require_anchor(namespace, tenant_id, domain, records, "budget_scope", value.scope_id)
    elif isinstance(value, ExecutionRecord) and value.budget_scope_id is not None:
        _require_anchor(namespace, tenant_id, domain, records, "budget_scope", value.budget_scope_id)
    elif isinstance(value, TaskGraphView):
        state = value.status.value
    elif isinstance(value, ExecutionHistoryHeadRecord):
        state = value.state.value
        _require_anchor(
            namespace, tenant_id, domain, records, "execution", value.execution_id
        )
    elif isinstance(value, ExecutionHistorySealRecord):
        _require_anchor(
            namespace, tenant_id, domain, records, "execution", value.execution_id
        )
    elif isinstance(value, ConversationHistoryRecord):
        _require_anchor(
            namespace, tenant_id, domain, records, "session", value.session_id
        )
    elif isinstance(value, TaskSubmissionRef):
        if (
            value.namespace != namespace or value.tenant_id != tenant_id
            or record.state not in {"prepared", "admitted", "cancelled"}
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = record.state
        if state == "prepared":
            _require_anchor(
                namespace, tenant_id, domain, records,
                "task_submission_payload", value.graph_id,
            )
        elif state == "admitted":
            _require_anchor(
                namespace, tenant_id, domain, records, "task_graph", value.graph_id,
            )
    elif isinstance(value, TaskGraphSubmission):
        if value.namespace != namespace or value.admission.principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = record_key_digest(
            namespace, tenant_id, domain.value, "task_submission", value.graph.graph_id,
        )
        head = records.get(key)
        if head is None or head.state != "prepared" or _decode_record(head) != value.ref:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif isinstance(value, TaskGraphAdmission):
        if value.principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_anchor(
            namespace, tenant_id, domain, records, "task_graph", value.graph_id
        )
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "recoverable",
            "graphs",
        )
    elif isinstance(value, TaskNode):
        if record.parent_digest is None or record.parent_digest not in graph_parents:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parent = record.parent_digest
        if any(
            reference.namespace != namespace or reference.tenant_id != tenant_id
            for reference in value.input_refs.values()
            if isinstance(reference, TaskResultRef)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif isinstance(value, TaskNodeView):
        _require_anchor(
            namespace, tenant_id, domain, records, "task_graph", value.graph_id
        )
    elif isinstance(value, TaskResultRecord):
        _require_anchor(
            namespace, tenant_id, domain, records, "task_graph", value.graph_id
        )
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            value.graph_id,
        )
        parent = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            value.graph_id,
        )
    elif isinstance(value, TaskPreparedInputRecord):
        _require_anchor(
            namespace, tenant_id, domain, records, "task_graph", value.graph_id
        )
        admission_key = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "task_admission",
            value.graph_id,
        )
        admission_record = records.get(admission_key)
        if admission_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        admission = _decode_record(admission_record)
        if (
            not isinstance(admission, TaskGraphAdmission)
            or admission.initial_request_digest != value.admission_digest
            or value.tenant_id != tenant_id
            or any(
                reference.namespace != namespace
                or reference.tenant_id != tenant_id
                for _name, reference in value.source_refs
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parent = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            value.graph_id,
        )
    elif isinstance(value, TranscriptHeadRecord):
        if (
            value.owner_domain.value != domain.value
            or value.pending is not None and value.pending.source_domain is not domain
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        anchor_kind = (
            "conversation_history"
            if domain is RuntimeDomain.CONVERSATION
            else "agent_run"
        )
        _require_anchor(
            namespace, tenant_id, domain, records, anchor_kind, value.owner_id
        )
    elif isinstance(value, TranscriptSeekRecord):
        if value.dimension is not TranscriptSeekDimension.MESSAGE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parent = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "transcript_head",
            value.owner_id,
        )
        if parent not in records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = f"b:{value.block_start:020d}"
    elif isinstance(value, ContextProjection):
        agent_run_id = record.sort_key
        if not agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = agent_run_id
        parent = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "transcript_head",
            agent_run_id,
        )
        anchor_kind = "conversation_history" if domain is RuntimeDomain.CONVERSATION else "agent_run"
        _require_anchor(namespace, tenant_id, domain, records, anchor_kind, agent_run_id)
        if parent not in records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = agent_run_id
    elif isinstance(value, ModelInteractionRecord):
        _require_anchor(
            namespace, tenant_id, domain, records, "agent_run", value.agent_run_id
        )
        parent = record_key_digest(
            namespace, tenant_id, domain.value, "agent_run", value.agent_run_id
        )
        sort_key = f"m:{value.model_request_seq:020d}"
        state = value.status
    elif isinstance(value, AgentRunRecord):
        if value.agent_conversation_id is not None:
            scope = scope_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "conversation",
                value.agent_conversation_id,
            )
        if value.parent_agent_run_id is not None:
            parent = parent_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "parent",
                value.parent_agent_run_id,
            )
        sort_key = sortable_timestamp(value.started_at, value.agent_run_id)
    elif isinstance(value, MemoryRecord):
        path = value.metadata.get("path")
        if not isinstance(path, str) or not path or not path.isascii():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = path

    return project_record(
        namespace=namespace,
        tenant_id=tenant_id,
        domain=domain,
        kind=kind,
        identity=identity,
        value=value,
        scope=scope,
        parent=parent,
        state=state,
        sort_key=sort_key,
        storage_version=record.storage_version,
    )

def _record_identity(
    kind: str,
    value: object,
    record: StoredRecord,
    graph_parents: Mapping[bytes, str],
) -> object:
    if isinstance(value, BudgetUsage):
        return value.scope_id
    if isinstance(value, BudgetModelReservation):
        return [value.scope_id, value.request_id]
    if isinstance(value, BudgetToolReservation):
        return [value.scope_id, value.call_id]
    if isinstance(value, ConversationHistoryRecord):
        return value.history_id
    if isinstance(value, ConversationHistoryIndexNodeRecord):
        return value.node_id
    if isinstance(value, (ExecutionHistoryHeadRecord, ExecutionHistorySealRecord)):
        return value.execution_id
    if isinstance(value, (TaskGraphAdmission, TaskSubmissionRef)):
        return value.graph_id
    if isinstance(value, TaskGraphSubmission):
        return value.graph.graph_id
    if isinstance(value, TaskNode):
        graph_id = (
            None
            if record.parent_digest is None
            else graph_parents.get(record.parent_digest)
        )
        if graph_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return [graph_id, value.node_id]
    if isinstance(value, TaskResultRecord):
        return [value.graph_id, value.node_id]
    if isinstance(value, TaskPreparedInputRecord):
        return [value.graph_id, value.node_id]
    if isinstance(value, TranscriptHeadRecord):
        return value.owner_id
    if isinstance(value, TranscriptSeekRecord):
        return [value.owner_id, value.dimension.value, value.block_start]
    if isinstance(value, ModelInteractionRecord):
        return [value.agent_run_id, value.model_request_seq]
    if isinstance(value, AgentRunRecord):
        return value.agent_run_id
    if isinstance(value, ContextProjection):
        return record.sort_key
    try:
        return canonical_record_identity(kind, value)
    except TypeError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error


def _validate_budget_projections(values: Mapping[bytes, object]) -> None:
    scopes = {value.scope_id: value for value in values.values() if isinstance(value, BudgetUsage)}
    observed = {scope_id: BudgetUsage(scope_id, value.limits) for scope_id, value in scopes.items()}
    for value in values.values():
        if not isinstance(value, (BudgetModelReservation, BudgetToolReservation)):
            continue
        usage = observed.get(value.scope_id)
        if usage is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(value, BudgetModelReservation):
            usage = replace(
                usage, model_requests=usage.model_requests + 1,
                total_tokens=usage.total_tokens + (0 if value.total_tokens is None else value.total_tokens),
                in_flight_model_requests=usage.in_flight_model_requests + int(value.status == "in_flight"),
                unknown_model_requests=usage.unknown_model_requests + int(value.status == "unknown"),
            )
        else:
            usage = replace(usage, tool_calls=usage.tool_calls + 1)
        observed[value.scope_id] = usage
    if observed != scopes:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

def _expected_session_turn_commit(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    record: StoredRecord,
    value: object,
    records: Mapping[bytes, StoredRecord],
) -> StoredRecord:
    fields = cast(Mapping[str, object], value)
    session_id = cast(str, fields["session_id"])
    turn_seq = cast(int, fields["turn_seq"])
    _require_anchor(
        namespace, tenant_id, domain, records, "session", session_id
    )
    identity = [session_id, turn_seq]
    return StoredRecord(
        record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "session_turn_commit",
            identity,
        ),
        scope_digest(
            namespace,
            tenant_id,
            domain.value,
            "session_turn_commit",
            "session",
            session_id,
        ),
        None,
        "session_turn_commit",
        sortable_identity(identity),
        None,
        record.storage_version,
        None,
        0,
        None,
        record.data,
    )


def _require_anchor(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    records: Mapping[bytes, StoredRecord],
    kind: str,
    identity: object,
) -> None:
    key = record_key_digest(namespace, tenant_id, domain.value, kind, identity)
    if key not in records:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _same_physical_identity(left: StoredRecord, right: StoredRecord) -> bool:
    return (
        left.key_digest == right.key_digest
        and left.scope_digest == right.scope_digest
        and left.parent_digest == right.parent_digest
        and left.kind == right.kind
        and left.sort_key == right.sort_key
        and left.state == right.state
        and left.storage_version == right.storage_version
        and left.lease_owner == right.lease_owner
        and left.lease_fence == right.lease_fence
        and left.lease_expires_at == right.lease_expires_at
    )


def _validate_facts(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    facts: tuple[StoredFact, ...],
    records: Mapping[bytes, StoredRecord],
    values: Mapping[bytes, object],
) -> None:
    previous_stream: bytes | None = None
    previous_sequence = 0
    previous_owner: object | None = None
    previous_fact: StoredFact | None = None
    fact_owners: set[bytes] = set()
    transcript_owners: set[bytes] = set()
    current_pending_owners: set[bytes] = set()
    pending_keys: set[str] = set()
    for fact in facts:
        if previous_stream is None or fact.stream_digest > previous_stream:
            if previous_owner is not None and previous_fact is not None:
                _validate_fact_high_water(previous_owner, previous_sequence, previous_fact)
            if fact.sequence != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_stream = fact.stream_digest
            previous_sequence = 1
            pending_keys.clear()
        elif fact.stream_digest == previous_stream:
            if fact.sequence != previous_sequence + 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_sequence = fact.sequence
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        owner = values.get(fact.owner_key_digest)
        if fact.owner_key_digest not in records or owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if fact.stream_digest != _fact_stream(namespace, tenant_id, domain, fact, owner):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, AgentRunRecord) and fact.kind == "step_checkpoint":
            checkpoint = _decode_enveloped_domain(fact.data, StoredAgentRunCheckpoint)
            transcript_owner_id = (
                owner.metadata.get("history_id") or owner.agent_run_id
                if domain is RuntimeDomain.CONVERSATION else owner.agent_run_id
            )
            head_key = record_key_digest(
                namespace, tenant_id, domain.value, "transcript_head", transcript_owner_id,
            )
            head = values.get(head_key)
            if (
                checkpoint.agent_run_id != owner.agent_run_id
                or checkpoint.state != fact.state
                or not isinstance(head, TranscriptHeadRecord)
                or (checkpoint.transcript_message_count is not None
                    and checkpoint.transcript_message_count > head.message_count)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, TranscriptHeadRecord):
            if fact.kind == "transcript_chunk":
                transcript_owners.add(fact.owner_key_digest)
            elif fact.kind == "transcript_pending_part":
                key = cast(str, decode_envelope(fact.data).value["pending_key"])
                if key in pending_keys:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                pending_keys.add(key)
                chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
                if chunk.first_message_index == owner.message_count:
                    current_pending_owners.add(fact.owner_key_digest)
        previous_owner = owner
        previous_fact = fact
        fact_owners.add(fact.owner_key_digest)

    if previous_owner is not None and previous_fact is not None:
        _validate_fact_high_water(previous_owner, previous_sequence, previous_fact)

    for owner_key, owner in values.items():
        if isinstance(owner, ExecutionRecord) and owner_key not in fact_owners and owner.event_seq != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, TranscriptHeadRecord):
            if owner.chunk_count != 0 and owner_key not in transcript_owners:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if owner.pending_part_count != 0 and owner_key not in current_pending_owners:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_fact_high_water(owner: object, sequence: int, fact: StoredFact) -> None:
    if isinstance(owner, ExecutionRecord) and owner.event_seq != sequence:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if isinstance(owner, TranscriptHeadRecord):
        if fact.kind == "transcript_chunk" and owner.chunk_count != sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if fact.kind == "transcript_pending_part":
            chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
            if chunk.first_message_index == owner.message_count and owner.pending_part_count != sequence:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _fact_stream(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    fact: StoredFact,
    owner: object,
) -> bytes:
    relation, value = _fact_storage_identity(domain, fact, owner)
    return stream_digest(
        namespace,
        tenant_id,
        domain.value,
        relation,
        value,
    )


def _fact_storage_identity(
    domain: RuntimeDomain,
    fact: StoredFact,
    owner: object,
) -> tuple[str, object]:
    if isinstance(owner, ExecutionRecord):
        return "execution", owner.execution_id
    if isinstance(owner, SessionRecord):
        if fact.kind != "session_turn":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            set(fact.data) != {"version", "execution_id"}
            or isinstance(fact.data.get("version"), bool)
            or not isinstance(fact.data.get("version"), int)
            or fact.data.get("version") != 1
            or not isinstance(fact.data.get("execution_id"), str)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_id = cast(str, fact.data["execution_id"])
        if fact.subject_digest != subject_digest(execution_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return "session_turn", [owner.session_id]
    if isinstance(owner, TranscriptHeadRecord):
        if fact.kind == "transcript_pending_part":
            chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
            key = decode_envelope(fact.data).value.get("pending_key")
            if (
                not isinstance(key, str) or not key
                or chunk.owner_id != owner.owner_id
                or chunk.first_message_index > owner.message_count
                or chunk.message_count != 1
                or chunk.content.source_domain is not domain
                or chunk.origin is not TranscriptOrigin.RAW
                or fact.subject_digest is not None
                or fact.state is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return "transcript_pending_parts", [owner.owner_id, chunk.first_message_index]
        if fact.kind != "transcript_chunk":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
        if chunk.owner_id != owner.owner_id or fact.state != chunk.origin.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        relation = (
            "history_transcript"
            if domain is RuntimeDomain.CONVERSATION
            else "run_transcript"
        )
        return relation, owner.owner_id
    if isinstance(owner, AgentRunRecord):
        if fact.kind == "model_interaction":
            interaction = _decode_enveloped_domain(fact.data, ModelInteractionRecord)
            expected_subject = bytes.fromhex(canonical_sha256({
                "agent_run_id": interaction.agent_run_id,
                "model_request_seq": interaction.model_request_seq,
            }))
            if (
                interaction.agent_run_id != owner.agent_run_id
                or interaction.model_request_seq != fact.sequence
                or interaction.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}
                or interaction.status != fact.state
                or fact.subject_digest != expected_subject
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return "interaction", owner.agent_run_id
        relation = {
            "step_event": "event",
            "step_checkpoint": "checkpoint",
        }.get(fact.kind)
        if relation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return relation, owner.agent_run_id
    if isinstance(owner, TaskGraphView):
        return "task_event", owner.graph_id
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)



def _canonical_sequences(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    facts: tuple[StoredFact, ...],
    operations: tuple[StoredOperation, ...],
    values: Mapping[bytes, object],
) -> Mapping[bytes, int]:
    sequences: dict[bytes, int] = {}
    interactions: dict[str, dict[int, ModelInteractionRecord]] = {}
    for value in values.values():
        if isinstance(value, ModelInteractionRecord):
            admitted = interactions.setdefault(value.agent_run_id, {})
            if value.model_request_seq in admitted:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            admitted[value.model_request_seq] = value
    for fact in facts:
        if fact.kind != "model_interaction":
            continue
        value = _decode_enveloped_domain(fact.data, ModelInteractionRecord)
        admitted = interactions.setdefault(value.agent_run_id, {})
        previous = admitted.get(value.model_request_seq)
        if previous is not None and previous != value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        admitted[value.model_request_seq] = value
    for agent_run_id, admitted in interactions.items():
        if set(admitted) != set(range(1, len(admitted) + 1)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = sequence_key(
            namespace, tenant_id, domain.value, "interaction", agent_run_id
        )
        sequences[key] = len(admitted)
    for fact in facts:
        owner = values.get(fact.owner_key_digest)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, ExecutionRecord) or fact.kind == "transcript_pending_part":
            continue
        relation, value = _fact_storage_identity(domain, fact, owner)
        key = sequence_key(
            namespace,
            tenant_id,
            domain.value,
            relation,
            value,
        )
        sequences[key] = max(sequences.get(key, 0), fact.sequence)

    for operation in operations:
        value = _decode_enveloped_domain(operation.data, OperationLedgerInput)
        if value.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = sequence_key(
            namespace,
            tenant_id,
            domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )
        sequences[key] = max(sequences.get(key, 0), operation.sequence)
    return sequences

def _validate_operations(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    operations: tuple[StoredOperation, ...],
) -> None:
    positions: set[tuple[bytes, int]] = set()
    for operation in operations:
        value = _decode_enveloped_domain(operation.data, OperationLedgerInput)
        expected_stream = stream_digest(
            namespace,
            tenant_id,
            domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )
        position = (operation.stream_digest, operation.sequence)
        if (
            value.tenant_id != tenant_id
            or operation.key_digest
            != operation_key(
                namespace,
                tenant_id,
                domain.value,
                value.operation_id,
            )
            or operation.stream_digest != expected_stream
            or operation.state != value.status.value
            or operation.compactable != value.compactable
            or operation.sequence < 1
            or position in positions
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        positions.add(position)


__all__ = [
    "canonical_snapshot_indexes", "validate_snapshot_domain", "validate_snapshot_references",
]
