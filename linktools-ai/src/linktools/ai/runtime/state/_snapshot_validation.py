#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Runtime snapshot logical identities before restore."""

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from ...core import OperationLedgerInput
from ...errors import AIError, ErrorCode
from ...task import (
    TaskGraphAdmission,
    TaskGraphView,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
)
from ._codec import _decode_enveloped_domain
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
    RecoveryCheckpoint,
    SessionRecord,
    ToolOperationRecord,
    TranscriptChunk,
    TranscriptHeadRecord,
    TranscriptSeekDimension,
    TranscriptSeekRecord,
)
from ._plan import RuntimeDomain
from ._repository_common import _restore_lease_fields
from ._step_contracts import RunRecord
from ._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    operation_key,
    parent_digest,
    partition_digest,
    record_key_digest,
    scope_digest,
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
            "step_run",
        }
    ),
    RuntimeDomain.EXECUTION: frozenset(
        {
            "execution",
            "execution_history_head",
            "execution_history_seal",
            "idempotency",
            "transcript_head",
            "transcript_seek",
            "context_projection",
            "step_run",
        }
    ),
    RuntimeDomain.MEMORY: frozenset({"memory"}),
    RuntimeDomain.ARTIFACT: frozenset({"artifact"}),
    RuntimeDomain.TASK: frozenset(
        {
            "task_graph",
            "task_admission",
            "task_node_definition",
            "task_node_state",
            "task_result",
        }
    ),
    RuntimeDomain.EVALUATION: frozenset({"evaluation", "idempotency"}),
    RuntimeDomain.RECOVERY: frozenset(
        {
            "recovery_checkpoint",
            "approval",
            "external_call",
            "tool_operation",
            "transcript_head",
            "transcript_seek",
            "context_projection",
            "step_run",
        }
    ),
}

_RECORD_TYPES = {
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
    "recovery_checkpoint": RecoveryCheckpoint,
    "approval": ApprovalRecord,
    "external_call": ExternalCallRecord,
    "tool_operation": ToolOperationRecord,
    "task_graph": TaskGraphView,
    "task_admission": TaskGraphAdmission,
    "task_node_definition": TaskNode,
    "task_node_state": TaskNodeView,
    "task_result": TaskResultRecord,
    "step_run": RunRecord,
}


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
) -> Mapping[bytes, object]:
    """Validate one exported domain without writing the restore target."""
    allowed = _ALLOWED_RECORD_KINDS[domain]
    values: dict[bytes, object] = {}
    records_by_key: dict[bytes, StoredRecord] = {}
    for record in records:
        if record.key_digest in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.kind not in allowed:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if record.partition_digest != partition_digest(
            namespace,
            tenant_id,
            domain.value,
            record.kind,
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = _decode_snapshot_record(record)
        records_by_key[record.key_digest] = record
        values[record.key_digest] = value

    graph_parent_ids = _task_graph_parent_identities(
        namespace,
        tenant_id,
        domain,
        values,
    )
    for record in records:
        _validate_record_shape(
            namespace,
            tenant_id,
            domain,
            record,
            values[record.key_digest],
            records_by_key,
            graph_parent_ids,
        )

    for alias in aliases:
        if alias.record_key_digest not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    _validate_facts(
        namespace,
        tenant_id,
        domain,
        facts,
        records_by_key,
        values,
    )
    _validate_operations(
        namespace,
        tenant_id,
        domain,
        operations,
    )
    if any(
        not isinstance(key, bytes)
        or len(key) != 32
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for key, value in sequences.items()
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return values


def _decode_snapshot_record(record: StoredRecord) -> object:
    if record.kind == "session_turn_commit":
        return _decode_session_turn_commit(record.data)
    target = _RECORD_TYPES.get(record.kind)
    if target is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    transform = (
        (lambda payload: _restore_lease_fields(payload, target))
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


def _decode_session_turn_commit(value: object) -> Mapping[str, object]:
    expected = {
        "version",
        "session_id",
        "sequence",
        "execution_id",
        "start_message_index",
        "end_message_index",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for name in ("session_id", "execution_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for name in ("sequence", "start_message_index", "end_message_index"):
        current = value[name]
        if isinstance(current, bool) or not isinstance(current, int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if (
        cast(int, value["sequence"]) < 1
        or cast(int, value["start_message_index"]) < 0
        or cast(int, value["end_message_index"])
        <= cast(int, value["start_message_index"])
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _task_graph_parent_identities(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    values: Mapping[bytes, object],
) -> Mapping[bytes, str]:
    if domain is not RuntimeDomain.TASK:
        return {}
    parents: dict[bytes, str] = {}
    for value in values.values():
        if not isinstance(value, (TaskGraphView, TaskGraphAdmission)):
            continue
        graph_id = value.graph_id
        candidate = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            "task_node_definition",
            "graph",
            graph_id,
        )
        previous = parents.get(candidate)
        if previous is not None and previous != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parents[candidate] = graph_id
    return parents


def _validate_record_shape(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    record: StoredRecord,
    value: object,
    records_by_key: Mapping[bytes, StoredRecord],
    graph_parent_ids: Mapping[bytes, str],
) -> None:
    kind = record.kind
    identity: object
    scope: bytes | None = None
    parent: bytes | None = None
    state: str | None = None
    sort_key: str
    lease = (None, 0, None)

    if kind == "session":
        candidate = cast(SessionRecord, value)
        identity = candidate.session_id
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "owner",
            candidate.owner_principal_id,
        )
        state = candidate.status.value
    elif kind == "session_turn_commit":
        candidate = cast(Mapping[str, object], value)
        identity = [candidate["session_id"], candidate["sequence"]]
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "session",
            candidate["session_id"],
        )
        owner_key = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "session",
            candidate["session_id"],
        )
        if owner_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif kind == "conversation_history":
        candidate = cast(ConversationHistoryRecord, value)
        identity = candidate.history_id
        session_key = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "session",
            candidate.session_id,
        )
        if session_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif kind == "conversation_index_node":
        identity = cast(ConversationHistoryIndexNodeRecord, value).node_id
    elif kind == "execution":
        candidate = cast(ExecutionRecord, value)
        identity = candidate.execution_id
        if candidate.session_id is not None:
            scope = scope_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "session",
                candidate.session_id,
            )
        if candidate.parent_execution_id is not None:
            parent = parent_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "execution",
                candidate.parent_execution_id,
            )
        state = candidate.status.value
    elif kind == "execution_history_head":
        candidate = cast(ExecutionHistoryHeadRecord, value)
        identity = candidate.execution_id
        _require_record_anchor(
            namespace, tenant_id, domain, records_by_key, "execution", identity
        )
        state = candidate.state.value
    elif kind == "execution_history_seal":
        identity = cast(ExecutionHistorySealRecord, value).execution_id
        _require_record_anchor(
            namespace, tenant_id, domain, records_by_key, "execution", identity
        )
    elif kind == "idempotency":
        candidate = cast(IdempotencyRecord, value)
        identity = [candidate.scope, candidate.idempotency_key_digest]
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "resource",
            [candidate.resource_kind.value, candidate.resource_id],
        )
        state = candidate.status.value
    elif kind == "memory":
        candidate = cast(MemoryRecord, value)
        identity = candidate.memory_id
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "memory_scope",
            candidate.memory_scope_digest,
        )
        path = candidate.metadata.get("path")
        if not isinstance(path, str) or not path or not path.isascii():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = path
        _validate_record_physical(
            namespace,
            tenant_id,
            domain,
            record,
            identity,
            scope,
            parent,
            state,
            sort_key,
            lease,
        )
        return
    elif kind == "artifact":
        candidate = cast(ArtifactRecord, value)
        identity = candidate.artifact_id
        scope = _execution_scope(
            namespace, tenant_id, domain, kind, candidate.execution_id
        )
    elif kind == "evaluation":
        candidate = cast(EvaluationRecord, value)
        identity = candidate.evaluation_id
        scope = _execution_scope(
            namespace, tenant_id, domain, kind, candidate.execution_id
        )
        state = candidate.status.value
    elif kind == "recovery_checkpoint":
        candidate = cast(RecoveryCheckpoint, value)
        identity = candidate.execution_id
        state = candidate.state.value
    elif kind == "approval":
        candidate = cast(ApprovalRecord, value)
        identity = candidate.approval_id
        scope = _execution_scope(
            namespace, tenant_id, domain, kind, candidate.execution_id
        )
        state = candidate.status.value
    elif kind == "external_call":
        candidate = cast(ExternalCallRecord, value)
        identity = candidate.call_id
        scope = _execution_scope(
            namespace, tenant_id, domain, kind, candidate.execution_id
        )
        state = candidate.status.value
    elif kind == "tool_operation":
        candidate = cast(ToolOperationRecord, value)
        identity = candidate.tool_operation_id
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "step_run",
            candidate.step_run_id,
        )
        parent = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "execution",
            candidate.execution_id,
        )
        state = candidate.status.value
        lease = (candidate.owner, candidate.fence, candidate.lease_expires_at)
    elif kind == "task_graph":
        candidate = cast(TaskGraphView, value)
        identity = candidate.graph_id
        state = candidate.status.value
    elif kind == "task_admission":
        candidate = cast(TaskGraphAdmission, value)
        if candidate.principal.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = candidate.graph_id
        graph_key = record_key_digest(
            namespace, tenant_id, domain.value, "task_graph", candidate.graph_id
        )
        if graph_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "recoverable",
            "graphs",
        )
    elif kind == "task_node_definition":
        candidate = cast(TaskNode, value)
        if record.parent_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_id = graph_parent_ids.get(record.parent_digest)
        if graph_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = [graph_id, candidate.node_id]
        for reference in candidate.input_refs.values():
            if reference.namespace != namespace or reference.tenant_id != tenant_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parent = record.parent_digest
    elif kind == "task_node_state":
        candidate = cast(TaskNodeView, value)
        identity = [candidate.graph_id, candidate.node_id]
        graph_key = record_key_digest(
            namespace, tenant_id, domain.value, "task_graph", candidate.graph_id
        )
        if graph_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parent = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            candidate.graph_id,
        )
        state = candidate.status.value
        lease = (candidate.owner, candidate.fence, candidate.lease_expires_at)
    elif kind == "task_result":
        candidate = cast(TaskResultRecord, value)
        identity = [candidate.graph_id, candidate.node_id]
        graph_key = record_key_digest(
            namespace, tenant_id, domain.value, "task_graph", candidate.graph_id
        )
        if graph_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope = scope_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            candidate.graph_id,
        )
        parent = parent_digest(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            candidate.graph_id,
        )
    elif kind == "transcript_head":
        candidate = cast(TranscriptHeadRecord, value)
        if candidate.owner_domain.value != domain.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = candidate.owner_id
        anchor_kind = (
            "conversation_history"
            if domain is RuntimeDomain.CONVERSATION
            else "step_run"
        )
        anchor = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            anchor_kind,
            candidate.owner_id,
        )
        if anchor not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    elif kind == "transcript_seek":
        candidate = cast(TranscriptSeekRecord, value)
        if candidate.dimension is not TranscriptSeekDimension.MESSAGE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = [
            candidate.owner_id,
            candidate.dimension.value,
            candidate.block_start,
        ]
        parent = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "transcript_head",
            candidate.owner_id,
        )
        if parent not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = f"b:{candidate.block_start:020d}"
        _validate_record_physical(
            namespace,
            tenant_id,
            domain,
            record,
            identity,
            scope,
            parent,
            state,
            sort_key,
            lease,
        )
        return
    elif kind == "context_projection":
        run_id = record.sort_key
        if not run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = run_id
        parent = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "transcript_head",
            run_id,
        )
        run_key = record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "step_run",
            run_id,
        )
        if parent not in records_by_key or run_key not in records_by_key:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = run_id
        _validate_record_physical(
            namespace,
            tenant_id,
            domain,
            record,
            identity,
            scope,
            parent,
            state,
            sort_key,
            lease,
        )
        return
    elif kind == "step_run":
        candidate = cast(RunRecord, value)
        identity = candidate.run_id
        if candidate.conversation_id is not None:
            scope = scope_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "conversation",
                candidate.conversation_id,
            )
        if candidate.parent_run_id is not None:
            parent = parent_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "parent",
                candidate.parent_run_id,
            )
        sort_key = sortable_timestamp(candidate.started_at, candidate.run_id)
        _validate_record_physical(
            namespace,
            tenant_id,
            domain,
            record,
            identity,
            scope,
            parent,
            state,
            sort_key,
            lease,
        )
        return
    else:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

    sort_key = sortable_identity(identity)
    _validate_record_physical(
        namespace,
        tenant_id,
        domain,
        record,
        identity,
        scope,
        parent,
        state,
        sort_key,
        lease,
    )


def _require_record_anchor(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    records_by_key: Mapping[bytes, StoredRecord],
    kind: str,
    identity: object,
) -> None:
    key = record_key_digest(
        namespace,
        tenant_id,
        domain.value,
        kind,
        identity,
    )
    if key not in records_by_key:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _execution_scope(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    kind: str,
    execution_id: str,
) -> bytes:
    return scope_digest(
        namespace,
        tenant_id,
        domain.value,
        kind,
        "execution",
        execution_id,
    )


def _validate_record_physical(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    record: StoredRecord,
    identity: object,
    scope: bytes | None,
    parent: bytes | None,
    state: str | None,
    sort_key: str,
    lease: tuple[object, int, object],
) -> None:
    expected_key = record_key_digest(
        namespace,
        tenant_id,
        domain.value,
        record.kind,
        identity,
    )
    if (
        record.key_digest != expected_key
        or record.scope_digest != scope
        or record.parent_digest != parent
        or record.sort_key != sort_key
        or record.state != state
        or record.lease_owner != lease[0]
        or record.lease_fence != lease[1]
        or record.lease_expires_at != lease[2]
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_facts(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    facts: tuple[StoredFact, ...],
    records_by_key: Mapping[bytes, StoredRecord],
    values: Mapping[bytes, object],
) -> None:
    previous_by_stream: dict[bytes, int] = {}
    for fact in sorted(facts, key=lambda value: (value.stream_digest, value.sequence)):
        owner_record = records_by_key.get(fact.owner_key_digest)
        owner = values.get(fact.owner_key_digest)
        if owner_record is None or owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_stream = _fact_stream(
            namespace,
            tenant_id,
            domain,
            fact,
            owner,
        )
        if fact.stream_digest != expected_stream:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous = previous_by_stream.get(fact.stream_digest, 0)
        if fact.sequence != previous + 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_by_stream[fact.stream_digest] = fact.sequence

        if isinstance(owner, ExecutionRecord) and fact.sequence > owner.event_sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, TranscriptHeadRecord):
            if fact.sequence > owner.chunk_count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _fact_stream(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    fact: StoredFact,
    owner: object,
) -> bytes:
    if isinstance(owner, ExecutionRecord):
        return stream_digest(
            namespace,
            tenant_id,
            domain.value,
            "execution",
            owner.execution_id,
        )
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
        return stream_digest(
            namespace,
            tenant_id,
            domain.value,
            "session_turn",
            [owner.session_id],
        )
    if isinstance(owner, TranscriptHeadRecord):
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
        return stream_digest(
            namespace,
            tenant_id,
            domain.value,
            relation,
            owner.owner_id,
        )
    if isinstance(owner, RunRecord):
        relation = {
            "step_event": "event",
            "step_snapshot": "snapshot",
            "model_interaction": "interaction",
        }.get(fact.kind)
        if relation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stream_digest(
            namespace,
            tenant_id,
            domain.value,
            relation,
            owner.run_id,
        )
    if isinstance(owner, TaskGraphView):
        return stream_digest(
            namespace,
            tenant_id,
            domain.value,
            "task_event",
            owner.graph_id,
        )
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_operations(
    namespace: str,
    tenant_id: str,
    domain: RuntimeDomain,
    operations: tuple[StoredOperation, ...],
) -> None:
    seen_positions: set[tuple[bytes, int]] = set()
    for operation in operations:
        value = _decode_enveloped_domain(operation.data, OperationLedgerInput)
        if value.tenant_id != tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_key = operation_key(
            namespace,
            tenant_id,
            domain.value,
            value.operation_id,
        )
        expected_stream = stream_digest(
            namespace,
            tenant_id,
            domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )
        position = (operation.stream_digest, operation.sequence)
        if (
            operation.key_digest != expected_key
            or operation.stream_digest != expected_stream
            or operation.state != value.status.value
            or operation.compactable != value.compactable
            or operation.sequence < 1
            or position in seen_positions
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen_positions.add(position)


__all__ = ["validate_snapshot_domain"]
