#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate logical Runtime snapshot identities before restore."""

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
from ._repository_common import (
    canonical_record_identity,
    project_record,
    record_state,
    restore_lease_fields,
)
from ._step_contracts import RunRecord
from ._store import (
    StoredAlias,
    StoredFact,
    StoredOperation,
    StoredRecord,
    alias_digest,
    operation_key,
    parent_digest,
    partition_digest,
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

    graph_parents = _task_graph_parents(
        namespace,
        tenant_id,
        domain,
        values,
    )
    for record in records:
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
            [value.step_run_id, value.tool_call_id],
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
    if any(
        not isinstance(value[name], str) or not value[name]
        for name in ("session_id", "execution_id")
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    sequence = value["sequence"]
    start = value["start_message_index"]
    end = value["end_message_index"]
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
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

    if isinstance(value, TaskGraphView):
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
    elif isinstance(value, TranscriptHeadRecord):
        if value.owner_domain.value != domain.value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        anchor_kind = (
            "conversation_history"
            if domain is RuntimeDomain.CONVERSATION
            else "step_run"
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
        _require_anchor(namespace, tenant_id, domain, records, "step_run", run_id)
        if parent not in records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sort_key = run_id
    elif isinstance(value, RunRecord):
        if value.conversation_id is not None:
            scope = scope_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "conversation",
                value.conversation_id,
            )
        if value.parent_run_id is not None:
            parent = parent_digest(
                namespace,
                tenant_id,
                domain.value,
                kind,
                "parent",
                value.parent_run_id,
            )
        sort_key = sortable_timestamp(value.started_at, value.run_id)
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
    if isinstance(value, ConversationHistoryRecord):
        return value.history_id
    if isinstance(value, ConversationHistoryIndexNodeRecord):
        return value.node_id
    if isinstance(value, (ExecutionHistoryHeadRecord, ExecutionHistorySealRecord)):
        return value.execution_id
    if isinstance(value, TaskGraphAdmission):
        return value.graph_id
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
    if isinstance(value, TranscriptHeadRecord):
        return value.owner_id
    if isinstance(value, TranscriptSeekRecord):
        return [value.owner_id, value.dimension.value, value.block_start]
    if isinstance(value, RunRecord):
        return value.run_id
    if isinstance(value, ContextProjection):
        return record.sort_key
    try:
        return canonical_record_identity(kind, value)
    except TypeError as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error

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
    sequence = cast(int, fields["sequence"])
    _require_anchor(
        namespace, tenant_id, domain, records, "session", session_id
    )
    identity = [session_id, sequence]
    return StoredRecord(
        record_key_digest(
            namespace,
            tenant_id,
            domain.value,
            "session_turn_commit",
            identity,
        ),
        partition_digest(
            namespace,
            tenant_id,
            domain.value,
            "session_turn_commit",
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
        and left.partition_digest == right.partition_digest
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
    fact_owners: set[bytes] = set()
    for fact in facts:
        if previous_stream is None or fact.stream_digest > previous_stream:
            if previous_owner is not None:
                _validate_fact_high_water(previous_owner, previous_sequence)
            if fact.sequence != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_stream = fact.stream_digest
            previous_sequence = 1
        elif fact.stream_digest == previous_stream:
            if fact.sequence != previous_sequence + 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_sequence = fact.sequence
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        owner = values.get(fact.owner_key_digest)
        if fact.owner_key_digest not in records or owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if fact.stream_digest != _fact_stream(
            namespace, tenant_id, domain, fact, owner
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_owner = owner
        fact_owners.add(fact.owner_key_digest)

    if previous_owner is not None:
        _validate_fact_high_water(previous_owner, previous_sequence)

    for owner_key, owner in values.items():
        if owner_key in fact_owners:
            continue
        if isinstance(owner, ExecutionRecord) and owner.event_sequence != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, TranscriptHeadRecord) and owner.chunk_count != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_fact_high_water(owner: object, sequence: int) -> None:
    if isinstance(owner, ExecutionRecord) and owner.event_sequence != sequence:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if isinstance(owner, TranscriptHeadRecord) and owner.chunk_count != sequence:
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
    if isinstance(owner, RunRecord):
        relation = {
            "step_event": "event",
            "step_snapshot": "snapshot",
            "model_interaction": "interaction",
        }.get(fact.kind)
        if relation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return relation, owner.run_id
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
    for fact in facts:
        owner = values.get(fact.owner_key_digest)
        if owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(owner, ExecutionRecord):
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


__all__ = ["canonical_snapshot_indexes", "validate_snapshot_domain"]
