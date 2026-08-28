#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral Runtime repositories built on the StateStore contract."""

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Generic, TypeVar

from linktools.core import environ

from ...core import (
    ApprovalDecision,
    ApprovalStatus,
    ExecutionEventType,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TaskStatus,
    ToolOperationStatus,
    canonical_json_bytes,
    canonical_sha256,
    operation_replay_matches,
    validate_agent_id,
    validate_lease_owner,
    validate_lease_seconds,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, StoredPayload
from ...task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphView,
    TaskLease,
    TaskNodeView,
    TaskTerminalRecord,
)
from .._tool import ToolOperationRecord
from ._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
    wire_type_id,
)
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ConversationCursor,
    ConversationHistoryIndexNodeRecord,
    ConversationHistoryRecord,
    EvaluationRecord,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionEventRecord,
    ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
    ExecutionRecord,
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
    MemoryRecord,
    RecoveryActiveRecord,
    RecoveryAdmissionRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryIntegrityReport,
    RecoveryStateRecord,
    ResultRecord,
    SessionForkResultRecord,
    SessionRecord,
    ToolOperationAdmission,
    TranscriptHeadRecord,
    TranscriptOwnerDomain,
)
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._history_index import (
    build_fork_index_node_from_roots,
)
from ._plan import RuntimeDomain
from ._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
    RecordReplacement,
    StateStore,
    StateTransaction,
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
    stream_digest,
)

_logger = environ.get_logger("ai.runtime.state.repositories")
ValueT = TypeVar("ValueT")
_RECOVERY_PAGE_SIZE = 128
_INDEX_ROOT_READ_LIMIT = 64
_ACTIVE_RECORD_UNSET = object()


class _RepositoryBase:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str, domain: RuntimeDomain) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._domain = domain

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @property
    def state_store(self) -> StateStore:
        return self._store

    def _partition(self, kind: str) -> bytes:
        return partition_digest(self._namespace, self._tenant_id, self._domain.value, kind)

    def _key(self, kind: str, identity: object) -> bytes:
        return record_key_digest(self._namespace, self._tenant_id, self._domain.value, kind, identity)

    def _scope(self, kind: str, relation: str, identity: object) -> bytes:
        return scope_digest(self._namespace, self._tenant_id, self._domain.value, kind, relation, identity)

    def _parent(self, kind: str, relation: str, identity: object) -> bytes:
        return parent_digest(self._namespace, self._tenant_id, self._domain.value, kind, relation, identity)

    def _stored(
        self,
        kind: str,
        identity: object,
        value: object,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        state: str | None = None,
    ) -> StoredRecord:
        if scope is None:
            scope = self._default_scope(kind, value)
        if parent is None:
            parent = self._default_parent(kind, value)
        lease_owner, lease_fence, lease_expires_at = _record_lease(value)
        return StoredRecord(
            self._key(kind, identity),
            self._partition(kind),
            scope,
            parent,
            kind,
            sortable_identity(identity),
            state,
            0,
            lease_owner,
            lease_fence,
            lease_expires_at,
            _domain_data(value),
        )

    def _default_scope(self, kind: str, value: object) -> bytes | None:
        if isinstance(value, SessionRecord):
            return self._scope(kind, "owner", value.owner_principal_id)
        if isinstance(value, ExecutionRecord) and value.session_id is not None:
            return self._scope(kind, "session", value.session_id)
        if isinstance(value, IdempotencyRecord):
            return self._scope(
                kind,
                "resource",
                [value.resource_kind.value, value.resource_id],
            )
        if isinstance(value, (EvaluationRecord, ArtifactRecord, ApprovalRecord, ExternalCallRecord)):
            return self._scope(kind, "execution", value.execution_id)
        if isinstance(value, MemoryRecord):
            return self._scope(kind, "memory_scope", value.memory_scope_digest)
        if isinstance(value, ToolOperationRecord):
            return self._scope(kind, "step_run", value.step_run_id)
        return None

    def _default_parent(self, kind: str, value: object) -> bytes | None:
        if isinstance(value, TaskNodeView):
            return self._parent(kind, "graph", value.graph_id)
        if isinstance(value, ExecutionRecord) and value.parent_execution_id is not None:
            return self._parent(kind, "execution", value.parent_execution_id)
        return None

    async def _record(self, key: bytes) -> StoredRecord | None:
        return await self._store.read(lambda transaction: transaction.get_record(key))

    async def _records(
        self,
        kind: str,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        states: frozenset[str] | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[StoredRecord, ...]:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)
        return await self._store.read(
            lambda transaction: transaction.list_records(
                RecordQuery(
                    partition_digest=(self._partition(kind) if scope is None and parent is None else None),
                    scope_digest=scope,
                    parent_digest=parent,
                    kind=kind,
                    states=states,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit,
                )
            )
        )

    async def _insert(self, record: StoredRecord) -> None:
        await self._store.mutate(lambda transaction: transaction.insert_record(record))

    async def _decode(self, record: StoredRecord, target: type[ValueT]) -> ValueT:
        value = _decode_enveloped_domain(
            record.data,
            target,
            payload_transform=lambda payload: _restore_lease_fields(payload, target),
        )
        if isinstance(value, (TaskNodeView, ToolOperationRecord)):
            return replace(
                value,
                owner=record.lease_owner,
                fence=record.lease_fence,
                lease_expires_at=record.lease_expires_at,
            )  # type: ignore[return-value]
        return value  # type: ignore[return-value]

    def _header(self, value: object, kind: ResourceKind, identity: str) -> ResourceRef:
        owner = value.owner_principal_id if isinstance(value, SessionRecord) else None
        return ResourceRef(kind, identity, self._tenant_id, owner)

    def _mark_changed(self) -> None:
        raise RuntimeError("storage mutation outside transaction")


class _ResourceRepository(_RepositoryBase, Generic[ValueT]):
    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        domain: RuntimeDomain,
        kind: str,
        resource_kind: ResourceKind,
        value_type: type[ValueT],
        identity_field: str,
    ) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
        self._kind = kind
        self._resource_kind = resource_kind
        self._value_type = value_type
        self._identity_field = identity_field

    async def initialize(self) -> None:
        return None

    def _identity(self, value: object) -> object:
        if isinstance(value, SessionRecord):
            return value.session_id
        if isinstance(value, ExecutionRecord):
            return value.execution_id
        if isinstance(value, MemoryRecord):
            return value.memory_id
        if isinstance(value, ArtifactRecord):
            return value.artifact_id
        if isinstance(value, EvaluationRecord):
            return value.evaluation_id
        if isinstance(value, RecoveryCheckpoint):
            return value.execution_id
        if isinstance(value, ApprovalRecord):
            return value.approval_id
        if isinstance(value, ExternalCallRecord):
            return value.call_id
        if isinstance(value, IdempotencyRecord):
            return self._identity_key(value.scope, value.idempotency_key_digest)
        raise TypeError(f"unsupported repository value: {type(value).__name__}")

    async def create(self, value: ValueT) -> ValueT:
        _require_tenant(value, self._tenant_id)
        identity = self._identity(value)
        await self._insert(self._stored(self._kind, identity, value, state=_record_state(value)))
        _logger.debug("created Runtime record: kind=%s id=%s", self._kind, identity)
        return value

    async def get(self, identity: str, *, tenant_id: str) -> ValueT | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key(self._kind, identity))
        return None if record is None else await self._decode(record, self._value_type)

    async def get_header(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        value = await self.get(identity, tenant_id=tenant_id)
        return None if value is None else self._header(value, self._resource_kind, identity)

    async def compare_and_swap(
        self,
        identity: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: ValueT,
    ) -> ValueT:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        async def mutate(transaction: StateTransaction) -> ValueT:
            current = await transaction.get_record(self._key(self._kind, identity))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, self._value_type)
            if _domain_revision(value) != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current, next_record),
                current.storage_version,
            )
            return next_record

        return await self._store.mutate(mutate)

    async def list_values(self, *, scope: bytes | None = None) -> tuple[ValueT, ...]:
        records = await self._records(self._kind, scope=scope)
        values = [await self._decode(record, self._value_type) for record in records]
        return tuple(values)


class OperationLedgerRepository(_RepositoryBase):
    def _stream(self, value: OperationLedgerInput | OperationLedgerRecord) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )

    def _stored_operation(self, value: OperationLedgerInput, sequence: int) -> StoredOperation:
        return StoredOperation(
            operation_key(self._namespace, self._tenant_id, self._domain.value, value.operation_id),
            self._stream(value),
            sequence,
            value.status.value,
            value.compactable,
            _domain_data(value),
        )

    async def append(self, value: OperationLedgerInput) -> OperationLedgerRecord:
        _require_tenant(value, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> OperationLedgerRecord:
            key = operation_key(self._namespace, self._tenant_id, self._domain.value, value.operation_id)
            existing = await transaction.get_operation(key)
            if existing is not None:
                current = _decode_operation(existing)
                if _operation_matches(current, value):
                    return current
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = await transaction.next_sequence(
                sequence_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    "operation",
                    [value.resource_kind.value, value.resource_id],
                )
            )
            await transaction.insert_operation(self._stored_operation(value, sequence))
            return _operation_record(value, sequence)

        return await self._store.mutate(mutate)

    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        if tenant_id != self._tenant_id:
            return None
        key = operation_key(self._namespace, self._tenant_id, self._domain.value, operation_id)
        stored = await self._store.read(lambda transaction: transaction.get_operation(key))
        return None if stored is None else _decode_operation(stored)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> OperationLedgerRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = operation_key(self._namespace, self._tenant_id, self._domain.value, operation_id)
        stored = await transaction.get_operation(key)
        return None if stored is None else _decode_operation(stored)

    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        key = operation_key(self._namespace, self._tenant_id, self._domain.value, operation_id)

        async def mutate(transaction: StateTransaction) -> OperationLedgerRecord:
            current = await transaction.get_operation(key)
            if current is None or current.state != expected_status.value:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if (
                current.state
                in {OperationStatus.SUCCEEDED.value, OperationStatus.FAILED.value, OperationStatus.CANCELLED.value}
                and next_record.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            candidate = _stored_from_operation(next_record, current)
            if not await transaction.replace_operation(candidate, expected_state=expected_status.value):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return next_record

        return await self._store.mutate(mutate)

    async def list_pending(
        self,
        resource_kind: ResourceKind,
        resource_id: str,
        *,
        tenant_id: str,
        limit: int,
    ) -> tuple[OperationLedgerRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "operation",
            [resource_kind.value, resource_id],
        )
        values = await self._store.read(
            lambda transaction: transaction.list_operations(
                OperationQuery(stream_digest=stream, states=frozenset({"PENDING", "RUNNING"}), limit=limit)
            )
        )
        return tuple(_decode_operation(value) for value in values)

    async def compact_terminal(
        self,
        resource_kind: ResourceKind,
        resource_id: str,
        *,
        tenant_id: str,
        through_sequence: int,
    ) -> str:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "operation",
            [resource_kind.value, resource_id],
        )
        values = await self._store.mutate(
            lambda transaction: transaction.delete_operations(
                OperationQuery(
                    stream_digest=stream,
                    states=frozenset({"SUCCEEDED", "FAILED", "CANCELLED"}),
                    through_sequence=through_sequence,
                    compactable=True,
                )
            )
        )
        return hashlib.sha256(
            canonical_json_bytes(
                [
                    {
                        "key": value.key_digest.hex(),
                        "stream": value.stream_digest.hex(),
                        "sequence": value.sequence,
                        "data": value.data,
                    }
                    for value in values
                ]
            )
        ).hexdigest()


# The implementations between OperationLedgerRepository and TaskRepositoryImpl are
# unchanged from the previous commit.


class TaskRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.TASK)

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _node_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node", [graph_id, node_id])

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
        return (
            None
            if await self.get_graph(graph_id, tenant_id=tenant_id) is None
            else ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)
        )

    async def create_graph(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        view = TaskGraphView(graph.graph_id, TaskStatus.PENDING, graph.nodes)

        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            records = [self._stored("task_graph", graph.graph_id, view, state=view.status.value)]
            for node in graph.nodes:
                status = TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
                node_view = TaskNodeView(
                    graph.graph_id, node.node_id, node.dependencies, status, None, 0, None, None, None, None
                )
                records.append(
                    self._stored(
                        "task_node",
                        [graph.graph_id, node.node_id],
                        node_view,
                        parent=self._parent("task_node", "graph", graph.graph_id),
                        state=status.value,
                    )
                )
            await transaction.insert_records(records)
            return view

        return await self._store.mutate(mutate)

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._graph_key(graph_id))
        if record is None:
            return None
        graph = await self._decode(record, TaskGraphView)
        nodes = await self.list_nodes(graph_id, tenant_id=tenant_id)
        return TaskGraphView(graph_id, _graph_status(nodes), graph.nodes)

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        _require_repository_tenant(tenant_id, self._tenant_id)
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph = await self._decode(graph_record, TaskGraphView)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            decoded_nodes = await self._decode_many(node_records)
            node_records_by_id = {
                node.node_id: record
                for node, record in zip(decoded_nodes, node_records, strict=True)
            }
            current_nodes = {node.node_id: node for node in decoded_nodes}
            next_nodes = dict(current_nodes)
            changed_nodes: list[tuple[TaskNodeView, TaskNodeView]] = []
            changed = True
            while changed:
                changed = False
                for node in tuple(next_nodes.values()):
                    if node.status in {
                        TaskStatus.SUCCEEDED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.BLOCKED,
                    }:
                        continue
                    dependencies = tuple(next_nodes[dependency] for dependency in node.dependencies)
                    if any(
                        dependency.status in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}
                        for dependency in dependencies
                    ):
                        value = replace(
                            node,
                            status=TaskStatus.BLOCKED,
                            error_code=ErrorCode.TASK_DEPENDENCY_FAILED.value,
                            error_digest=None,
                        )
                    elif node.status is TaskStatus.PENDING and all(
                        dependency.status is TaskStatus.SUCCEEDED for dependency in dependencies
                    ):
                        value = replace(node, status=TaskStatus.READY)
                    else:
                        continue
                    next_nodes[node.node_id] = value
                    changed_nodes.append((node, value))
                    changed = True
            next_status = _graph_status(tuple(next_nodes.values()))
            graph_changed = graph.status is not next_status or graph_record.state != next_status.value
            if not changed_nodes and not graph_changed:
                return TaskGraphView(graph.graph_id, next_status, graph.nodes)
            next_graph = replace(graph, status=next_status)
            replacements: list[RecordReplacement] = []
            for current, value in changed_nodes:
                node_record = node_records_by_id.get(current.node_id)
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stored_node = await self._decode(node_record, TaskNodeView)
                if stored_node != current:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                replacements.append(
                    RecordReplacement(
                        _projected_record(self, node_record, value),
                        node_record.storage_version,
                    )
                )
            if graph_changed:
                replacements.append(
                    RecordReplacement(
                        _task_graph_record(self, graph_record, next_graph),
                        graph_record.storage_version,
                    )
                )
            await transaction.replace_records(replacements)
            _logger.info(
                "task graph reconciled atomically: graph_id=%s changed_nodes=%s",
                graph_id,
                len(changed_nodes),
            )
            return TaskGraphView(graph.graph_id, next_graph.status, graph.nodes)

        return await self._store.mutate(mutate)

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        _require_repository_tenant(tenant_id, self._tenant_id)
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            graph_record = await transaction.get_record(self._graph_key(graph_id))
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph = await self._decode(graph_record, TaskGraphView)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            nodes = await self._decode_many(node_records)
            node_records_by_id = {
                node.node_id: record
                for node, record in zip(nodes, node_records, strict=True)
            }
            changed = [
                (node, replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None))
                for node in nodes
                if node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            ]
            changed_ids = {current.node_id for current, _ in changed}
            next_nodes = tuple(value for _, value in changed) + tuple(
                node for node in nodes if node.node_id not in changed_ids
            )
            next_status = _graph_status(next_nodes)
            graph_changed = (
                graph.status is not next_status
                or graph_record.state != next_status.value
            )
            if not changed and not graph_changed:
                return TaskGraphView(graph.graph_id, next_status, graph.nodes)
            next_graph = replace(graph, status=next_status)
            replacements: list[RecordReplacement] = []
            for current, value in changed:
                node_record = node_records_by_id.get(current.node_id)
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                replacements.append(
                    RecordReplacement(
                        _projected_record(self, node_record, value),
                        node_record.storage_version,
                    )
                )
            if graph_changed:
                replacements.append(
                    RecordReplacement(
                        _task_graph_record(self, graph_record, next_graph),
                        graph_record.storage_version,
                    )
                )
            await transaction.replace_records(replacements)
            _logger.info("task graph cancelled atomically: graph_id=%s changed_nodes=%s", graph_id, len(changed))
            return TaskGraphView(graph.graph_id, next_graph.status, graph.nodes)

        return await self._store.mutate(mutate)

    async def claim(self, graph_id: str, node_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        validate_lease_seconds(lease_seconds)

        async def mutate(transaction: StateTransaction) -> TaskLease:
            records = await transaction.get_records(
                (self._graph_key(graph_id), self._node_key(graph_id, node_id))
            )
            graph_record = records.get(self._graph_key(graph_id))
            record = records.get(self._node_key(graph_id, node_id))
            if graph_record is None or record is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            graph = await self._decode(graph_record, TaskGraphView)
            node = await self._decode(record, TaskNodeView)
            dependency_keys = tuple(
                self._node_key(graph_id, dependency)
                for dependency in node.dependencies
            )
            dependency_records = await transaction.get_records(dependency_keys)
            nodes = {
                dependency: await self._decode(
                    dependency_records[self._node_key(graph_id, dependency)],
                    TaskNodeView,
                )
                for dependency in node.dependencies
                if self._node_key(graph_id, dependency) in dependency_records
            }
            if graph.graph_id != graph_id or any(
                dependency not in nodes for dependency in node.dependencies
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            now = await transaction.now()
            expired = node.lease_expires_at is not None and node.lease_expires_at <= now
            if node.status is TaskStatus.RUNNING and node.owner not in {None, owner} and not expired:
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
            dependencies_succeeded = all(
                nodes[dependency].status is TaskStatus.SUCCEEDED
                for dependency in node.dependencies
            )
            if node.status not in {TaskStatus.PENDING, TaskStatus.READY} and not (
                node.status is TaskStatus.RUNNING and expired
            ):
                raise AIError(ErrorCode.TASK_NOT_READY)
            if node.status in {TaskStatus.PENDING, TaskStatus.READY} and not dependencies_succeeded:
                raise AIError(ErrorCode.TASK_NOT_READY)
            fence = node.fence + 1
            expires = now + timedelta(seconds=lease_seconds)
            value = replace(node, status=TaskStatus.RUNNING, owner=owner, fence=fence, lease_expires_at=expires)
            await self._update_node_in_transaction(transaction, node, value)
            return TaskLease(graph_id, node_id, tenant_id, owner, fence, expires)

        return await self._store.mutate(mutate)

    async def renew(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_seconds(lease_seconds)

        async def mutate(transaction: StateTransaction) -> TaskLease:
            records = await transaction.get_records(
                (self._graph_key(lease.graph_id), self._node_key(lease.graph_id, lease.node_id))
            )
            graph_record = records.get(self._graph_key(lease.graph_id))
            node_record = records.get(self._node_key(lease.graph_id, lease.node_id))
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            node = await self._decode(node_record, TaskNodeView)
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            expires = now + timedelta(seconds=lease_seconds)
            if not await transaction.update_record_lease(
                node_record.key_digest,
                expected_storage_version=node_record.storage_version,
                lease_owner=node.owner,
                lease_fence=node.fence,
                lease_expires_at=expires,
            ):
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            return replace(lease, lease_expires_at=expires)

        return await self._store.mutate(mutate)

    async def complete(
        self, lease: TaskLease, *, tenant_id: str, execution_id: str | None, result_digest: str
    ) -> TaskTerminalRecord:
        return await self._finish(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.SUCCEEDED,
            execution_id=execution_id,
            result_digest=result_digest,
            error_code=None,
            error_digest=None,
        )

    async def fail(self, lease: TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> TaskTerminalRecord:
        return await self._finish(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.FAILED,
            execution_id=None,
            result_digest=None,
            error_code=error_code,
            error_digest=error_digest,
        )

    async def _finish(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        status: TaskStatus,
        execution_id: str | None,
        result_digest: str | None,
        error_code: str | None,
        error_digest: str | None,
    ) -> TaskTerminalRecord:
        _validate_task_lease_scope(lease, tenant_id)
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskTerminalRecord:
            record = await transaction.get_record(self._node_key(lease.graph_id, lease.node_id))
            if record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            node = await self._decode(record, TaskNodeView)
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            value = replace(
                node,
                status=status,
                owner=None,
                lease_expires_at=None,
                result_digest=result_digest,
                error_code=error_code,
                error_digest=error_digest,
                execution_id=execution_id,
            )
            await self._update_node_in_transaction(transaction, node, value)
            return TaskTerminalRecord(
                lease.node_id,
                lease.owner,
                lease.fence,
                status,
                result_digest,
                error_code,
                error_digest,
                execution_id=execution_id,
            )

        return await self._store.mutate(mutate)

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[TaskNodeView, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records("task_node", parent=self._parent("task_node", "graph", graph_id))
        return await self._decode_many(records)

    async def _decode_many(self, records: tuple[StoredRecord, ...]) -> tuple[TaskNodeView, ...]:
        return tuple([await self._decode(record, TaskNodeView) for record in records])

    async def _node(self, graph_id: str, node_id: str, tenant_id: str) -> TaskNodeView:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await self._record(self._node_key(graph_id, node_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return await self._decode(record, TaskNodeView)

    async def _update_node(self, current: TaskNodeView, value: TaskNodeView) -> None:
        async def mutate(transaction: StateTransaction) -> None:
            await self._update_node_in_transaction(transaction, current, value)

        await self._store.mutate(mutate)

    async def _update_node_in_transaction(
        self,
        transaction: StateTransaction,
        current: TaskNodeView,
        value: TaskNodeView,
    ) -> None:
        graph_record = await transaction.get_record(self._graph_key(current.graph_id))
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        graph = await self._decode(graph_record, TaskGraphView)
        if graph.graph_id != current.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        node_record = await transaction.get_record(
            self._node_key(current.graph_id, current.node_id)
        )
        if node_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        stored_node = await self._decode(node_record, TaskNodeView)
        if stored_node != current:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        node_candidate = _projected_record(self, node_record, value)
        await _replace_checked(transaction, node_candidate, node_record.storage_version)


class TaskAdmissionRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.TASK)
        self._background_tasks: set[asyncio.Task[object]] = set()

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_graph", "recoverable", "graphs")

    async def admit(self, admission: TaskGraphAdmission, graph: TaskGraph) -> TaskGraphView:
        launch = admission.bind(graph)
        if launch.principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def operation() -> TaskGraphView:
            return await self._store.mutate(
                lambda transaction: self._admit_in_transaction(transaction, admission, graph)
            )

        async def readback() -> CommitObservation[TaskGraphView]:
            return await self._store.read(
                lambda transaction: self._read_admission(transaction, admission, graph)
            )

        result = await run_durable_commit(operation, readback, background_tasks=self._background_tasks)
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            if (
                isinstance(result.error, AIError)
                and result.error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
            ):
                raise result.error
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            if isinstance(result.error, AIError):
                raise result.error
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.cancelled:
            raise asyncio.CancelledError
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def list_recoverable_page(
        self, *, cursor: str | None, limit: int
    ) -> Page[TaskGraphLaunch]:
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._recovery_scope(),
                    kind="task_graph",
                    states=frozenset({TaskStatus.PENDING.value, TaskStatus.READY.value, TaskStatus.RUNNING.value}),
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            graph_views = tuple(
                [await self._decode(record, TaskGraphView) for record in selected]
            )
            admission_records = await transaction.get_records(
                tuple(
                    self._admission_key(graph_view.graph_id)
                    for graph_view in graph_views
                )
            )
            launches: list[TaskGraphLaunch] = []
            for record, graph_view in zip(selected, graph_views, strict=True):
                admission_record = admission_records.get(
                    self._admission_key(graph_view.graph_id)
                )
                if admission_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                admission = await self._decode(admission_record, TaskGraphAdmission)
                if (
                    record.scope_digest != self._recovery_scope()
                    or record.key_digest != self._graph_key(graph_view.graph_id)
                    or record.state != graph_view.status.value
                    or admission.graph_id != graph_view.graph_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(
                    admission.bind(TaskGraph(graph_view.graph_id, graph_view.nodes))
                )
            next_cursor = _record_cursor(selected[-1]) if len(records) > limit and selected else None
            return Page(tuple(launches), next_cursor)

        return await self._store.read(read)

    async def _admit_in_transaction(
        self, transaction: StateTransaction, admission: TaskGraphAdmission, graph: TaskGraph
    ) -> TaskGraphView:
        operation_key_value = operation_key(
            self._namespace, self._tenant_id, self._domain.value, admission.operation_id
        )
        records = await transaction.get_records(
            (self._graph_key(graph.graph_id), self._admission_key(graph.graph_id))
        )
        graph_record = records.get(self._graph_key(graph.graph_id))
        admission_record = records.get(self._admission_key(graph.graph_id))
        stored_operation = await transaction.get_operation(operation_key_value)
        node_records = await transaction.list_records(
            RecordQuery(parent_digest=self._parent("task_node", "graph", graph.graph_id), kind="task_node")
        )
        if graph_record is None and admission_record is None and stored_operation is None:
            if node_records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await self._create_admission(transaction, admission, graph)
        if stored_operation is None:
            await self._raise_existing_graph_conflict(
                transaction,
                graph_record,
                admission_record,
                node_records,
            )
        if stored_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation = _decode_operation(stored_operation)
        if operation.request_digest != admission.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        if operation.status in {OperationStatus.FAILED, OperationStatus.CANCELLED}:
            if admission_record is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raise _stored_operation_error(operation)
        if graph_record is None:
            if admission_record is not None or node_records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if operation.status in {
                OperationStatus.PENDING,
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return await self._create_admission(
                    transaction,
                    admission,
                    graph,
                    stored_operation=stored_operation,
                    operation=operation,
                )
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        if admission_record is not None:
            existing = await self._decode(admission_record, TaskGraphAdmission)
            stored_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
            existing.bind(stored_graph)
            if existing.operation_id != admission.operation_id:
                await self._raise_existing_graph_conflict(
                    transaction,
                    graph_record,
                    admission_record,
                    node_records,
                )
            self._validate_graph(graph_view, graph, nodes)
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_committed_operation(operation, graph)
            if graph_record.scope_digest != self._recovery_scope():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskGraphView(graph.graph_id, _graph_status(nodes), graph.nodes)
        self._validate_graph(graph_view, graph, nodes)
        if operation.status not in {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
            OperationStatus.SUCCEEDED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        status = _graph_status(nodes)
        next_graph = TaskGraphView(graph.graph_id, status, graph.nodes)
        await _replace_checked(
            transaction,
            replace(
                self._stored(
                    "task_graph", graph.graph_id, next_graph,
                    scope=self._recovery_scope(), state=status.value,
                ),
                storage_version=graph_record.storage_version + 1,
            ),
            graph_record.storage_version,
        )
        await transaction.insert_record(self._stored("task_admission", graph.graph_id, admission))
        await self._settle_operation(transaction, stored_operation, operation, graph)
        _logger.info("legacy task graph admission upgraded: tenant=%s graph=%s", self._tenant_id, graph.graph_id)
        return next_graph

    async def _create_admission(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
        *,
        stored_operation: StoredOperation | None = None,
        operation: OperationLedgerRecord | None = None,
    ) -> TaskGraphView:
        if (stored_operation is None) != (operation is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
        view = TaskGraphView(graph.graph_id, status, graph.nodes)
        records = [
            self._stored(
                "task_graph", graph.graph_id, view,
                scope=self._recovery_scope(), state=status.value,
            ),
            self._stored("task_admission", graph.graph_id, admission),
        ]
        for node in graph.nodes:
            node_status = TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
            node_view = TaskNodeView(
                graph.graph_id, node.node_id, node.dependencies, node_status, None, 0, None, None, None, None
            )
            records.append(
                self._stored(
                    "task_node", [graph.graph_id, node.node_id], node_view,
                    parent=self._parent("task_node", "graph", graph.graph_id),
                    state=node_status.value,
                )
            )
        await transaction.insert_records(records)
        if stored_operation is None:
            now = await transaction.now()
            operation_input = OperationLedgerInput(
                admission.operation_id, self._tenant_id, ResourceKind.TASK_GRAPH, graph.graph_id, None,
                OperationKind.TASK_NODE, OperationStatus.SUCCEEDED, admission.request_digest,
                graph.graph_id, _task_submit_result_digest(graph), None, True, now, now,
            )
            await _append_operation(transaction, self, operation_input)
            _logger.info("task graph durably admitted: tenant=%s graph=%s", self._tenant_id, graph.graph_id)
        else:
            if operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._settle_operation(
                transaction,
                stored_operation,
                operation,
                graph,
            )
            _logger.info(
                "legacy task graph admission completed: tenant=%s graph=%s",
                self._tenant_id,
                graph.graph_id,
            )
        return view

    async def _settle_operation(
        self,
        transaction: StateTransaction,
        stored: StoredOperation,
        operation: OperationLedgerRecord,
        graph: TaskGraph,
    ) -> None:
        now = await transaction.now()
        next_operation = replace(
            operation, status=OperationStatus.SUCCEEDED, result_ref=graph.graph_id,
            result_digest=_task_submit_result_digest(graph), error_code=None, updated_at=now,
        )
        candidate = _stored_from_operation(next_operation, stored)
        if not await transaction.replace_operation(candidate, expected_state=operation.status.value):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _read_admission(
        self, transaction: StateTransaction, admission: TaskGraphAdmission, graph: TaskGraph
    ) -> CommitObservation[TaskGraphView]:
        operation_key_value = operation_key(
            self._namespace, self._tenant_id, self._domain.value, admission.operation_id
        )
        records = await transaction.get_records(
            (self._graph_key(graph.graph_id), self._admission_key(graph.graph_id))
        )
        graph_record = records.get(self._graph_key(graph.graph_id))
        admission_record = records.get(self._admission_key(graph.graph_id))
        stored_operation = await transaction.get_operation(operation_key_value)
        node_records = await transaction.list_records(
            RecordQuery(parent_digest=self._parent("task_node", "graph", graph.graph_id), kind="task_node")
        )
        if graph_record is None and admission_record is None and stored_operation is None and not node_records:
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        operation: OperationLedgerRecord | None = None
        if stored_operation is not None:
            operation = _decode_operation(stored_operation)
            if operation.request_digest != admission.request_digest:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                )
            try:
                self._validate_operation_identity(operation, admission)
            except AIError as error:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=error,
                )
            if operation.status in {OperationStatus.FAILED, OperationStatus.CANCELLED}:
                if admission_record is not None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=_stored_operation_error(operation),
                )
        if graph_record is None:
            if admission_record is not None or node_records:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            if operation is not None and operation.status in {
                OperationStatus.PENDING,
                OperationStatus.RUNNING,
                OperationStatus.EFFECT_UNKNOWN,
            }:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        if stored_operation is None:
            try:
                await self._raise_existing_graph_conflict(
                    transaction,
                    graph_record,
                    admission_record,
                    node_records,
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_CONFLICT:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=error,
                    )
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=error,
                )
        if stored_operation is None or operation is None:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        try:
            graph_view = await self._decode(graph_record, TaskGraphView)
            nodes = await self._decode_task_nodes(node_records)
            if admission_record is None:
                self._validate_graph(graph_view, graph, nodes)
                if operation.status in {
                    OperationStatus.PENDING,
                    OperationStatus.RUNNING,
                    OperationStatus.EFFECT_UNKNOWN,
                    OperationStatus.SUCCEEDED,
                }:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing = await self._decode(admission_record, TaskGraphAdmission)
            stored_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
            existing.bind(stored_graph)
            if existing.operation_id != admission.operation_id:
                try:
                    await self._raise_existing_graph_conflict(
                        transaction,
                        graph_record,
                        admission_record,
                        node_records,
                    )
                except AIError as error:
                    if error.code is ErrorCode.STORAGE_CONFLICT:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED,
                            error=error,
                        )
                    raise
            self._validate_operation_identity(operation, admission)
            self._validate_graph(graph_view, graph, nodes)
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_committed_operation(operation, graph)
            if (
                graph_record.scope_digest != self._recovery_scope()
                or graph_record.state != graph_view.status.value
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return CommitObservation(
                DurableCommitState.COMMITTED,
                TaskGraphView(graph.graph_id, _graph_status(nodes), graph.nodes),
            )
        except (KeyError, TypeError, ValueError) as error:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        except AIError as error:
            return CommitObservation(DurableCommitState.PARTIAL_INTEGRITY_ERROR, error=error)

    async def _raise_existing_graph_conflict(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord | None,
        admission_record: StoredRecord | None,
        node_records: tuple[StoredRecord, ...],
    ) -> None:
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        stored_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
        self._validate_graph(graph_view, stored_graph, nodes)
        if admission_record is not None:
            existing = await self._decode(admission_record, TaskGraphAdmission)
            existing.bind(stored_graph)
            existing_operation_key = operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                existing.operation_id,
            )
            stored_existing_operation = await transaction.get_operation(
                existing_operation_key
            )
            if stored_existing_operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing_operation = _decode_operation(stored_existing_operation)
            self._validate_operation_identity(existing_operation, existing)
            self._validate_committed_operation(existing_operation, stored_graph)
            if graph_record.scope_digest != self._recovery_scope():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _validate_committed_operation(
        self,
        operation: OperationLedgerRecord,
        graph: TaskGraph,
    ) -> None:
        if (
            operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != graph.graph_id
            or operation.result_digest != _task_submit_result_digest(graph)
            or operation.error_code is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_operation_identity(
        self, operation: OperationLedgerRecord, admission: TaskGraphAdmission
    ) -> None:
        if (
            operation.operation_id != admission.operation_id
            or operation.tenant_id != self._tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != admission.graph_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.TASK_NODE
            or not operation.compactable
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_graph(
        self, stored: TaskGraphView, graph: TaskGraph, nodes: tuple[TaskNodeView, ...]
    ) -> None:
        if stored.graph_id != graph.graph_id or stored.nodes != graph.nodes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected = {node.node_id: node.dependencies for node in graph.nodes}
        actual = {node.node_id: node.dependencies for node in nodes}
        if actual != expected or len(nodes) != len(expected):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _decode_task_nodes(
        self, records: tuple[StoredRecord, ...]
    ) -> tuple[TaskNodeView, ...]:
        return tuple([await self._decode(record, TaskNodeView) for record in records])


# The remaining repository implementations and helpers are unchanged from the
# previous commit.
