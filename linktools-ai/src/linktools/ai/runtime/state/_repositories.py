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


class ConversationHistoryRepositoryImpl(_RepositoryBase):
    """Persist immutable branch descriptors and their skew prefix index."""

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.CONVERSATION,
        )

    async def create(self, record: ConversationHistoryRecord) -> ConversationHistoryRecord:
        _require_tenant(record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ConversationHistoryRecord:
            return await self.create_in_transaction(transaction, record)

        return await self._store.mutate(mutate)

    async def create_in_transaction(
        self,
        transaction: StateTransaction,
        record: ConversationHistoryRecord,
    ) -> ConversationHistoryRecord:
        _require_tenant(record, self._tenant_id)
        history_key = self._key("conversation_history", record.history_id)
        head_key = self._key("transcript_head", record.history_id)
        records = await transaction.get_records((history_key, head_key))
        current = records.get(history_key)
        head = records.get(head_key)
        if current is not None:
            existing = await self._decode_history(current)
            if existing != record:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if head is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        if head is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_records(
            (
                self._stored("conversation_history", record.history_id, record),
                self._stored(
                    "transcript_head",
                    record.history_id,
                    TranscriptHeadRecord(
                        TranscriptOwnerDomain.CONVERSATION,
                        record.history_id,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        HistoryQuality.COMPLETE,
                        0,
                    ),
                ),
            )
        )
        _logger.debug(
            "conversation history admitted with head: history=%s",
            record.history_id,
        )
        return record

    async def get(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryRecord | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key("conversation_history", history_id))
        return None if record is None else await self._decode_history(record)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        history_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("conversation_history", history_id))
        return None if record is None else await self._decode_history(record)

    async def local_head_in_transaction(
        self,
        transaction: StateTransaction,
        history_id: str,
    ) -> tuple[int, int]:
        """Read one branch's local message and history-item counts."""
        record = await transaction.get_record(
            self._key("transcript_head", history_id)
        )
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await self._decode(record, TranscriptHeadRecord)
        return head.message_count, head.session_history_item_count

    async def get_index_node_in_transaction(
        self,
        transaction: StateTransaction,
        node_id: str,
    ) -> ConversationHistoryIndexNodeRecord:
        """Read exactly one skew index node without traversing its children."""
        record = await transaction.get_record(
            self._key("conversation_index_node", node_id)
        )
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        node = _decode_enveloped_domain(
            record.data,
            ConversationHistoryIndexNodeRecord,
        )
        if node.node_id != node_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return node

    async def get_index_node(
        self,
        node_id: str,
        *,
        tenant_id: str,
    ) -> ConversationHistoryIndexNodeRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self.get_index_node_in_transaction(
                transaction,
                node_id,
            )
        )

    async def get_forest_roots_in_transaction(
        self,
        transaction: StateTransaction,
        head_id: str | None,
        *,
        max_roots: int,
    ) -> tuple[ConversationHistoryIndexNodeRecord, ...]:
        if max_roots < 1:
            raise ValueError("max_roots must be positive")
        roots: list[ConversationHistoryIndexNodeRecord] = []
        cursor = head_id
        visited: set[str] = set()
        while cursor is not None and len(roots) < max_roots:
            if cursor in visited:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(cursor)
            node = await self.get_index_node_in_transaction(transaction, cursor)
            roots.append(node)
            cursor = node.next_forest_id
        if cursor is not None and max_roots > 2:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(roots)

    async def get_forest_roots(
        self,
        head_id: str | None,
        *,
        tenant_id: str,
        max_roots: int,
    ) -> tuple[ConversationHistoryIndexNodeRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self.get_forest_roots_in_transaction(
                transaction,
                head_id,
                max_roots=max_roots,
            )
        )

    async def insert_index_node_in_transaction(
        self,
        transaction: StateTransaction,
        node: ConversationHistoryIndexNodeRecord,
    ) -> None:
        key = self._key("conversation_index_node", node.node_id)
        if await transaction.get_record(key) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_record(
            self._stored("conversation_index_node", node.node_id, node)
        )

    async def fork(
        self,
        source_history_id: str,
        child_history_id: str,
        *,
        session_id: str,
        tenant_id: str,
    ) -> ConversationHistoryRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ConversationHistoryRecord:
            source = await self.get_in_transaction(
                transaction,
                source_history_id,
                tenant_id=tenant_id,
            )
            if source is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            local_messages, local_items = await self.local_head_in_transaction(
                transaction,
                source_history_id,
            )
            prefix_head = source.prefix_index_head_id
            if local_messages > 0 or local_items > 0:
                roots = await self.get_forest_roots_in_transaction(
                    transaction,
                    prefix_head,
                    max_roots=2,
                )
                node = build_fork_index_node_from_roots(
                    roots,
                    source_history_id=source_history_id,
                    source_local_message_count=local_messages,
                    source_local_history_item_count=local_items,
                )
                if node is None:
                    pass
                elif isinstance(node, str):
                    prefix_head = node
                else:
                    await self.insert_index_node_in_transaction(transaction, node)
                    prefix_head = node.node_id
            inherited_messages = source.inherited_message_count + local_messages
            inherited_items = source.inherited_history_item_count + local_items
            child = ConversationHistoryRecord(
                history_id=child_history_id,
                session_id=session_id,
                tenant_id=tenant_id,
                parent_history_id=source_history_id,
                prefix_index_head_id=prefix_head,
                inherited_message_count=inherited_messages,
                inherited_history_item_count=inherited_items,
            )
            key = self._key("conversation_history", child_history_id)
            current = await transaction.get_record(key)
            if current is None:
                return await self.create_in_transaction(transaction, child)
            existing = await self._decode_history(current)
            if existing == child:
                return existing
            if (
                existing.session_id == session_id
                and existing.parent_history_id is None
                and existing.inherited_message_count == 0
                and existing.inherited_history_item_count == 0
            ):
                await _replace_checked(
                    transaction,
                    replace(
                        self._stored(
                            "conversation_history",
                            child_history_id,
                            child,
                        ),
                        storage_version=current.storage_version + 1,
                    ),
                    current.storage_version,
                )
                return child
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)

        return await self._store.mutate(mutate)

    async def _decode_history(self, record: StoredRecord) -> ConversationHistoryRecord:
        return _decode_enveloped_domain(record.data, ConversationHistoryRecord)



class SessionRepositoryImpl(_ResourceRepository[SessionRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.CONVERSATION,
            kind="session",
            resource_kind=ResourceKind.SESSION,
            value_type=SessionRecord,
            identity_field="session_id",
        )

    def _list_generation_key(self, owner_principal_id: str | None = None) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "session_list_owner" if owner_principal_id is not None else "session_list_tenant",
            owner_principal_id if owner_principal_id is not None else [],
        )

    async def _bump_list_generation(
        self,
        transaction: StateTransaction,
        owner_principal_id: str,
    ) -> None:
        await transaction.reserve_sequences(
            {
                self._list_generation_key(owner_principal_id): 1,
                self._list_generation_key(): 1,
            }
        )

    async def create(self, value: SessionRecord) -> SessionRecord:
        _require_tenant(value, self._tenant_id)
        _require_explicit_session_agent_id(value)
        value = _ensure_session_history(value)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        value.session_id,
                        value,
                        scope=self._scope("session", "owner", value.owner_principal_id),
                        state=value.status.value,
                    ),
                    self._stored(
                        "conversation_history",
                        value.history_id,
                        _new_session_history(value),
                    ),
                    self._stored(
                        "transcript_head",
                        value.history_id,
                        _empty_conversation_transcript_head(value.history_id),
                    ),
                )
            )
            await self._bump_list_generation(transaction, value.owner_principal_id)
            _logger.debug(
                "session admitted with history: session=%s history=%s",
                value.session_id,
                value.history_id,
            )
            return value

        return await self._store.mutate(mutate)

    async def create_with_operation(
        self, record: SessionRecord, *, operation: OperationLedgerInput
    ) -> tuple[SessionRecord, bool]:
        _require_tenant(record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)
        _require_explicit_session_agent_id(record)
        record = _ensure_session_history(record)

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            _, replayed = await _append_operation(transaction, self, operation)
            if replayed:
                current = await transaction.get_record(self._key("session", record.session_id))
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self._decode(current, SessionRecord), True
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        record.session_id,
                        record,
                        scope=self._scope("session", "owner", record.owner_principal_id),
                        state=record.status.value,
                    ),
                    self._stored(
                        "conversation_history",
                        record.history_id,
                        _new_session_history(record),
                    ),
                    self._stored(
                        "transcript_head",
                        record.history_id,
                        _empty_conversation_transcript_head(record.history_id),
                    ),
                )
            )
            await self._bump_list_generation(transaction, record.owner_principal_id)
            _logger.debug(
                "session operation admitted with history: session=%s history=%s",
                record.session_id,
                record.history_id,
            )
            return record, False

        return await self._store.mutate(mutate)

    async def create_fork_with_operation(
        self,
        source_session_id: str,
        target: SessionRecord,
        *,
        expected_source_revision: int,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]:
        _require_tenant(target, self._tenant_id)
        _require_tenant(operation, self._tenant_id)
        _require_explicit_session_agent_id(target)
        if (
            operation.resource_kind is not ResourceKind.SESSION
            or operation.resource_id != target.session_id
            or operation.operation_kind is not OperationKind.SESSION_FORK
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if expected_source_revision < 0:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        target = _ensure_session_history(replace(target, history_id=None))

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            operation_record, replayed = await _append_operation(
                transaction,
                self,
                operation,
            )
            if replayed:
                return await self._replay_fork_in_transaction(
                    transaction,
                    source_session_id=source_session_id,
                    target=target,
                    operation=operation,
                    operation_record=operation_record,
                )
            source_stored = await transaction.get_record(
                self._key("session", source_session_id)
            )
            if source_stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            source = await self._decode(source_stored, SessionRecord)
            if source.tenant_id != self._tenant_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if source.revision != expected_source_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if source.history_id is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            if await transaction.guard_record(
                self._key("session", source_session_id),
                expected_storage_version=source_stored.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            source_history_stored = await transaction.get_record(
                self._key("conversation_history", source.history_id)
            )
            if source_history_stored is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            source_history = await self._decode_history(source_history_stored)
            if (
                source_history.session_id != source.session_id
                or source_history.tenant_id != self._tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_stored = await transaction.get_record(
                self._key("session", target.session_id)
            )
            child_history_id = target.history_id
            if child_history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            child_stored = await transaction.get_record(
                self._key("conversation_history", child_history_id)
            )
            histories = ConversationHistoryRepositoryImpl(
                self._store,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
            )
            local_messages, local_items = await histories.local_head_in_transaction(
                transaction,
                source.history_id,
            )
            prefix_head = source_history.prefix_index_head_id
            if local_messages > 0 or local_items > 0:
                roots = await histories.get_forest_roots_in_transaction(
                    transaction,
                    prefix_head,
                    max_roots=2,
                )
                node = build_fork_index_node_from_roots(
                    roots,
                    source_history_id=source.history_id,
                    source_local_message_count=local_messages,
                    source_local_history_item_count=local_items,
                )
                if isinstance(node, str):
                    prefix_head = node
                else:
                    await histories.insert_index_node_in_transaction(
                        transaction,
                        node,
                    )
                    prefix_head = node.node_id
            inherited = source_history.inherited_message_count + local_messages
            inherited_items = (
                source_history.inherited_history_item_count + local_items
            )
            child = ConversationHistoryRecord(
                history_id=child_history_id,
                session_id=target.session_id,
                tenant_id=self._tenant_id,
                parent_history_id=source.history_id,
                prefix_index_head_id=prefix_head,
                inherited_message_count=inherited,
                inherited_history_item_count=inherited_items,
            )
            expected_target = replace(
                target,
                history_id=child_history_id,
                history_quality="complete",
                continuation=(
                    None
                    if target.continuation is None
                    else replace(
                        target.continuation,
                        history_id=child_history_id,
                    )
                ),
            )
            if target_stored is not None or child_stored is not None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await transaction.insert_record(
                self._stored(
                    "session",
                    expected_target.session_id,
                    expected_target,
                    scope=self._scope(
                        "session",
                        "owner",
                        expected_target.owner_principal_id,
                    ),
                    state=expected_target.status.value,
                )
            )
            await transaction.insert_record(
                self._stored(
                    "conversation_history",
                    child.history_id,
                    child,
                )
            )
            await transaction.insert_record(
                self._stored(
                    "transcript_head",
                    child.history_id,
                    _empty_conversation_transcript_head(child.history_id),
                )
            )
            source_head_stored = await transaction.get_record(
                self._key("transcript_head", source.history_id)
            )
            if source_head_stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_head = _decode_enveloped_domain(
                source_head_stored.data,
                TranscriptHeadRecord,
            )
            fork_result = SessionForkResultRecord(
                operation.operation_id,
                source.session_id,
                source.history_id,
                source.revision,
                source_head.revision,
                local_messages,
                local_items,
                source_history.prefix_index_head_id,
                inherited,
                inherited_items,
                expected_target.session_id,
                child.history_id,
                child.prefix_index_head_id,
                operation.request_digest,
                operation.result_digest or "",
            )
            if not fork_result.result_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await transaction.insert_record(
                self._stored(
                    "session_fork_result",
                    operation.operation_id,
                    fork_result,
                )
            )
            await self._bump_list_generation(
                transaction,
                expected_target.owner_principal_id,
            )
            _logger.info(
                "session fork committed: source=%s target=%s inherited=%s",
                source_session_id,
                expected_target.session_id,
                inherited,
            )
            return expected_target, False

        return await self._store.mutate(mutate)

    async def _replay_fork_in_transaction(
        self,
        transaction: StateTransaction,
        *,
        source_session_id: str,
        target: SessionRecord,
        operation: OperationLedgerInput,
        operation_record: OperationLedgerRecord,
    ) -> tuple[SessionRecord, bool]:
        result_stored = await transaction.get_record(
            self._key("session_fork_result", operation.operation_id)
        )
        if result_stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = _decode_enveloped_domain(
            result_stored.data,
            SessionForkResultRecord,
        )
        if (
            result.operation_id != operation.operation_id
            or result.request_digest != operation.request_digest
            or result.result_digest != operation_record.result_digest
            or result.source_session_id != source_session_id
            or operation.result_digest != result.result_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        target_stored = await transaction.get_record(
            self._key("session", result.target_session_id)
        )
        child_stored = await transaction.get_record(
            self._key("conversation_history", result.target_history_id)
        )
        head_stored = await transaction.get_record(
            self._key("transcript_head", result.target_history_id)
        )
        if target_stored is None or child_stored is None or head_stored is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing_target = await self._decode(target_stored, SessionRecord)
        child = await self._decode_history(child_stored)
        if (
            existing_target.session_id != result.target_session_id
            or existing_target.history_id != result.target_history_id
            or existing_target.tenant_id != self._tenant_id
            or existing_target.owner_principal_id != target.owner_principal_id
            or existing_target.agent_id != target.agent_id
            or child.session_id != result.target_session_id
            or child.tenant_id != self._tenant_id
            or child.parent_history_id != result.source_history_id
            or child.prefix_index_head_id != result.target_prefix_index_head_id
            or child.inherited_message_count != result.inherited_message_count
            or child.inherited_history_item_count != result.inherited_history_item_count
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = _decode_enveloped_domain(
            head_stored.data,
            TranscriptHeadRecord,
        )
        if (
            head.owner_domain is not TranscriptOwnerDomain.CONVERSATION
            or head.owner_id != result.target_history_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "session fork replayed from first result: source=%s target=%s",
            source_session_id,
            existing_target.session_id,
        )
        return existing_target, True

    async def _visible_history_count_in_transaction(
        self,
        transaction: StateTransaction,
        record: ConversationHistoryRecord,
    ) -> int:
        histories = ConversationHistoryRepositoryImpl(
            self._store,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
        )
        local_messages, _items = await histories.local_head_in_transaction(
            transaction,
            record.history_id,
        )
        return record.inherited_message_count + local_messages

    async def list(self, *, tenant_id: str, owner_principal_id: str | None = None) -> tuple[SessionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        scope = None if owner_principal_id is None else self._scope("session", "owner", owner_principal_id)

        async def read(transaction: StateTransaction) -> tuple[SessionRecord, ...]:
            await transaction.get_sequence(self._list_generation_key(owner_principal_id))
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("session") if scope is None else None,
                    scope_digest=scope,
                    kind="session",
                )
            )
            return tuple([await self._decode(record, SessionRecord) for record in records])

        return await self._store.read(read)

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[SessionRecord]]:
        if tenant_id != self._tenant_id:
            return 0, Page(())
        scope = (
            None
            if owner_principal_id is None
            else self._scope(
                "session",
                "owner",
                owner_principal_id,
            )
        )

        async def read(transaction: StateTransaction) -> tuple[int, Page[SessionRecord]]:
            generation = await transaction.get_sequence(self._list_generation_key(owner_principal_id))
            if snapshot is not None and snapshot != generation:
                raise AIError(ErrorCode.CURSOR_INVALID)
            after_sort_key, after_key_digest = _decode_record_cursor(cursor)
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("session") if scope is None else None,
                    scope_digest=scope,
                    kind="session",
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            values = tuple([await self._decode(record, SessionRecord) for record in records[:limit]])
            next_cursor = _record_cursor(records[limit - 1]) if len(records) > limit else None
            return generation, Page(values, next_cursor)

        return await self._store.read(read)

    async def compare_and_swap_with_operation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            current = await transaction.get_record(self._key("session", session_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, SessionRecord)
            if value.revision != expected_revision:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            proposed = next_record
            if value.active_execution_id is not None and proposed.active_execution_id != value.active_execution_id:
                proposed = replace(proposed, active_execution_id=value.active_execution_id)
            _, replayed = await _append_operation(transaction, self, operation)
            if replayed:
                return value, True
            candidate = _projected_record(self, current, proposed)
            await _replace_checked(transaction, candidate, current.storage_version)
            await self._bump_list_generation(transaction, value.owner_principal_id)
            return proposed, False

        return await self._store.mutate(mutate)

    async def compare_and_swap(
        self, session_id: str, *, tenant_id: str, expected_revision: int, next_record: SessionRecord
    ) -> SessionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        _require_tenant(next_record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            record = await transaction.get_record(self._key("session", session_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, SessionRecord)
            if current.revision != expected_revision:
                raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            proposed = next_record
            if current.active_execution_id is not None and proposed.active_execution_id != current.active_execution_id:
                proposed = replace(proposed, active_execution_id=current.active_execution_id)
            await _replace_checked(
                transaction,
                _projected_record(self, record, proposed),
                record.storage_version,
            )
            await self._bump_list_generation(transaction, current.owner_principal_id)
            return proposed

        return await self._store.mutate(mutate)

    async def admit_execution(
        self, session_id: str, *, tenant_id: str, execution_id: str, expected: ConversationCursor | None
    ) -> SessionRecord:
        return await self._admission(
            session_id, tenant_id=tenant_id, execution_id=execution_id, expected=expected, release=False
        )

    async def admit_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            release=False,
            transaction=transaction,
        )

    async def release_execution(self, session_id: str, *, tenant_id: str, execution_id: str) -> SessionRecord:
        return await self._admission(
            session_id, tenant_id=tenant_id, execution_id=execution_id, expected=None, release=True
        )

    async def release_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=None,
            release=True,
            transaction=transaction,
        )

    async def _admission(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        release: bool,
        transaction: StateTransaction | None = None,
    ) -> SessionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            current = await transaction.get_record(self._key("session", session_id))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, SessionRecord)
            if release:
                if value.active_execution_id is None:
                    return value
                if value.active_execution_id != execution_id:
                    return value
                next_value = replace(value, active_execution_id=None)
            else:
                if value.active_execution_id == execution_id and value.continuation == expected:
                    return value
                if value.status is not SessionStatus.OPEN:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                if value.active_execution_id is not None or value.continuation != expected:
                    raise AIError(ErrorCode.SESSION_BUSY)
                next_value = replace(value, active_execution_id=execution_id)
            candidate = _projected_record(self, current, next_value)
            await _replace_checked(transaction, candidate, current.storage_version)
            return next_value

        if transaction is not None:
            return await mutate(transaction)
        try:
            return await self._store.mutate(mutate)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT and not release:
                latest = await self.get(session_id, tenant_id=tenant_id)
                if latest is not None and latest.active_execution_id not in {None, execution_id}:
                    raise AIError(ErrorCode.SESSION_BUSY) from error
            raise

    async def transition_status(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected: frozenset[SessionStatus],
        next_status: SessionStatus,
        closed_at: datetime | None = None,
        require_no_active: bool = False,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            record = await transaction.get_record(self._key("session", session_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, SessionRecord)
            if current.status not in expected:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            if require_no_active and current.active_execution_id is not None:
                raise AIError(ErrorCode.SESSION_ACTIVE_EXECUTIONS)
            now = await transaction.now()
            next_value = replace(
                current,
                status=next_status,
                closed_at=closed_at,
                revision=current.revision + 1,
                resource_generation=current.resource_generation + 1,
                updated_at=now,
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, next_value),
                record.storage_version,
            )
            await self._bump_list_generation(transaction, current.owner_principal_id)
            return next_value

        return await self._store.mutate(mutate)

    async def advance_continuation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        history_quality: str | None = None,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        return await self._store.mutate(
            lambda transaction: self.advance_continuation_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                expected=expected,
                next_cursor=next_cursor,
                history_quality=history_quality,
            )
        )

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        record = await transaction.get_record(self._key("session", session_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return await self._decode(record, SessionRecord)

    async def _decode_history(self, record: StoredRecord) -> ConversationHistoryRecord:
        return _decode_enveloped_domain(record.data, ConversationHistoryRecord)

    async def advance_continuation_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        release_execution: bool = False,
        history_quality: str | None = None,
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        record = await transaction.get_record(self._key("session", session_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, SessionRecord)
        if current.continuation == next_cursor:
            return current
        if (
            current.active_execution_id != execution_id
            or current.continuation != expected
            or current.status
            not in {
                SessionStatus.OPEN,
                SessionStatus.CLOSING,
                SessionStatus.CLEANUP_REQUIRED,
            }
        ):
            raise AIError(ErrorCode.SESSION_BUSY)
        now = await transaction.now()
        next_value = replace(
            current,
            continuation=next_cursor,
            history_quality=(
                current.history_quality
                if history_quality is None
                else history_quality
            ),
            active_execution_id=None if release_execution else current.active_execution_id,
            revision=current.revision + 1,
            resource_generation=current.resource_generation + 1,
            updated_at=now,
        )
        await _replace_checked(
            transaction,
            _projected_record(self, record, next_value),
            record.storage_version,
        )
        await self._bump_list_generation(transaction, current.owner_principal_id)
        return next_value

    async def complete_execution_in_transaction(
        self,
        transaction: StateTransaction,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        history_quality: str | None = None,
    ) -> SessionRecord:
        return await self.advance_continuation_in_transaction(
            transaction,
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            next_cursor=next_cursor,
            release_execution=True,
            history_quality=history_quality,
        )


class IdempotencyRepositoryImpl(_ResourceRepository[IdempotencyRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str, domain: RuntimeDomain) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=domain,
            kind="idempotency",
            resource_kind=ResourceKind.EXECUTION if domain is RuntimeDomain.EXECUTION else ResourceKind.EVALUATION,
            value_type=IdempotencyRecord,
            identity_field="idempotency_key_digest",
        )

    def _identity_key(self, scope: str, key: str) -> list[str]:
        return [scope, key]

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        _require_tenant(record, self._tenant_id)
        identity = self._identity_key(record.scope, record.idempotency_key_digest)
        try:
            await self._insert(self._stored("idempotency", identity, record, state=record.status.value))
            return record
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            existing = await super().get(identity, tenant_id=record.tenant_id)
            if existing is not None and _same_idempotency(existing, record):
                return existing
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def get(
        self,
        scope: str,
        idempotency_key_digest: str,
        *,
        tenant_id: str,
    ) -> IdempotencyRecord | None:  # type: ignore[override]
        return await super().get(self._identity_key(scope, idempotency_key_digest), tenant_id=tenant_id)

    async def list_by_resource(
        self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str
    ) -> tuple[IdempotencyRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            self._kind,
            scope=self._scope("idempotency", "resource", [resource_kind.value, resource_id]),
        )
        values = [await self._decode(record, self._value_type) for record in records]
        return tuple(values)

    async def compare_and_swap(
        self,
        scope: str,
        idempotency_key_digest: str,
        *,
        tenant_id: str,
        expected_status: IdempotencyStatus,
        next_record: IdempotencyRecord,
    ) -> IdempotencyRecord:  # type: ignore[override]
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        identity = self._identity_key(scope, idempotency_key_digest)

        async def mutate(transaction: StateTransaction) -> IdempotencyRecord:
            current_record = await transaction.get_record(self._key(self._kind, identity))
            if current_record is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(current_record, IdempotencyRecord)
            if current.status is not expected_status:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current_record, next_record),
                current_record.storage_version,
            )
            return next_record

        return await self._store.mutate(mutate)


class ExecutionRepositoryImpl(_ResourceRepository[ExecutionRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EXECUTION,
            kind="execution",
            resource_kind=ResourceKind.EXECUTION,
            value_type=ExecutionRecord,
            identity_field="execution_id",
        )
        self._idempotency = IdempotencyRepositoryImpl(
            store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.EXECUTION
        )

    async def create_with_history_head(self, execution: ExecutionRecord) -> ExecutionRecord:
        """Create a recovery execution and its OPEN history head atomically."""
        return await self._store.mutate(
            lambda transaction: self.create_with_history_head_in_transaction(
                transaction,
                execution,
            )
        )

    async def create_with_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        _require_tenant(execution, self._tenant_id)
        execution_key = self._key("execution", execution.execution_id)
        head_key = self._key("execution_history_head", execution.execution_id)
        records = await transaction.get_records((execution_key, head_key))
        current = records.get(execution_key)
        head_record = records.get(head_key)
        if current is not None:
            existing = await self._decode(current, ExecutionRecord)
            if existing != execution:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if head_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._decode_history_head_record(
                head_record,
                execution.execution_id,
            )
            return existing
        if head_record is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        head = ExecutionHistoryHeadRecord(
            execution.execution_id,
            execution.tenant_id,
            ExecutionHistoryState.OPEN,
            0,
            None,
        )
        await transaction.insert_records(
            (
                self._stored(
                    "execution",
                    execution.execution_id,
                    execution,
                    state=execution.status.value,
                ),
                self._stored_history_head(head),
            )
        )
        _logger.debug(
            "execution admitted with history head: execution=%s",
            execution.execution_id,
        )
        return execution

    async def list_by_session(
        self, session_id: str, *, tenant_id: str, statuses: frozenset[ExecutionStatus] | None = None
    ) -> tuple[ExecutionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "execution",
            scope=self._scope("execution", "session", session_id),
            states=None if statuses is None else frozenset(status.value for status in statuses),
        )
        return tuple([await self._decode(record, ExecutionRecord) for record in records])

    async def list_children(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "execution",
            parent=self._parent("execution", "execution", execution_id),
        )
        return tuple([await self._decode(record, ExecutionRecord) for record in records])

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", execution_id))
        return None if record is None else await self._decode(record, ExecutionRecord)

    async def get_start_idempotency(
        self,
        claim: ExecutionStartClaim,
    ) -> IdempotencyRecord | None:
        return await self._idempotency.get(
            claim.scope,
            claim.idempotency_key_digest,
            tenant_id=claim.tenant_id,
        )

    async def get_terminal_idempotency(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> IdempotencyRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        values = await self._idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution_id,
            tenant_id=tenant_id,
        )
        if len(values) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values[0] if values else None

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        _require_tenant(reservation.execution, self._tenant_id)
        _require_tenant(reservation.idempotency, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionStartReservationResult:
            identity = self._idempotency._identity_key(
                reservation.idempotency.scope, reservation.idempotency.idempotency_key_digest
            )
            id_key = self._idempotency._key("idempotency", identity)
            execution_key = self._key("execution", reservation.execution.execution_id)
            head_key = self._key(
                "execution_history_head",
                reservation.execution.execution_id,
            )
            records = await transaction.get_records(
                (id_key, execution_key, head_key)
            )
            existing_id = records.get(id_key)
            if existing_id is not None:
                existing_idempotency = await self._idempotency._decode(existing_id, IdempotencyRecord)
                if not _same_idempotency_identity(existing_idempotency, reservation.idempotency):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                winner_key = self._key("execution", existing_idempotency.resource_id)
                if winner_key == execution_key:
                    winner_records = records
                    winner_head_key = head_key
                else:
                    winner_head_key = self._key(
                        "execution_history_head",
                        existing_idempotency.resource_id,
                    )
                    winner_records = await transaction.get_records(
                        (winner_key, winner_head_key)
                    )
                winner_record = winner_records.get(winner_key)
                if winner_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_value = await self._decode(winner_record, ExecutionRecord)
                winner_head = winner_records.get(winner_head_key)
                if winner_head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._decode_history_head_record(
                    winner_head,
                    existing_value.execution_id,
                )
                _logger.debug(
                    "execution start reservation replayed: execution=%s",
                    existing_value.execution_id,
                )
                return ExecutionStartReservationResult(existing_value, existing_idempotency, False)
            if execution_key in records or head_key in records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head = ExecutionHistoryHeadRecord(
                reservation.execution.execution_id,
                reservation.execution.tenant_id,
                ExecutionHistoryState.OPEN,
                0,
                None,
            )
            await transaction.insert_records(
                (
                    self._stored(
                        "execution",
                        reservation.execution.execution_id,
                        reservation.execution,
                        state=reservation.execution.status.value,
                    ),
                    self._idempotency._stored(
                        "idempotency",
                        identity,
                        reservation.idempotency,
                        state=reservation.idempotency.status.value,
                    ),
                    self._stored_history_head(head),
                )
            )
            _logger.debug(
                "execution start reserved: execution=%s",
                reservation.execution.execution_id,
            )
            return ExecutionStartReservationResult(reservation.execution, reservation.idempotency, True)

        return await self._store.mutate(mutate)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        return await self._store.mutate(
            lambda transaction: self.claim_start_in_transaction(transaction, claim)
        )

    async def claim_start_in_transaction(
        self,
        transaction: StateTransaction,
        claim: ExecutionStartClaim,
    ) -> ExecutionRecord:
        _require_repository_tenant(claim.tenant_id, self._tenant_id)
        execution_key = self._key("execution", claim.execution_id)
        identity = self._idempotency._identity_key(
            claim.scope,
            claim.idempotency_key_digest,
        )
        idempotency_key = self._idempotency._key("idempotency", identity)
        stored = await transaction.get_records((execution_key, idempotency_key))
        execution_record = stored.get(execution_key)
        if execution_record is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        current = await self._decode(execution_record, ExecutionRecord)
        if (
            current.revision != claim.expected_revision
            or current.event_sequence != claim.expected_event_sequence
            or current.status is not ExecutionStatus.PENDING_START
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        idempotency_record = stored.get(idempotency_key)
        idempotency_created = idempotency_record is None
        if idempotency_record is None:
            idempotency = IdempotencyRecord(
                tenant_id=claim.tenant_id,
                runtime_domain=RuntimeDomain.EXECUTION,
                scope=claim.scope,
                idempotency_key_digest=claim.idempotency_key_digest,
                request_digest=claim.request_digest,
                resource_kind=ResourceKind.EXECUTION,
                resource_id=claim.execution_id,
                status=IdempotencyStatus.STARTED,
                result_digest=None,
                error_code=None,
                created_at=claim.started_at,
                updated_at=claim.started_at,
            )
            idempotency_record = self._idempotency._stored(
                "idempotency",
                identity,
                idempotency,
                state=idempotency.status.value,
            )
        if idempotency_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not idempotency_created:
            idempotency = await self._idempotency._decode(
                idempotency_record,
                IdempotencyRecord,
            )
            if (
                not _same_idempotency(
                    idempotency,
                    IdempotencyRecord(
                        claim.tenant_id,
                        RuntimeDomain.EXECUTION,
                        claim.scope,
                        claim.idempotency_key_digest,
                        claim.request_digest,
                        ResourceKind.EXECUTION,
                        claim.execution_id,
                        idempotency.status,
                        idempotency.result_digest,
                        idempotency.error_code,
                        idempotency.created_at,
                        idempotency.updated_at,
                    ),
                )
                or idempotency.status is not IdempotencyStatus.RESERVED
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = claim.started_at
        next_execution = replace(
            current,
            status=ExecutionStatus.STARTED,
            revision=current.revision + 1,
            event_sequence=current.event_sequence + 1,
            updated_at=now,
        )
        execution_replacement = RecordReplacement(
            _projected_record(self, execution_record, next_execution),
            execution_record.storage_version,
        )
        if idempotency_created:
            await transaction.insert_records((idempotency_record,))
            await _replace_checked(
                transaction,
                execution_replacement.record,
                execution_replacement.expected_storage_version,
            )
        else:
            replacements = [execution_replacement]
            next_idempotency = replace(
                idempotency,
                status=IdempotencyStatus.STARTED,
                updated_at=now,
            )
            replacements.append(
                RecordReplacement(
                    _projected_record(
                        self._idempotency,
                        idempotency_record,
                        next_idempotency,
                    ),
                    idempotency_record.storage_version,
                )
            )
            await transaction.replace_records(tuple(replacements))
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            claim.execution_id,
        )
        await transaction.insert_facts(
            (
                StoredFact(
                    stream,
                    next_execution.event_sequence,
                    execution_key,
                    ExecutionEventType.EXECUTION_STARTED.value,
                    None,
                    None,
                    {},
                ),
            )
        )
        _logger.debug(
            "execution start claimed: execution=%s idempotency_created=%s",
            claim.execution_id,
            idempotency_created,
        )
        return next_execution

    async def claim_next_agent_run(
        self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            return await self.claim_next_agent_run_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                expected_agent_run_sequence=expected_agent_run_sequence,
            )

        return await self._store.mutate(mutate)

    async def claim_next_agent_run_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_sequence: int,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", execution_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ExecutionRecord)
        if (
            current.revision != expected_revision
            or current.agent_run_sequence != expected_agent_run_sequence
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            agent_run_sequence=current.agent_run_sequence + 1,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            _projected_record(self, record, next_value),
            record.storage_version,
        )
        return next_value

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=commit.tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_sequence=commit.expected_event_sequence,
            next_status=ExecutionStatus.START_UNKNOWN,
            event_type=ExecutionEventType.EXECUTION_START_UNKNOWN,
            payload={},
            updated_at=commit.occurred_at,
        )


    async def request_cancel(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=commit.tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_sequence=commit.expected_event_sequence,
            next_status=ExecutionStatus.CANCELLING,
            event_type=ExecutionEventType.CANCEL_REQUESTED,
            payload={"operation_id": commit.operation_id},
            updated_at=commit.requested_at,
            pending_events=pending_events,
        )

    async def advance_event_sequence(
        self, execution_id: str, *, tenant_id: str, expected_sequence: int
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            record = await transaction.get_record(self._key("execution", execution_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ExecutionRecord)
            if current.event_sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            next_value = replace(
                current,
                event_sequence=current.event_sequence + 1,
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, next_value),
                record.storage_version,
            )
            return next_value

        return await self._store.mutate(mutate)


    async def _transition_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_sequence: int,
        next_status: ExecutionStatus,
        expected_status: ExecutionStatus | None = None,
        event_type: ExecutionEventType,
        payload: Mapping[str, object],
        updated_at: datetime,
        pending_events: Sequence[ExecutionEventAppend] = (),
        transaction: StateTransaction | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if any(not isinstance(event.event_type, ExecutionEventType) for event in pending_events):
            raise TypeError("pending execution events must use ExecutionEventType")
        key = self._key("execution", execution_id)
        stream = stream_digest(self._namespace, self._tenant_id, self._domain.value, "execution", execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != expected_revision
                or stored_value.event_sequence != expected_event_sequence
                or expected_status is not None and stored_value.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            event_count = len(pending_events) + 1
            next_value = replace(
                stored_value,
                status=next_status,
                revision=stored_value.revision + event_count,
                event_sequence=stored_value.event_sequence + event_count,
                updated_at=updated_at,
            )
            candidate = _projected_record(self, stored, next_value)
            await _replace_checked(transaction, candidate, stored.storage_version)
            first_sequence = stored_value.event_sequence + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type.value,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_value.event_sequence,
                    key,
                    event_type.value,
                    None,
                    None,
                    payload,
                )
            )
            await transaction.insert_facts(tuple(facts))
            return next_value

        if transaction is not None:
            return await mutate(transaction)
        return await self._store.mutate(mutate)

    async def terminal_idempotency_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
    ) -> IdempotencyTerminalUpdate | None:
        """Build the terminal idempotency update from the active transaction."""
        scope = self._idempotency._scope(
            "idempotency",
            "resource",
            [ResourceKind.EXECUTION.value, commit.execution.execution_id],
        )
        records = await transaction.list_records(RecordQuery(scope_digest=scope, kind="idempotency"))
        if len(records) > 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not records:
            return None
        identity = await self._idempotency._decode(records[0], IdempotencyRecord)
        next_status = (
            IdempotencyStatus.COMPLETED
            if commit.execution.status is ExecutionStatus.SUCCEEDED
            else IdempotencyStatus.CANCELLED
            if commit.execution.status is ExecutionStatus.CANCELLED
            else IdempotencyStatus.FAILED
        )
        return IdempotencyTerminalUpdate(
            identity.scope,
            identity.idempotency_key_digest,
            identity.status,
            next_status,
            identity.request_digest,
            None if commit.result.output is None else commit.result.output.digest,
            commit.execution.error_code,
        )

    async def commit_terminal(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        return await self._commit_terminal(commit, pending_events=pending_events)

    async def _commit_terminal(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
        transaction: StateTransaction | None = None,
    ) -> ExecutionTerminalCommitResult:
        _require_repository_tenant(commit.execution.tenant_id, self._tenant_id)
        key = self._key("execution", commit.execution.execution_id)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            commit.execution.execution_id,
        )

        async def mutate(transaction: StateTransaction) -> ExecutionTerminalCommitResult:
            if any(not isinstance(event.event_type, ExecutionEventType) for event in pending_events):
                raise TypeError("pending execution events must use ExecutionEventType")
            record_keys = [key]
            id_key = None
            if commit.idempotency is not None:
                identity = self._idempotency._identity_key(
                    commit.idempotency.scope,
                    commit.idempotency.idempotency_key_digest,
                )
                id_key = self._idempotency._key("idempotency", identity)
                record_keys.append(id_key)
            stored_records = await transaction.get_records(record_keys)
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != commit.expected_revision
                or stored_value.event_sequence != commit.expected_event_sequence
            ):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            id_record = None
            id_value = None
            if commit.idempotency is not None:
                assert id_key is not None
                id_record = stored_records.get(id_key)
                if id_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                id_value = await self._idempotency._decode(
                    id_record,
                    IdempotencyRecord,
                )
                if id_value.status is not commit.idempotency.expected_status:
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            operation_record = None
            current_operation = None
            if commit.operation is not None:
                operation_key_value = operation_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    commit.operation.operation_id,
                )
                operation_record = await transaction.get_operation(operation_key_value)
                if operation_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = _decode_operation(operation_record)
                if current_operation.status is not commit.operation.expected_status:
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            now = (
                await transaction.now()
                if commit.idempotency is not None or commit.operation is not None
                else None
            )
            next_execution = replace(
                commit.execution,
                revision=stored_value.revision + len(pending_events) + 1,
                event_sequence=stored_value.event_sequence + len(pending_events) + 1,
                result=commit.result,
            )
            replacements = [
                RecordReplacement(
                    _projected_record(self, stored, next_execution),
                    stored.storage_version,
                )
            ]
            if commit.idempotency is not None:
                assert id_record is not None
                assert id_value is not None
                assert now is not None
                next_id = replace(
                    id_value,
                    status=commit.idempotency.next_status,
                    request_digest=commit.idempotency.request_digest,
                    result_digest=commit.idempotency.result_digest,
                    error_code=commit.idempotency.error_code,
                    updated_at=now,
                )
                replacements.append(
                    RecordReplacement(
                        _projected_record(
                            self._idempotency,
                            id_record,
                            next_id,
                        ),
                        id_record.storage_version,
                    )
                )
            first_sequence = stored_value.event_sequence + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type.value,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_execution.event_sequence,
                    key,
                    commit.terminal_event_type.value,
                    None,
                    None,
                    commit.terminal_event_payload,
                )
            )
            next_operation = None
            if commit.operation is not None:
                assert current_operation is not None
                assert operation_record is not None
                assert now is not None
                next_operation = replace(
                    current_operation,
                    status=commit.operation.next_status,
                    result_ref=commit.operation.result_ref,
                    result_digest=commit.operation.result_digest,
                    error_code=commit.operation.error_code,
                    updated_at=now,
                )
            await transaction.replace_records(tuple(replacements))
            await transaction.insert_facts(tuple(facts))
            if commit.operation is not None:
                assert next_operation is not None
                assert operation_record is not None
                if not await transaction.replace_operation(
                    _stored_from_operation(next_operation, operation_record),
                    expected_state=commit.operation.expected_status.value,
                ):
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            _logger.debug(
                "execution terminal boundary committed: execution=%s pending_events=%s "
                "idempotency=%s operation=%s",
                commit.execution.execution_id,
                len(pending_events),
                commit.idempotency is not None,
                commit.operation is not None,
            )
            return ExecutionTerminalCommitResult(next_execution, commit.result)

        if transaction is not None:
            return await mutate(transaction)
        return await self._store.mutate(mutate)

    async def commit_terminal_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionTerminalCommitResult:
        return await self._commit_terminal(
            commit,
            pending_events=pending_events,
            transaction=transaction,
        )

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return None if execution is None else execution.result

    async def get_history_seal(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionHistorySealRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def read(transaction: StateTransaction) -> ExecutionHistorySealRecord | None:
            return await self._get_history_seal_in_transaction(transaction, execution_id)

        return await self._store.read(read)

    async def _get_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
    ) -> ExecutionHistorySealRecord | None:
        record = await transaction.get_record(
            self._key("execution_history_seal", execution_id)
        )
        if record is None:
            return None
        if record.kind != "execution_history_seal":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._decode(record, ExecutionHistorySealRecord)
        if value.execution_id != execution_id or value.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def get_history_head(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionHistoryHeadRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        return await self._store.read(
            lambda transaction: self._get_history_head_in_transaction(
                transaction,
                execution_id,
            )
        )

    async def _get_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
    ) -> ExecutionHistoryHeadRecord | None:
        record = await transaction.get_record(
            self._key("execution_history_head", execution_id)
        )
        if record is None:
            return None
        return await self._decode_history_head_record(record, execution_id)

    async def _decode_history_head_record(
        self,
        record: StoredRecord,
        execution_id: str,
    ) -> ExecutionHistoryHeadRecord:
        if record.kind != "execution_history_head":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._decode(record, ExecutionHistoryHeadRecord)
        if value.execution_id != execution_id or value.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def require_open_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        expected_revision: "int | None" = None,
    ) -> tuple[ExecutionHistoryHeadRecord, StoredRecord]:
        """Read and guard the OPEN history head for one execution-domain mutation."""
        key = self._key("execution_history_head", execution_id)
        record = await transaction.get_record(key)
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.kind != "execution_history_head":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        head = await self._decode(record, ExecutionHistoryHeadRecord)
        if head.execution_id != execution_id or head.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if head.state is not ExecutionHistoryState.OPEN:
            _logger.info(
                "execution history head is sealed: execution=%s revision=%s",
                execution_id,
                head.revision,
            )
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if expected_revision is not None and head.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        guarded = await transaction.guard_record(
            key,
            expected_storage_version=record.storage_version,
        )
        if guarded is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return head, guarded

    async def replace_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        current_record: StoredRecord,
        next_head: ExecutionHistoryHeadRecord,
    ) -> ExecutionHistoryHeadRecord:
        key = self._key("execution_history_head", next_head.execution_id)
        if key != current_record.key_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        upgraded = replace(
            self._stored(
                "execution_history_head",
                next_head.execution_id,
                next_head,
                state=next_head.state.value,
            ),
            storage_version=current_record.storage_version + 1,
        )
        if not await transaction.replace_record(
            upgraded,
            expected_storage_version=current_record.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return next_head

    async def insert_history_head_in_transaction(
        self,
        transaction: StateTransaction,
        head: ExecutionHistoryHeadRecord,
    ) -> ExecutionHistoryHeadRecord:
        _require_repository_tenant(head.tenant_id, self._tenant_id)
        key = self._key("execution_history_head", head.execution_id)
        if await transaction.get_record(key) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await transaction.insert_record(self._stored_history_head(head))
        return head

    def _stored_history_head(self, head: ExecutionHistoryHeadRecord) -> StoredRecord:
        return self._stored(
            "execution_history_head",
            head.execution_id,
            head,
            state=head.state.value,
        )

    async def put_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        seal: ExecutionHistorySealRecord,
    ) -> ExecutionHistorySealRecord:
        _require_repository_tenant(seal.tenant_id, self._tenant_id)
        key = self._key("execution_history_seal", seal.execution_id)
        current = await transaction.get_record(key)
        if current is not None:
            existing = await self._decode(current, ExecutionHistorySealRecord)
            if existing != seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        await transaction.insert_record(
            self._stored(
                "execution_history_seal",
                seal.execution_id,
                seal,
            )
        )
        _logger.info(
            "execution history seal persisted: execution=%s runs=%s",
            seal.execution_id,
            len(seal.run_heads),
        )
        return seal


class EventRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.EXECUTION)

    async def append_many(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_sequence: int | None = None,
    ) -> tuple[ExecutionEventRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not events:
            return ()
        result = await self._store.mutate(
            lambda transaction: self.append_many_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                events=events,
                expected_sequence=expected_sequence,
            )
        )
        _logger.debug(
            "execution events appended: execution=%s count=%s last_sequence=%s",
            execution_id,
            len(events),
            result[-1].sequence,
        )
        return result

    async def append_many_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_sequence: int | None = None,
    ) -> tuple[ExecutionEventRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not events:
            return ()
        if any(not isinstance(event.event_type, ExecutionEventType) for event in events):
            raise TypeError("event repository accepts durable ExecutionEventType only")
        if any(not isinstance(event.payload, Mapping) for event in events):
            raise TypeError("event payload must be a mapping")
        key = self._key("execution", execution_id)
        stream = stream_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            execution_id,
        )
        current = await transaction.get_record(key)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        execution = await self._decode(current, ExecutionRecord)
        if expected_sequence is not None and execution.event_sequence != expected_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = await transaction.now()
        first_sequence = execution.event_sequence + 1
        last_sequence = execution.event_sequence + len(events)
        next_execution = replace(
            execution,
            event_sequence=last_sequence,
            revision=execution.revision + len(events),
            updated_at=now,
        )
        await _replace_checked(
            transaction,
            _projected_record(self, current, next_execution),
            current.storage_version,
        )
        facts = tuple(
            StoredFact(
                stream,
                first_sequence + index,
                key,
                event.event_type.value,
                None,
                None,
                event.payload,
            )
            for index, event in enumerate(events)
        )
        await transaction.insert_facts(facts)
        return tuple(
            ExecutionEventRecord(
                execution_id,
                tenant_id,
                first_sequence + index,
                event.event_type,
                event.payload,
            )
            for index, event in enumerate(events)
        )

    async def append_next(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        event_type: ExecutionEventType,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_sequence=None,
        )
        return values[0]

    async def append_expected(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_sequence: int,
        event_type: ExecutionEventType,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_sequence=expected_sequence,
        )
        return values[0]

    async def append(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_sequence: int,
        event_type: ExecutionEventType,
        payload: object,
    ) -> ExecutionEventRecord:
        return await self.append_expected(
            execution_id,
            tenant_id=tenant_id,
            expected_sequence=expected_sequence,
            event_type=event_type,
            payload=payload,
        )

    async def list(
        self, execution_id: str, *, tenant_id: str, after_sequence: int, limit: int
    ) -> Page[ExecutionEventRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        stream = stream_digest(self._namespace, self._tenant_id, self._domain.value, "execution", execution_id)
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(stream, after_sequence=after_sequence, limit=limit + 1)
            )
        )
        items = tuple(
            ExecutionEventRecord(execution_id, tenant_id, value.sequence, ExecutionEventType(value.kind), value.data)
            for value in values[:limit]
        )
        return Page(items, str(items[-1].sequence) if len(values) > limit and items else None)


class ApprovalRepositoryImpl(_ResourceRepository[ApprovalRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
            kind="approval",
            resource_kind=ResourceKind.APPROVAL,
            value_type=ApprovalRecord,
            identity_field="approval_id",
        )

    async def decide(
        self,
        approval_id: str,
        *,
        tenant_id: str,
        expected_status: ApprovalStatus,
        idempotency_key_digest: str,
        decision: ApprovalDecision,
        principal_id: str,
        decision_digest: str,
        decided_at: datetime,
    ) -> ApprovalRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        async def mutate(transaction: StateTransaction) -> ApprovalRecord:
            record = await transaction.get_record(self._key("approval", approval_id))
            if record is None:
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            current = await self._decode(record, ApprovalRecord)
            if current.status is not expected_status:
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            value = replace(
                current,
                status=ApprovalStatus.APPROVED if decision is ApprovalDecision.APPROVE else ApprovalStatus.DENIED,
                idempotency_key_digest=idempotency_key_digest,
                decision=decision,
                decided_by=principal_id,
                decision_digest=decision_digest,
                decided_at=decided_at,
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, value),
                record.storage_version,
            )
            return value

        return await self._store.mutate(mutate)

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "approval",
            scope=self._scope("approval", "execution", execution_id),
            states=frozenset({ApprovalStatus.PENDING.value}),
        )
        return tuple([await self._decode(record, ApprovalRecord) for record in records])

class ExternalCallRepositoryImpl(_ResourceRepository[ExternalCallRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
            kind="external_call",
            resource_kind=ResourceKind.EXTERNAL_CALL,
            value_type=ExternalCallRecord,
            identity_field="call_id",
        )

    async def create_call(self, record: ExternalCallRecord) -> ExternalCallRecord:
        return await self.create(record)

    async def supply(
        self,
        call_id: str,
        *,
        tenant_id: str,
        expected_status: ExternalCallStatus,
        idempotency_key_digest: str,
        object_ref: ObjectRef,
        payload_digest: str,
        supplied_at: datetime,
    ) -> ExternalCallRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        async def mutate(transaction: StateTransaction) -> ExternalCallRecord:
            record = await transaction.get_record(self._key("external_call", call_id))
            if record is None:
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            current = await self._decode(record, ExternalCallRecord)
            if current.status is not expected_status:
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            value = replace(
                current,
                status=ExternalCallStatus.SUPPLIED,
                idempotency_key_digest=idempotency_key_digest,
                object_ref=object_ref,
                payload_digest=payload_digest,
                supplied_at=supplied_at,
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, value),
                record.storage_version,
            )
            return value

        return await self._store.mutate(mutate)

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalCallRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "external_call",
            scope=self._scope("external_call", "execution", execution_id),
            states=frozenset({ExternalCallStatus.PENDING.value}),
        )
        return tuple([await self._decode(record, ExternalCallRecord) for record in records])

class RecoveryCheckpointRepositoryImpl(_ResourceRepository[RecoveryCheckpoint]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
            kind="recovery_state",
            resource_kind=ResourceKind.EXECUTION,
            value_type=RecoveryCheckpoint,
            identity_field="execution_id",
        )

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        if tenant_id != self._tenant_id:
            return ()
        async def read(transaction: StateTransaction) -> tuple[RecoveryCheckpoint, ...]:
            admissions = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("recovery_admission"),
                    kind="recovery_admission",
                )
            )
            decoded_admissions = []
            for record in admissions:
                decoded_admissions.append(
                    (record, await self._decode(record, RecoveryAdmissionRecord))
                )
            states = await transaction.get_records(
                tuple(self._state_key(admission.execution_id) for _, admission in decoded_admissions)
            )
            values = []
            for admission_record, admission in decoded_admissions:
                state_record = states.get(self._state_key(admission.execution_id))
                if state_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values.append(await self._compose(admission_record, state_record))
            return tuple(values)

        return await self._store.read(read)

    async def list_recoverable_page(
        self,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[RecoveryCheckpoint]:
        if tenant_id != self._tenant_id:
            return Page(())
        if not 1 <= limit <= 1000:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)

        async def read(transaction: StateTransaction) -> Page[RecoveryCheckpoint]:
            after_sort_key, after_key_digest = _decode_record_cursor(cursor)
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("recovery_active"),
                    kind="recovery_active",
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            has_more = len(records) > limit
            selected = records[:limit]
            active_values = []
            for record in selected:
                active = await self._decode(record, RecoveryActiveRecord)
                self._validate_active_record(record, active, active.execution_id)
                active_values.append(active)
            keys = tuple(
                key
                for value in active_values
                for key in (
                    self._admission_key(value.execution_id),
                    self._state_key(value.execution_id),
                )
            )
            related = await transaction.get_records(keys)
            values: list[RecoveryCheckpoint] = []
            for value in active_values:
                admission = related.get(self._admission_key(value.execution_id))
                state = related.get(self._state_key(value.execution_id))
                if admission is None or state is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                checkpoint = await self._compose(admission, state)
                if checkpoint.state is RecoveryCheckpointState.COMPLETED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values.append(checkpoint)
            next_cursor = _record_cursor(selected[-1]) if has_more and selected else None
            return Page(tuple(values), next_cursor)

        return await self._store.read(read)

    async def validate_recovery_active_index(
        self,
        *,
        tenant_id: str,
    ) -> RecoveryIntegrityReport:
        """Validate the recovery indexes against one stable generation."""
        if tenant_id != self._tenant_id:
            return RecoveryIntegrityReport(0, 0, ())
        _logger.info(
            "recovery active index validation started: tenant=%s",
            self._tenant_id,
        )

        g0 = await self._store.read(
            lambda transaction: transaction.get_sequence(
                self._integrity_generation_key()
            )
        )
        active_ids: set[str] = set()
        inconsistent: set[str] = set()
        cursor: tuple[str, bytes] | None = None
        while True:
            page_cursor = cursor
            records = await self._store.read(
                lambda transaction, cursor_value=page_cursor: transaction.list_records(
                    RecordQuery(
                        partition_digest=self._partition("recovery_active"),
                        kind="recovery_active",
                        after_sort_key=(
                            None if cursor_value is None else cursor_value[0]
                        ),
                        after_key_digest=(
                            None if cursor_value is None else cursor_value[1]
                        ),
                        limit=_RECOVERY_PAGE_SIZE,
                    )
                )
            )
            if not records:
                break
            for record in records:
                try:
                    active = await self._decode(record, RecoveryActiveRecord)
                    self._validate_active_record(
                        record,
                        active,
                        active.execution_id,
                    )
                except (KeyError, TypeError, ValueError):
                    self._record_invalid_recovery_id(
                        record,
                        active_ids,
                        inconsistent,
                    )
                    continue
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                        raise
                    self._record_invalid_recovery_id(
                        record,
                        active_ids,
                        inconsistent,
                    )
                    continue
                active_ids.add(active.execution_id)
            if len(records) < _RECOVERY_PAGE_SIZE:
                break
            last = records[-1]
            cursor = (last.sort_key, last.key_digest)

        expected_active: set[str] = set()
        admission_ids: set[str] = set()
        state_ids: set[str] = set()
        cursor = None
        while True:
            page_cursor = cursor
            async def read_page(
                transaction: StateTransaction,
                cursor_value: tuple[str, bytes] | None = page_cursor,
            ) -> tuple[
                tuple[tuple[StoredRecord, RecoveryAdmissionRecord], ...],
                Mapping[bytes, StoredRecord],
                tuple[str, ...],
                tuple[str, bytes] | None,
            ]:
                admissions = await transaction.list_records(
                    RecordQuery(
                        partition_digest=self._partition("recovery_admission"),
                        kind="recovery_admission",
                        after_sort_key=(
                            None if cursor_value is None else cursor_value[0]
                        ),
                        after_key_digest=(
                            None if cursor_value is None else cursor_value[1]
                        ),
                        limit=_RECOVERY_PAGE_SIZE,
                    )
                )
                decoded: list[tuple[StoredRecord, RecoveryAdmissionRecord]] = []
                invalid_ids: list[str] = []
                for record in admissions:
                    try:
                        admission = await self._decode(
                            record,
                            RecoveryAdmissionRecord,
                        )
                        self._validate_admission_record(
                            record,
                            admission,
                            admission.execution_id,
                        )
                    except (KeyError, TypeError, ValueError):
                        invalid_ids.append(record.sort_key)
                        continue
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                            raise
                        invalid_ids.append(record.sort_key)
                        continue
                    decoded.append((record, admission))
                states = await transaction.get_records(
                    tuple(
                        self._state_key(admission.execution_id)
                        for _record, admission in decoded
                    )
                )
                last_cursor = None
                if admissions:
                    last = admissions[-1]
                    last_cursor = (last.sort_key, last.key_digest)
                return tuple(decoded), states, tuple(invalid_ids), last_cursor

            (
                decoded_admissions,
                states,
                invalid_ids,
                last_cursor,
            ) = await self._store.read(read_page)
            admission_ids.update(invalid_ids)
            inconsistent.update(invalid_ids)
            if not decoded_admissions:
                if len(invalid_ids) < _RECOVERY_PAGE_SIZE:
                    break
                cursor = last_cursor
                continue
            for record, admission in decoded_admissions:
                admission_ids.add(admission.execution_id)
                state = states.get(self._state_key(admission.execution_id))
                if state is None:
                    inconsistent.add(admission.execution_id)
                    continue
                try:
                    state_value = await self._decode(state, RecoveryStateRecord)
                    self._validate_state_record(
                        state,
                        state_value,
                        admission.execution_id,
                    )
                except (KeyError, TypeError, ValueError):
                    self._record_invalid_recovery_id(
                        state,
                        state_ids,
                        inconsistent,
                    )
                    inconsistent.add(admission.execution_id)
                    continue
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                        raise
                    self._record_invalid_recovery_id(
                        state,
                        state_ids,
                        inconsistent,
                    )
                    inconsistent.add(admission.execution_id)
                    continue
                state_ids.add(admission.execution_id)
                checkpoint = await self._compose(record, state)
                if checkpoint.state is not RecoveryCheckpointState.COMPLETED:
                    expected_active.add(admission.execution_id)
            page_size = len(decoded_admissions) + len(invalid_ids)
            if page_size < _RECOVERY_PAGE_SIZE:
                break
            cursor = last_cursor

        cursor = None
        while True:
            page_cursor = cursor
            states = await self._store.read(
                lambda transaction, cursor_value=page_cursor: self._read_recovery_state_page(
                    transaction,
                    cursor_value,
                )
            )
            if not states:
                break
            for record in states:
                try:
                    state = await self._decode(record, RecoveryStateRecord)
                    self._validate_state_record(
                        record,
                        state,
                        state.execution_id,
                    )
                except (KeyError, TypeError, ValueError):
                    self._record_invalid_recovery_id(
                        record,
                        state_ids,
                        inconsistent,
                    )
                    continue
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                        raise
                    self._record_invalid_recovery_id(
                        record,
                        state_ids,
                        inconsistent,
                    )
                    continue
                state_ids.add(state.execution_id)
            if len(states) < _RECOVERY_PAGE_SIZE:
                break
            last = states[-1]
            cursor = (last.sort_key, last.key_digest)

        g1 = await self._store.read(
            lambda transaction: transaction.get_sequence(
                self._integrity_generation_key()
            )
        )
        if g0 != g1:
            raise AIError(
                ErrorCode.STORAGE_CONFLICT,
                "recovery integrity validation snapshot changed",
            )
        inconsistent.update(active_ids ^ expected_active)
        inconsistent.update(admission_ids ^ state_ids)
        report = RecoveryIntegrityReport(
            len(active_ids),
            len(admission_ids),
            tuple(sorted(inconsistent)),
        )
        if report.inconsistent_execution_ids:
            _logger.error(
                "recovery active index validation failed: tenant=%s "
                "inconsistent=%s",
                self._tenant_id,
                len(report.inconsistent_execution_ids),
            )
        else:
            _logger.info(
                "recovery active index validation passed: tenant=%s active=%s",
                self._tenant_id,
                report.active_count,
            )
        return report

    async def _ensure_active_in_transaction(
        self,
        transaction: StateTransaction,
        checkpoint: RecoveryCheckpoint,
        *,
        active_record: StoredRecord | None | object = _ACTIVE_RECORD_UNSET,
    ) -> bool:
        key = self._active_key(checkpoint.execution_id)
        if active_record is _ACTIVE_RECORD_UNSET:
            current = await transaction.get_record(key)
        else:
            current = active_record
        if checkpoint.state is RecoveryCheckpointState.COMPLETED:
            if current is None:
                return False
            if not await transaction.delete_record(
                key,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return True
        if current is None:
            await transaction.insert_record(
                self._stored(
                    "recovery_active",
                    checkpoint.execution_id,
                    RecoveryActiveRecord(
                        checkpoint.execution_id,
                        checkpoint.tenant_id,
                    ),
                    state=checkpoint.state.value,
                )
            )
            return True
        active = await self._decode(current, RecoveryActiveRecord)
        self._validate_active_record(current, active, checkpoint.execution_id)
        return False

    async def _read_recovery_state_page(
        self,
        transaction: StateTransaction,
        cursor: tuple[str, bytes] | None,
    ) -> tuple[StoredRecord, ...]:
        return await transaction.list_records(
            RecordQuery(
                partition_digest=self._partition("recovery_state"),
                kind="recovery_state",
                after_sort_key=None if cursor is None else cursor[0],
                after_key_digest=None if cursor is None else cursor[1],
                limit=_RECOVERY_PAGE_SIZE,
            )
        )

    async def _bump_integrity_generation_in_transaction(
        self,
        transaction: StateTransaction,
    ) -> None:
        await transaction.next_sequence(self._integrity_generation_key())

    def _integrity_generation_key(self) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            "recovery_integrity_generation",
            "global",
        )

    def _validate_active_record(
        self,
        record: StoredRecord,
        active: RecoveryActiveRecord,
        execution_id: str,
    ) -> None:
        if (
            record.key_digest != self._active_key(execution_id)
            or record.kind != "recovery_active"
            or active.execution_id != execution_id
            or active.tenant_id != self._tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_admission_record(
        self,
        record: StoredRecord,
        admission: RecoveryAdmissionRecord,
        execution_id: str,
    ) -> None:
        if (
            record.key_digest != self._admission_key(execution_id)
            or record.kind != "recovery_admission"
            or admission.execution_id != execution_id
            or admission.tenant_id != self._tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_state_record(
        self,
        record: StoredRecord,
        state: RecoveryStateRecord,
        execution_id: str,
    ) -> None:
        if (
            record.key_digest != self._state_key(execution_id)
            or record.kind != "recovery_state"
            or state.execution_id != execution_id
            or state.tenant_id != self._tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _record_invalid_recovery_id(
        self,
        record: StoredRecord,
        ids: set[str],
        inconsistent: set[str],
    ) -> None:
        if record.sort_key:
            ids.add(record.sort_key)
            inconsistent.add(record.sort_key)

    def _active_key(self, execution_id: str) -> bytes:
        return self._key("recovery_active", execution_id)

    async def get(self, execution_id: str, *, tenant_id: str) -> RecoveryCheckpoint | None:
        if tenant_id != self._tenant_id:
            return None

        async def read(transaction: StateTransaction) -> RecoveryCheckpoint | None:
            return await self.get_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
            )

        return await self._store.read(read)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None:
        if tenant_id != self._tenant_id:
            return None
        keys = (self._admission_key(execution_id), self._state_key(execution_id))
        records = await transaction.get_records(keys)
        admission = records.get(keys[0])
        state = records.get(keys[1])
        if admission is not None and state is not None:
            return await self._compose(admission, state)
        return None

    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        return await self._store.mutate(
            lambda transaction: self.admit_in_transaction(transaction, record)
        )

    async def compare_and_swap(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: RecoveryCheckpoint,
    ) -> RecoveryCheckpoint:
        result = await self._store.mutate(
            lambda transaction: self.compare_and_swap_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                next_record=next_record,
            )
        )
        _logger.debug(
            "recovery checkpoint compare-and-swap committed: execution=%s revision=%s",
            execution_id,
            next_record.revision,
        )
        return result

    async def admit_in_transaction(
        self,
        transaction: StateTransaction,
        record: RecoveryCheckpoint,
    ) -> RecoveryCheckpoint:
        _require_tenant(record, self._tenant_id)
        keys = (
            self._admission_key(record.execution_id),
            self._state_key(record.execution_id),
            self._active_key(record.execution_id),
        )
        records = await transaction.get_records(keys)
        current = records.get(keys[0])
        if current is not None:
            if keys[1] not in records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing = await self._compose(current, records[keys[1]])
            if not _recovery_admission_matches(existing, record):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if await self._ensure_active_in_transaction(
                transaction,
                existing,
                active_record=records.get(keys[2]),
            ):
                await self._bump_integrity_generation_in_transaction(transaction)
            return existing
        if keys[1] in records:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        admission = RecoveryAdmissionRecord(
            record.execution_id,
            record.tenant_id,
            record.input,
            record.created_at,
        )
        state = _recovery_state_record(record)
        active_record = records.get(keys[2])
        values = [
            self._stored("recovery_admission", record.execution_id, admission),
            self._stored("recovery_state", record.execution_id, state),
        ]
        if (
            active_record is None
            and record.state is not RecoveryCheckpointState.COMPLETED
        ):
            values.append(
                self._stored(
                    "recovery_active",
                    record.execution_id,
                    RecoveryActiveRecord(
                        record.execution_id,
                        record.tenant_id,
                    ),
                    state=record.state.value,
                )
            )
        await transaction.insert_records(tuple(values))
        if active_record is not None:
            await self._ensure_active_in_transaction(
                transaction,
                record,
                active_record=active_record,
            )
        await self._bump_integrity_generation_in_transaction(transaction)
        _logger.debug(
            "recovery checkpoint admitted: execution=%s active=%s",
            record.execution_id,
            record.state is not RecoveryCheckpointState.COMPLETED,
        )
        return record

    async def compare_and_swap_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: RecoveryCheckpoint,
    ) -> RecoveryCheckpoint:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        keys = (self._admission_key(execution_id), self._state_key(execution_id))
        records = await transaction.get_records(keys)
        admission_record = records.get(keys[0])
        state_record = records.get(keys[1])
        if admission_record is None or state_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._compose(admission_record, state_record)
        if current.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if not _recovery_admission_matches(current, next_record):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_state = _recovery_state_record(next_record)
        await _replace_checked(
            transaction,
            replace(
                self._stored("recovery_state", execution_id, next_state),
                storage_version=state_record.storage_version + 1,
            ),
            state_record.storage_version,
        )
        await self._ensure_active_in_transaction(transaction, next_record)
        await self._bump_integrity_generation_in_transaction(transaction)
        return next_record

    def _admission_key(self, execution_id: str) -> bytes:
        return self._key("recovery_admission", execution_id)

    def _state_key(self, execution_id: str) -> bytes:
        return self._key("recovery_state", execution_id)

    async def _compose(
        self,
        admission_record: StoredRecord,
        state_record: StoredRecord,
    ) -> RecoveryCheckpoint:
        admission = await self._decode(admission_record, RecoveryAdmissionRecord)
        state = await self._decode(state_record, RecoveryStateRecord)
        return RecoveryCheckpoint(
            admission.execution_id,
            admission.tenant_id,
            admission.input,
            state.step_run_id,
            state.agent_run_sequence,
            state.state,
            state.handoff_phase,
            state.terminal_handoff,
            state.handoff_contract_digest,
            state.pending_operation_id,
            state.revision,
            admission.created_at,
            state.updated_at,
        )


class EvaluationRepositoryImpl(_ResourceRepository[EvaluationRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EVALUATION,
            kind="evaluation",
            resource_kind=ResourceKind.EVALUATION,
            value_type=EvaluationRecord,
            identity_field="evaluation_id",
        )

    async def list_by_execution(self, execution_id: str, *, tenant_id: str) -> tuple[EvaluationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "evaluation",
            scope=self._scope("evaluation", "execution", execution_id),
        )
        return tuple([await self._decode(record, EvaluationRecord) for record in records])


class MemoryRepositoryImpl(_ResourceRepository[MemoryRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.MEMORY,
            kind="memory",
            resource_kind=ResourceKind.MEMORY,
            value_type=MemoryRecord,
            identity_field="memory_id",
        )

    async def put(self, record: MemoryRecord, *, expected_revision: int | None) -> MemoryRecord:
        _require_tenant(record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> MemoryRecord:
            key = self._key("memory", record.memory_id)
            current = await transaction.get_record(key)
            if current is None:
                if expected_revision not in (None, 0):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await transaction.insert_record(self._stored("memory", record.memory_id, record))
                return record
            value = await self._decode(current, MemoryRecord)
            if expected_revision != value.revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current, record),
                current.storage_version,
            )
            return record

        return await self._store.mutate(mutate)

    async def put_with_operation(
        self, record: MemoryRecord, *, expected_revision: int | None, operation: OperationLedgerInput | None
    ) -> tuple[MemoryRecord | None, bool]:
        _require_tenant(record, self._tenant_id)
        if operation is not None:
            _require_tenant(operation, self._tenant_id)
        if operation is None:
            return await self.put(record, expected_revision=expected_revision), False

        async def mutate(transaction: StateTransaction) -> tuple[MemoryRecord | None, bool]:
            _, replayed = await _append_operation(transaction, self, operation)
            key = self._key("memory", record.memory_id)
            current = await transaction.get_record(key)
            if replayed:
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self._decode(current, MemoryRecord), True
            if current is None:
                if expected_revision not in (None, 0):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await transaction.insert_record(self._stored("memory", record.memory_id, record))
                return record, False
            value = await self._decode(current, MemoryRecord)
            if expected_revision != value.revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current, record),
                current.storage_version,
            )
            return record, False

        return await self._store.mutate(mutate)

    async def list(
        self, *, tenant_id: str, memory_scope_digest: str, cursor: str | None, limit: int
    ) -> Page[MemoryRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        records = await self._records(
            "memory",
            scope=self._scope("memory", "memory_scope", memory_scope_digest),
            cursor=cursor,
            limit=limit + 1,
        )
        values = tuple([await self._decode(record, MemoryRecord) for record in records[:limit]])
        next_cursor = _record_cursor(records[limit - 1]) if len(records) > limit else None
        return Page(values, next_cursor)

    async def delete(self, memory_id: str, *, tenant_id: str, expected_revision: int) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> None:
            key = self._key("memory", memory_id)
            current = await transaction.get_record(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            value = await self._decode(current, MemoryRecord)
            if value.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not await transaction.delete_record(
                key,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._store.mutate(mutate)

    async def delete_with_operation(
        self, memory_id: str, *, tenant_id: str, expected_revision: int | None, operation: OperationLedgerInput | None
    ) -> tuple[bool, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if operation is not None:
            _require_tenant(operation, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> tuple[bool, bool]:
            replayed = False
            if operation is not None:
                _, replayed = await _append_operation(transaction, self, operation)
            key = self._key("memory", memory_id)
            current = await transaction.get_record(key)
            if replayed:
                return current is not None, True
            if current is None:
                return False, False
            value = await self._decode(current, MemoryRecord)
            if expected_revision is not None and value.revision != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not await transaction.delete_record(
                key,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return True, False

        return await self._store.mutate(mutate)


class ArtifactRepositoryImpl(_ResourceRepository[ArtifactRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.ARTIFACT,
            kind="artifact",
            resource_kind=ResourceKind.ARTIFACT,
            value_type=ArtifactRecord,
            identity_field="artifact_id",
        )

    async def put_metadata(self, record: ArtifactRecord) -> ArtifactRecord:
        _require_tenant(record, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ArtifactRecord:
            key = self._key("artifact", record.artifact_id)
            current = await transaction.get_record(key)
            if current is not None:
                existing = await self._decode(current, ArtifactRecord)
                if existing == record:
                    return existing
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            await transaction.insert_record(self._stored("artifact", record.artifact_id, record))
            return record

        return await self._store.mutate(mutate)

    async def get_metadata(self, artifact_id: str, *, tenant_id: str) -> ArtifactRecord | None:
        return await self.get(artifact_id, tenant_id=tenant_id)

    async def list_by_execution(
        self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int
    ) -> Page[ArtifactRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        records = await self._records(
            "artifact",
            scope=self._scope("artifact", "execution", execution_id),
            cursor=cursor,
            limit=limit + 1,
        )
        values = tuple([await self._decode(record, ArtifactRecord) for record in records[:limit]])
        next_cursor = _record_cursor(records[limit - 1]) if len(records) > limit else None
        return Page(values, next_cursor)


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
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation = _decode_operation(stored_operation)
        if operation.request_digest != admission.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        if operation.status in {OperationStatus.FAILED, OperationStatus.CANCELLED}:
            raise _stored_operation_error(operation)
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        self._validate_graph(graph_view, graph, nodes)
        if admission_record is not None:
            existing = await self._decode(admission_record, TaskGraphAdmission)
            if existing != admission or operation.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskGraphView(graph.graph_id, _graph_status(nodes), graph.nodes)
        if operation.status not in {
            OperationStatus.PENDING, OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN, OperationStatus.SUCCEEDED,
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
        self, transaction: StateTransaction, admission: TaskGraphAdmission, graph: TaskGraph
    ) -> TaskGraphView:
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
        now = await transaction.now()
        operation_input = OperationLedgerInput(
            admission.operation_id, self._tenant_id, ResourceKind.TASK_GRAPH, graph.graph_id, None,
            OperationKind.TASK_NODE, OperationStatus.SUCCEEDED, admission.request_digest,
            graph.graph_id, _task_submit_result_digest(graph), None, True, now, now,
        )
        await transaction.insert_records(records)
        await _append_operation(transaction, self, operation_input)
        _logger.info("task graph durably admitted: tenant=%s graph=%s", self._tenant_id, graph.graph_id)
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
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=_stored_operation_error(operation),
                )
        if graph_record is None or admission_record is None or stored_operation is None:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR, error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            )
        try:
            operation = _decode_operation(stored_operation)
            existing = await self._decode(admission_record, TaskGraphAdmission)
            graph_view = await self._decode(graph_record, TaskGraphView)
            nodes = await self._decode_task_nodes(node_records)
            self._validate_operation_identity(operation, admission)
            self._validate_graph(graph_view, graph, nodes)
            if (
                operation.status is not OperationStatus.SUCCEEDED
                or operation.request_digest != admission.request_digest
                or operation.result_ref != graph.graph_id
                or operation.result_digest != _task_submit_result_digest(graph)
                or existing != admission
                or graph_record.scope_digest != self._recovery_scope()
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


class ToolRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.RECOVERY)

    def _tool_key(self, identity: str) -> bytes:
        return self._key("tool_operation", identity)

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
        result = await self._store.mutate(
            lambda transaction: self.admit_in_transaction(transaction, request)
        )
        _logger.debug(
            "tool operation admitted: operation=%s status=%s fence=%s",
            result.tool_operation_id,
            result.status.value,
            result.fence,
        )
        return result

    async def admit_in_transaction(
        self,
        transaction: StateTransaction,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        _require_repository_tenant(request.tenant_id, self._tenant_id)
        validate_lease_owner(request.owner)
        validate_lease_seconds(request.lease_seconds)
        aliases = tuple(
            dict.fromkeys(
                alias_digest(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    "tool_call",
                    [step_run_id, request.tool_call_id],
                )
                for step_run_id in (request.step_run_id, request.recovery_step_run_id)
                if step_run_id is not None
            )
        )

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            resolved = await transaction.resolve_aliases(aliases)
            record_keys = tuple(dict.fromkeys((*resolved.values(), self._tool_key(request.tool_operation_id))))
            records = await transaction.get_records(record_keys)
            resolved_keys = tuple(dict.fromkeys(resolved.values()))
            if len(resolved_keys) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            candidate_key = self._tool_key(request.tool_operation_id)
            candidate_record = records.get(candidate_key)
            if resolved_keys and candidate_record is not None and candidate_key not in resolved_keys:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            existing_key = resolved_keys[0] if resolved_keys else candidate_key
            record = records.get(existing_key)
            if record is None:
                now = await transaction.now()
                value = ToolOperationRecord(
                    tool_operation_id=request.tool_operation_id,
                    tenant_id=self._tenant_id,
                    step_run_id=request.step_run_id,
                    tool_call_id=request.tool_call_id,
                    idempotency_key_digest=request.idempotency_key_digest,
                    tool_name=request.tool_name,
                    arguments_digest=request.arguments_digest,
                    binding_digest=request.binding_digest,
                    replay_safe=request.replay_safe,
                    status=ToolOperationStatus.CLAIMED,
                    owner=request.owner,
                    fence=1,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                )
                await transaction.insert_record(
                    self._stored(
                        "tool_operation",
                        request.tool_operation_id,
                        value,
                        scope=self._scope("tool_operation", "step_run", request.step_run_id),
                        state=value.status.value,
                    )
                )
                await transaction.insert_aliases(
                    tuple(StoredAlias(alias, self._tool_key(request.tool_operation_id)) for alias in aliases)
                )
                return value
            current = await self._decode(record, ToolOperationRecord)
            if not _tool_admission_matches(current, request):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            missing_aliases = tuple(alias for alias in aliases if resolved.get(alias) is None)
            if missing_aliases:
                guarded = await transaction.guard_record(
                    record.key_digest,
                    expected_storage_version=record.storage_version,
                )
                if guarded is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                record = guarded
                await transaction.insert_aliases(
                    tuple(StoredAlias(alias, record.key_digest) for alias in missing_aliases)
                )
            now = await transaction.now() if current.status in {
                ToolOperationStatus.PENDING,
                ToolOperationStatus.CLAIMED,
            } else None
            if current.status in {
                ToolOperationStatus.COMPLETED,
                ToolOperationStatus.FAILED,
                ToolOperationStatus.EFFECT_UNKNOWN,
                ToolOperationStatus.CANCELLED,
            }:
                if current.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                if current.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                return current
            if now is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            expired = current.lease_expires_at is not None and current.lease_expires_at <= now
            if current.status is ToolOperationStatus.CLAIMED and not expired:
                if current.owner == request.owner:
                    return current
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is ToolOperationStatus.CLAIMED and not current.replay_safe:
                value = replace(
                    current,
                    status=ToolOperationStatus.EFFECT_UNKNOWN,
                    lease_expires_at=None,
                    updated_at=now,
                )
            elif current.status in {ToolOperationStatus.PENDING, ToolOperationStatus.CLAIMED}:
                value = replace(
                    current,
                    status=ToolOperationStatus.CLAIMED,
                    owner=request.owner,
                    fence=current.fence + 1,
                    lease_expires_at=now + timedelta(seconds=request.lease_seconds),
                    updated_at=now,
                )
            else:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            await self._replace_tool_in_transaction(transaction, record, value)
            if value.status is ToolOperationStatus.EFFECT_UNKNOWN:
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
            return value

        return await mutate(transaction)

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        _require_tenant(record, self._tenant_id)
        key = self._tool_key(record.tool_operation_id)
        replay_alias = alias_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "tool_call",
            [record.step_run_id, record.tool_call_id],
        )

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            alias_key = await transaction.resolve_alias(replay_alias)
            existing_key = alias_key or key
            existing_record = await transaction.get_record(existing_key)
            if alias_key is not None and existing_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if existing_record is not None:
                existing = await self._decode(existing_record, ToolOperationRecord)
                if not _tool_replay_matches(existing, record):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                if alias_key is None:
                    guarded = await transaction.guard_record(
                        existing_record.key_digest,
                        expected_storage_version=existing_record.storage_version,
                    )
                    if guarded is None:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    await transaction.insert_aliases((StoredAlias(replay_alias, existing_key),))
                return existing
            stored = self._stored(
                "tool_operation",
                record.tool_operation_id,
                record,
                scope=self._scope("tool_operation", "step_run", record.step_run_id),
                state=record.status.value,
            )
            await transaction.insert_record(stored)
            await transaction.insert_aliases((StoredAlias(replay_alias, key),))
            return record

        return await self._store.mutate(mutate)

    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._tool_key(tool_operation_id))
        return None if record is None else await self._decode(record, ToolOperationRecord)

    async def get_by_call(
        self,
        step_run_id: str,
        tool_call_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None:
        if tenant_id != self._tenant_id:
            return None
        replay_alias = alias_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "tool_call",
            [step_run_id, tool_call_id],
        )
        async def read(transaction: StateTransaction) -> ToolOperationRecord | None:
            key = await transaction.resolve_alias(replay_alias)
            if key is None:
                return None
            stored = await transaction.get_record(key)
            return None if stored is None else await self._decode(stored, ToolOperationRecord)

        return await self._store.read(read)

    async def claim(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        validate_lease_seconds(lease_seconds)

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            record = await transaction.get_record(self._tool_key(tool_operation_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ToolOperationRecord)
            if current.status in {
                ToolOperationStatus.COMPLETED,
                ToolOperationStatus.FAILED,
                ToolOperationStatus.EFFECT_UNKNOWN,
                ToolOperationStatus.CANCELLED,
            }:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            now = await transaction.now()
            expired = current.lease_expires_at is not None and current.lease_expires_at <= now
            if current.status is ToolOperationStatus.CLAIMED and not expired:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is ToolOperationStatus.CLAIMED and not current.replay_safe:
                unknown = replace(
                    current,
                    status=ToolOperationStatus.EFFECT_UNKNOWN,
                    lease_expires_at=None,
                    updated_at=now,
                )
                await self._replace_tool_in_transaction(transaction, record, unknown)
                return unknown
            if current.status is not ToolOperationStatus.PENDING and current.status is not ToolOperationStatus.CLAIMED:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            value = replace(
                current,
                status=ToolOperationStatus.CLAIMED,
                owner=owner,
                fence=current.fence + 1,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
            )
            await self._replace_tool_in_transaction(transaction, record, value)
            return value

        result = await self._store.mutate(mutate)
        if result.status is ToolOperationStatus.EFFECT_UNKNOWN:
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        return result

    async def renew(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        validate_lease_seconds(lease_seconds)

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            record = await transaction.get_record(self._tool_key(tool_operation_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ToolOperationRecord)
            now = await transaction.now()
            _require_live_tool_lease(current, owner=owner, fence=fence, now=now)
            expires = now + timedelta(seconds=lease_seconds)
            if not await transaction.update_record_lease(
                record.key_digest,
                expected_storage_version=record.storage_version,
                lease_owner=current.owner,
                lease_fence=current.fence,
                lease_expires_at=expires,
            ):
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            value = replace(current, lease_expires_at=expires)
            return value

        return await self._store.mutate(mutate)

    async def complete_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.COMPLETED,
            requested_result_payload=result_payload,
            value=lambda current, now: replace(
                current,
                status=ToolOperationStatus.COMPLETED,
                result_payload=result_payload,
                lease_expires_at=None,
                updated_at=now,
            ),
        )

    async def fail(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str
    ) -> ToolOperationRecord:
        def value(current: ToolOperationRecord, now: datetime) -> ToolOperationRecord:
            return replace(
                current,
                status=ToolOperationStatus.FAILED,
                error_code=error_code,
                lease_expires_at=None,
                updated_at=now,
            )

        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.FAILED,
            requested_error=error_code,
            value=value,
        )

    async def fail_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.FAILED,
            requested_error=error_code,
            requested_error_payload=error_payload,
            value=lambda current, now: replace(
                current,
                status=ToolOperationStatus.FAILED,
                error_code=error_code,
                error_payload=error_payload,
                lease_expires_at=None,
                updated_at=now,
            ),
        )

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            record = await transaction.get_record(self._tool_key(tool_operation_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ToolOperationRecord)
            now = await transaction.now()
            if current.status is ToolOperationStatus.EFFECT_UNKNOWN:
                if current.owner == owner and current.fence == fence and current.error_code == error_code:
                    return current
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            _require_live_tool_lease(current, owner=owner, fence=fence, now=now)
            value = replace(
                current,
                status=ToolOperationStatus.EFFECT_UNKNOWN,
                error_code=error_code,
                lease_expires_at=None,
                updated_at=now,
            )
            await self._replace_tool_in_transaction(transaction, record, value)
            return value

        return await self._store.mutate(mutate)

    async def _finish_tool(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        terminal_status: ToolOperationStatus,
        requested_result_payload: StoredPayload | None = None,
        requested_error: str | None = None,
        requested_error_payload: StoredPayload | None = None,
        value: object,
        transaction: StateTransaction | None = None,
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            record = await transaction.get_record(self._tool_key(tool_operation_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ToolOperationRecord)
            now = await transaction.now()
            if current.status is ToolOperationStatus.COMPLETED:
                if (
                    terminal_status is ToolOperationStatus.COMPLETED
                    and current.owner == owner
                    and current.fence == fence
                    and current.result_payload == requested_result_payload
                ):
                    return current
                if terminal_status is ToolOperationStatus.COMPLETED:
                    raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is ToolOperationStatus.FAILED:
                if (
                    terminal_status is ToolOperationStatus.FAILED
                    and current.owner == owner
                    and current.fence == fence
                    and current.error_code == requested_error
                    and current.error_payload == requested_error_payload
                ):
                    return current
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status in {
                ToolOperationStatus.EFFECT_UNKNOWN,
                ToolOperationStatus.CANCELLED,
            }:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            _require_live_tool_lease(current, owner=owner, fence=fence, now=now)
            next_value = value(current, now)
            await self._replace_tool_in_transaction(transaction, record, next_value)
            return next_value

        if transaction is not None:
            return await mutate(transaction)
        return await self._store.mutate(mutate)

    async def complete_in_transaction(
        self,
        transaction: StateTransaction,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.COMPLETED,
            requested_result_payload=result_payload,
            value=lambda current, now: replace(
                current,
                status=ToolOperationStatus.COMPLETED,
                result_payload=result_payload,
                lease_expires_at=None,
                updated_at=now,
            ),
            transaction=transaction,
        )

    async def fail_in_transaction(
        self,
        transaction: StateTransaction,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.FAILED,
            requested_error=error_code,
            requested_error_payload=error_payload,
            value=lambda current, now: replace(
                current,
                status=ToolOperationStatus.FAILED,
                error_code=error_code,
                error_payload=error_payload,
                lease_expires_at=None,
                updated_at=now,
            ),
            transaction=transaction,
        )

    async def _replace_tool_in_transaction(
        self,
        transaction: StateTransaction,
        record: StoredRecord,
        value: ToolOperationRecord,
    ) -> None:
        candidate = _projected_record(self, record, value)
        await _replace_checked(transaction, candidate, record.storage_version)


def build_repository_bundle(
    store: StateStore, *, namespace: str, tenant_id: str, domain: RuntimeDomain
) -> dict[str, _RepositoryBase]:
    """Build the one semantic repository implementation for a domain."""
    values: dict[str, _RepositoryBase] = {
        "operations": OperationLedgerRepository(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
    }
    if domain is RuntimeDomain.CONVERSATION:
        values.update(
            sessions=SessionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            histories=ConversationHistoryRepositoryImpl(
                store,
                namespace=namespace,
                tenant_id=tenant_id,
            ),
        )
    elif domain is RuntimeDomain.EXECUTION:
        values.update(
            executions=ExecutionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            events=EventRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is RuntimeDomain.MEMORY:
        values["records"] = MemoryRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is RuntimeDomain.ARTIFACT:
        values["records"] = ArtifactRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is RuntimeDomain.TASK:
        values["tasks"] = TaskRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
        values["admissions"] = TaskAdmissionRepositoryImpl(
            store, namespace=namespace, tenant_id=tenant_id
        )
    elif domain is RuntimeDomain.EVALUATION:
        values.update(
            records=EvaluationRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is RuntimeDomain.RECOVERY:
        values.update(
            approvals=ApprovalRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            external_calls=ExternalCallRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            checkpoints=RecoveryCheckpointRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            tools=ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        )
    return values


def _canonical_record_identity(kind: str, value: object) -> object:
    if isinstance(value, IdempotencyRecord):
        return [value.scope, value.idempotency_key_digest]
    if isinstance(value, TaskGraphView):
        return value.graph_id
    if isinstance(value, TaskNodeView):
        return [value.graph_id, value.node_id]
    if isinstance(value, ToolOperationRecord):
        return value.tool_operation_id
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
    raise TypeError(f"unsupported record kind: {kind}")


def _ensure_session_history(value: SessionRecord) -> SessionRecord:
    if value.history_id is not None:
        return value
    history_id = canonical_sha256(
        {
            "kind": "conversation_history",
            "session_id": value.session_id,
            "tenant_id": value.tenant_id,
        }
    )
    return replace(value, history_id=history_id)


def _new_session_history(value: SessionRecord) -> ConversationHistoryRecord:
    if value.history_id is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ConversationHistoryRecord(
        history_id=value.history_id,
        session_id=value.session_id,
        tenant_id=value.tenant_id,
        parent_history_id=None,
        prefix_index_head_id=None,
        inherited_message_count=0,
        inherited_history_item_count=0,
    )


def _empty_conversation_transcript_head(
    history_id: str,
) -> TranscriptHeadRecord:
    return TranscriptHeadRecord(
        TranscriptOwnerDomain.CONVERSATION,
        history_id,
        0,
        0,
        1,
        0,
        1,
        0,
        HistoryQuality.COMPLETE,
        0,
    )


def _domain_data(value: object) -> dict[str, object]:
    payload = _encode_persisted_domain(value)
    if isinstance(value, (TaskNodeView, ToolOperationRecord)) and isinstance(payload, Mapping):
        fields = payload.get("fields")
        if isinstance(fields, Mapping):
            payload = dict(payload)
            payload["fields"] = {
                key: item for key, item in fields.items() if key not in {"owner", "fence", "lease_expires_at"}
            }
    return encode_envelope({"type": wire_type_id(value), "payload": payload})


def _restore_lease_fields(payload: object, target: type[ValueT]) -> object:
    if target not in {TaskNodeView, ToolOperationRecord} or not isinstance(payload, Mapping):
        return payload
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        return payload
    restored = dict(payload)
    restored["fields"] = {
        **fields,
        "owner": None,
        "fence": 0,
        "lease_expires_at": None,
    }
    return restored


def _record_lease(value: object) -> tuple[str | None, int, datetime | None]:
    if isinstance(value, (TaskNodeView, ToolOperationRecord)):
        return value.owner, value.fence, value.lease_expires_at
    return None, 0, None


def _require_tenant(value: object, tenant_id: str) -> None:
    tenant_value = None
    if isinstance(
        value,
        (
            SessionRecord,
            ExecutionRecord,
            IdempotencyRecord,
            OperationLedgerRecord,
            OperationLedgerInput,
            MemoryRecord,
            EvaluationRecord,
            ArtifactRecord,
            ApprovalRecord,
            ExternalCallRecord,
            RecoveryCheckpoint,
            RecoveryActiveRecord,
            ConversationHistoryRecord,
            ToolOperationRecord,
        ),
    ):
        tenant_value = value.tenant_id
    if tenant_value is not None and tenant_value != tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


def _require_repository_tenant(actual: str, expected: str) -> None:
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


def _record_state(value: object) -> str | None:
    return _status_value(value)


def _domain_revision(value: object) -> int:
    if isinstance(
        value, (SessionRecord, ExecutionRecord, IdempotencyRecord, MemoryRecord, EvaluationRecord, RecoveryCheckpoint)
    ):
        return value.revision
    return 0


def _status_value(value: object) -> str | None:
    if isinstance(
        value,
        (
            SessionRecord,
            ExecutionRecord,
            IdempotencyRecord,
            EvaluationRecord,
            ApprovalRecord,
            ExternalCallRecord,
            TaskNodeView,
            ToolOperationRecord,
        ),
    ):
        return value.status.value
    return None


async def _replace_checked(transaction: StateTransaction, candidate: StoredRecord, expected: int) -> None:
    if not await transaction.replace_record(candidate, expected_storage_version=expected):
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _projected_record(
    repository: _RepositoryBase,
    current: StoredRecord,
    value: object,
) -> StoredRecord:
    _require_tenant(value, repository._tenant_id)
    if isinstance(value, SessionRecord):
        if value.agent_id is None:
            current_value = _decode_enveloped_domain(current.data, SessionRecord)
            value = replace(value, agent_id=current_value.resolved_agent_id())
        else:
            value.resolved_agent_id()
    identity = _canonical_record_identity(current.kind, value)
    projected = repository._stored(current.kind, identity, value, state=_record_state(value))
    return replace(projected, storage_version=current.storage_version + 1)


def _require_explicit_session_agent_id(value: SessionRecord) -> None:
    if value.agent_id is None:
        raise AIError(ErrorCode.AGENT_ID_INVALID)
    try:
        validate_agent_id(value.agent_id)
    except TypeError as error:
        raise AIError(ErrorCode.AGENT_ID_INVALID) from error


def _event_payload(value: object) -> Mapping[str, JsonValue]:
    if isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return {"value": value}  # type: ignore[dict-item]


async def _append_operation(
    transaction: StateTransaction, repository: _RepositoryBase, value: OperationLedgerInput
) -> tuple[OperationLedgerRecord, bool]:
    key = operation_key(repository._namespace, repository._tenant_id, repository._domain.value, value.operation_id)
    existing = await transaction.get_operation(key)
    if existing is not None:
        current = _decode_operation(existing)
        if _operation_matches(current, value):
            return current, True
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    stream = stream_digest(
        repository._namespace,
        repository._tenant_id,
        repository._domain.value,
        "operation",
        [value.resource_kind.value, value.resource_id],
    )
    sequence = await transaction.next_sequence(
        sequence_key(
            repository._namespace,
            repository._tenant_id,
            repository._domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )
    )
    await transaction.insert_operation(
        StoredOperation(key, stream, sequence, value.status.value, value.compactable, _domain_data(value))
    )
    return _operation_record(value, sequence), False


def _decode_operation(value: StoredOperation) -> OperationLedgerRecord:
    candidate = _decode_enveloped_domain(value.data, OperationLedgerInput)
    return _operation_record(candidate, value.sequence)  # type: ignore[arg-type]


def _operation_record(value: OperationLedgerInput, sequence: int) -> OperationLedgerRecord:
    return OperationLedgerRecord(
        value.operation_id,
        value.tenant_id,
        value.resource_kind,
        value.resource_id,
        value.execution_id,
        value.operation_kind,
        value.status,
        value.request_digest,
        value.result_ref,
        value.result_digest,
        value.error_code,
        value.compactable,
        sequence,
        value.created_at,
        value.updated_at,
    )


def _stored_from_operation(value: OperationLedgerRecord, current: StoredOperation) -> StoredOperation:
    candidate = OperationLedgerInput(
        value.operation_id,
        value.tenant_id,
        value.resource_kind,
        value.resource_id,
        value.execution_id,
        value.operation_kind,
        value.status,
        value.request_digest,
        value.result_ref,
        value.result_digest,
        value.error_code,
        value.compactable,
        value.created_at,
        value.updated_at,
    )
    return replace(current, state=value.status.value, compactable=value.compactable, data=_domain_data(candidate))


def _operation_matches(current: OperationLedgerRecord, candidate: OperationLedgerInput) -> bool:
    return operation_replay_matches(current, candidate)


def _same_idempotency(left: IdempotencyRecord, right: IdempotencyRecord) -> bool:
    return (
        _same_idempotency_identity(left, right)
        and left.resource_id == right.resource_id
    )


def _same_idempotency_identity(left: IdempotencyRecord, right: IdempotencyRecord) -> bool:
    """Compare only the immutable request identity, never a candidate resource id."""
    return (
        left.tenant_id == right.tenant_id
        and left.runtime_domain is right.runtime_domain
        and left.scope == right.scope
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.request_digest == right.request_digest
        and left.resource_kind is right.resource_kind
    )


def _tool_replay_matches(left: ToolOperationRecord, right: ToolOperationRecord) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.step_run_id == right.step_run_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and left.binding_digest == right.binding_digest
        and left.replay_safe == right.replay_safe
    )


def _tool_admission_matches(left: ToolOperationRecord, right: ToolOperationAdmission) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.tool_operation_id == right.tool_operation_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and left.binding_digest == right.binding_digest
        and left.replay_safe is right.replay_safe
        and left.step_run_id in {right.step_run_id, right.recovery_step_run_id}
    )


def _execution_replay_matches(left: ExecutionRecord, right: ExecutionRecord) -> bool:
    return (
        left.execution_id == right.execution_id
        and left.tenant_id == right.tenant_id
        and left.session_id == right.session_id
        and left.binding_digest == right.binding_digest
        and left.parent_execution_id == right.parent_execution_id
        and left.root_execution_id == right.root_execution_id
        and left.source_execution_id == right.source_execution_id
        and left.base_execution_id == right.base_execution_id
        and left.lineage_kind is right.lineage_kind
    )


def _recovery_admission_matches(
    left: RecoveryCheckpoint,
    right: RecoveryCheckpoint,
) -> bool:
    return (
        left.execution_id == right.execution_id
        and left.tenant_id == right.tenant_id
        and left.input == right.input
        and left.created_at == right.created_at
    )


def _recovery_state_record(value: RecoveryCheckpoint) -> RecoveryStateRecord:
    return RecoveryStateRecord(
        value.execution_id,
        value.tenant_id,
        value.step_run_id,
        value.agent_run_sequence,
        value.state,
        value.handoff_phase,
        value.terminal_handoff,
        value.handoff_contract_digest,
        value.pending_operation_id,
        value.revision,
        value.updated_at,
    )


def _require_live_task_lease(node: TaskNodeView, lease: TaskLease, now: datetime) -> None:
    if (
        node.status is not TaskStatus.RUNNING
        or node.owner != lease.owner
        or node.fence != lease.fence
        or node.lease_expires_at is None
        or node.lease_expires_at <= now
    ):
        raise AIError(ErrorCode.TASK_FENCE_STALE)


def _validate_task_lease_scope(lease: TaskLease, tenant_id: str) -> None:
    if lease.tenant_id != tenant_id:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _require_live_tool_lease(
    current: ToolOperationRecord,
    *,
    owner: str,
    fence: int,
    now: datetime,
) -> None:
    if (
        current.status is not ToolOperationStatus.CLAIMED
        or current.owner != owner
        or current.fence != fence
        or current.lease_expires_at is None
        or current.lease_expires_at <= now
    ):
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)


def _task_graph_record(
    repository: _RepositoryBase, current: StoredRecord, value: TaskGraphView
) -> StoredRecord:
    return replace(
        repository._stored(
            "task_graph", value.graph_id, value,
            scope=current.scope_digest, state=value.status.value,
        ),
        storage_version=current.storage_version + 1,
    )


def _task_submit_result_digest(graph: TaskGraph) -> str:
    status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
    return canonical_sha256({"graph_id": graph.graph_id, "status": status.value})


def _stored_operation_error(operation: OperationLedgerRecord) -> AIError:
    if operation.error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(operation.error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:
    statuses = {node.status for node in nodes}
    if not statuses:
        return TaskStatus.SUCCEEDED
    if statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if TaskStatus.BLOCKED in statuses:
        return TaskStatus.BLOCKED
    if statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
        return TaskStatus.CANCELLED
    if TaskStatus.RUNNING in statuses:
        return TaskStatus.RUNNING
    return TaskStatus.PENDING


def _record_cursor(record: StoredRecord) -> str:
    payload = {
        "sort_key": record.sort_key,
        "key_digest": record.key_digest.hex(),
    }
    return base64.urlsafe_b64encode(canonical_json_bytes(payload)).decode("ascii")


def _decode_record_cursor(cursor: str | None) -> tuple[str | None, bytes | None]:
    if cursor is None:
        return None, None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value = json.loads(raw.decode("utf-8"))
        sort_key = str(value["sort_key"])
        key_digest = bytes.fromhex(str(value["key_digest"]))
        if not sort_key or len(key_digest) != 32:
            raise ValueError("cursor identity")
        return sort_key, key_digest
    except (TypeError, ValueError, KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


__all__ = [
    "ApprovalRepositoryImpl",
    "ArtifactRepositoryImpl",
    "EvaluationRepositoryImpl",
    "EventRepositoryImpl",
    "ExecutionRepositoryImpl",
    "ExternalCallRepositoryImpl",
    "IdempotencyRepositoryImpl",
    "MemoryRepositoryImpl",
    "OperationLedgerRepository",
    "RecoveryCheckpointRepositoryImpl",
    "SessionRepositoryImpl",
    "TaskAdmissionRepositoryImpl",
    "TaskRepositoryImpl",
    "ToolRepositoryImpl",
    "build_repository_bundle",
]
