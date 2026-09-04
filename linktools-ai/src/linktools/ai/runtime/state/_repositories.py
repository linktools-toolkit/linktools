#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral Runtime repositories built on the StateStore contract."""

import asyncio
import base64
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    TaskGraphView,
    TaskNodeView,
    TaskResultRecord,
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
from ._history_index import build_fork_index_node_from_roots
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
    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        domain: RuntimeDomain,
    ) -> None:
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
        return partition_digest(
            self._namespace, self._tenant_id, self._domain.value, kind
        )

    def _key(self, kind: str, identity: object) -> bytes:
        return record_key_digest(
            self._namespace, self._tenant_id, self._domain.value, kind, identity
        )

    def _scope(self, kind: str, relation: str, identity: object) -> bytes:
        return scope_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            kind,
            relation,
            identity,
        )

    def _parent(self, kind: str, relation: str, identity: object) -> bytes:
        return parent_digest(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            kind,
            relation,
            identity,
        )

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
        if isinstance(
            value,
            (EvaluationRecord, ArtifactRecord, ApprovalRecord, ExternalCallRecord),
        ):
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
                    partition_digest=(
                        self._partition(kind)
                        if scope is None and parent is None
                        else None
                    ),
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
        await self._insert(
            self._stored(self._kind, identity, value, state=_record_state(value))
        )
        _logger.debug("created Runtime record: kind=%s id=%s", self._kind, identity)
        return value

    async def get(self, identity: str, *, tenant_id: str) -> ValueT | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key(self._kind, identity))
        return None if record is None else await self._decode(record, self._value_type)

    async def get_header(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        value = await self.get(identity, tenant_id=tenant_id)
        return (
            None
            if value is None
            else self._header(value, self._resource_kind, identity)
        )

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

    def _stored_operation(
        self, value: OperationLedgerInput, sequence: int
    ) -> StoredOperation:
        return StoredOperation(
            operation_key(
                self._namespace, self._tenant_id, self._domain.value, value.operation_id
            ),
            self._stream(value),
            sequence,
            value.status.value,
            value.compactable,
            _domain_data(value),
        )

    async def append(self, value: OperationLedgerInput) -> OperationLedgerRecord:
        _require_tenant(value, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> OperationLedgerRecord:
            key = operation_key(
                self._namespace, self._tenant_id, self._domain.value, value.operation_id
            )
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

    async def get(
        self, operation_id: str, *, tenant_id: str
    ) -> OperationLedgerRecord | None:
        if tenant_id != self._tenant_id:
            return None
        key = operation_key(
            self._namespace, self._tenant_id, self._domain.value, operation_id
        )
        stored = await self._store.read(
            lambda transaction: transaction.get_operation(key)
        )
        return None if stored is None else _decode_operation(stored)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> OperationLedgerRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = operation_key(
            self._namespace, self._tenant_id, self._domain.value, operation_id
        )
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
        key = operation_key(
            self._namespace, self._tenant_id, self._domain.value, operation_id
        )

        async def mutate(transaction: StateTransaction) -> OperationLedgerRecord:
            current = await transaction.get_operation(key)
            if current is None or current.state != expected_status.value:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if (
                current.state
                in {
                    OperationStatus.SUCCEEDED.value,
                    OperationStatus.FAILED.value,
                    OperationStatus.CANCELLED.value,
                }
                and next_record.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            candidate = _stored_from_operation(next_record, current)
            if not await transaction.replace_operation(
                candidate, expected_state=expected_status.value
            ):
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
                OperationQuery(
                    stream_digest=stream,
                    states=frozenset({"PENDING", "RUNNING"}),
                    limit=limit,
                )
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

    async def create(
        self, record: ConversationHistoryRecord
    ) -> ConversationHistoryRecord:
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
        record = await transaction.get_record(
            self._key("conversation_history", history_id)
        )
        return None if record is None else await self._decode_history(record)

    async def local_head_in_transaction(
        self,
        transaction: StateTransaction,
        history_id: str,
    ) -> tuple[int, int]:
        """Read one branch's local message and history-item counts."""
        record = await transaction.get_record(self._key("transcript_head", history_id))
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
                current = await transaction.get_record(
                    self._key("session", record.session_id)
                )
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self._decode(current, SessionRecord), True
            await transaction.insert_records(
                (
                    self._stored(
                        "session",
                        record.session_id,
                        record,
                        scope=self._scope(
                            "session", "owner", record.owner_principal_id
                        ),
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
            if (
                await transaction.guard_record(
                    self._key("session", source_session_id),
                    expected_storage_version=source_stored.storage_version,
                )
                is None
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            child_history_id = target.history_id
            if child_history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_history_key = self._key("conversation_history", source.history_id)
            target_key = self._key("session", target.session_id)
            child_history_key = self._key("conversation_history", child_history_id)
            source_head_key = self._key("transcript_head", source.history_id)
            related = await transaction.get_records(
                (
                    source_history_key,
                    target_key,
                    child_history_key,
                    source_head_key,
                )
            )
            source_history_stored = related.get(source_history_key)
            if source_history_stored is None:
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            source_history = await self._decode_history(source_history_stored)
            if (
                source_history.session_id != source.session_id
                or source_history.tenant_id != self._tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            target_stored = related.get(target_key)
            child_stored = related.get(child_history_key)
            source_head_stored = related.get(source_head_key)
            if source_head_stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source_head = _decode_enveloped_domain(
                source_head_stored.data,
                TranscriptHeadRecord,
            )
            local_messages = source_head.message_count
            local_items = source_head.session_history_item_count
            histories = ConversationHistoryRepositoryImpl(
                self._store,
                namespace=self._namespace,
                tenant_id=self._tenant_id,
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
            inherited_items = source_history.inherited_history_item_count + local_items
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
            await transaction.insert_records(
                (
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
                    ),
                    self._stored(
                        "conversation_history",
                        child.history_id,
                        child,
                    ),
                    self._stored(
                        "transcript_head",
                        child.history_id,
                        _empty_conversation_transcript_head(child.history_id),
                    ),
                    self._stored(
                        "session_fork_result",
                        operation.operation_id,
                        fork_result,
                    ),
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
        target_key = self._key("session", result.target_session_id)
        child_key = self._key("conversation_history", result.target_history_id)
        head_key = self._key("transcript_head", result.target_history_id)
        related = await transaction.get_records((target_key, child_key, head_key))
        target_stored = related.get(target_key)
        child_stored = related.get(child_key)
        head_stored = related.get(head_key)
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

    async def list(
        self, *, tenant_id: str, owner_principal_id: str | None = None
    ) -> tuple[SessionRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        scope = (
            None
            if owner_principal_id is None
            else self._scope("session", "owner", owner_principal_id)
        )

        async def read(transaction: StateTransaction) -> tuple[SessionRecord, ...]:
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("session") if scope is None else None,
                    scope_digest=scope,
                    kind="session",
                )
            )
            return tuple(
                [await self._decode(record, SessionRecord) for record in records]
            )

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

        async def read(
            transaction: StateTransaction,
        ) -> tuple[int, Page[SessionRecord]]:
            generation = await transaction.get_sequence(
                self._list_generation_key(owner_principal_id)
            )
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
            values = tuple(
                [await self._decode(record, SessionRecord) for record in records[:limit]]
            )
            next_cursor = (
                _record_cursor(records[limit - 1]) if len(records) > limit else None
            )
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
            if (
                value.active_execution_id is not None
                and proposed.active_execution_id != value.active_execution_id
            ):
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
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
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
            if (
                current.active_execution_id is not None
                and proposed.active_execution_id != current.active_execution_id
            ):
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
        self,
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

    async def release_execution(
        self, session_id: str, *, tenant_id: str, execution_id: str
    ) -> SessionRecord:
        return await self._admission(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=None,
            release=True,
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
            history_quality=current.history_quality if history_quality is None else history_quality,
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


# The remaining repository implementations are unchanged from the baseline.
