#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral Runtime repositories built on the StateStore contract."""

import base64
import hashlib
import json
from collections.abc import Mapping
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
    operation_replay_matches,
)
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef
from ...task import TaskGraph, TaskGraphView, TaskLease, TaskNodeView, TaskTerminalRecord
from .._tool import ToolOperationRecord
from ._codec import decode_domain, decode_envelope, encode_domain, encode_envelope
from ._contracts import (
    ApprovalRecord,
    ArtifactRecord,
    ConversationCursor,
    EvaluationRecord,
    ExecutionCancelRequestCommit,
    ExecutionEventRecord,
    ExecutionRecord,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    IdempotencyRecord,
    MemoryRecord,
    RecoveryCheckpoint,
    ResultRecord,
    SessionRecord,
)
from ._plan import RuntimeDomain
from ._store import (
    FactQuery,
    OperationQuery,
    RecordQuery,
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
    subject_digest,
)

_logger = environ.get_logger("ai.runtime.state.repositories")
ValueT = TypeVar("ValueT")


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
        payload = _restore_lease_fields(_domain_payload(record), target)
        value = decode_domain(payload, target)
        if isinstance(value, (TaskNodeView, ToolOperationRecord)):
            return replace(
                value,
                owner=record.lease_owner,
                fence=record.lease_fence,
                lease_expires_at=record.lease_expires_at,
            )  # type: ignore[return-value]
        return value  # type: ignore[return-value]

    async def _replace_value(self, current: StoredRecord, value: object) -> None:
        candidate = _projected_record(self, current, value)
        await self._store.mutate(lambda transaction: _replace_checked(transaction, candidate, current.storage_version))

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
        current = await self._record(self._key(self._kind, identity))
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        value = await self._decode(current, self._value_type)
        if _domain_revision(value) != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._replace_value(current, next_record)
        return next_record

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

    async def _bump_list_generation(self, transaction: StateTransaction, owner_principal_id: str) -> None:
        await transaction.next_sequence(self._list_generation_key(owner_principal_id))
        await transaction.next_sequence(self._list_generation_key())

    async def create(self, value: SessionRecord) -> SessionRecord:
        _require_tenant(value, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            await transaction.insert_record(
                self._stored(
                    "session",
                    value.session_id,
                    value,
                    scope=self._scope("session", "owner", value.owner_principal_id),
                    state=value.status.value,
                )
            )
            await self._bump_list_generation(transaction, value.owner_principal_id)
            return value

        return await self._store.mutate(mutate)

    async def create_with_operation(
        self, record: SessionRecord, *, operation: OperationLedgerInput
    ) -> tuple[SessionRecord, bool]:
        _require_tenant(record, self._tenant_id)
        _require_tenant(operation, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> tuple[SessionRecord, bool]:
            _, replayed = await _append_operation(transaction, self, operation)
            if replayed:
                current = await transaction.get_record(self._key("session", record.session_id))
                if current is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return await self._decode(current, SessionRecord), True
            await transaction.insert_record(
                self._stored(
                    "session",
                    record.session_id,
                    record,
                    scope=self._scope("session", "owner", record.owner_principal_id),
                    state=record.status.value,
                )
            )
            await self._bump_list_generation(transaction, record.owner_principal_id)
            return record, False

        return await self._store.mutate(mutate)

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

    async def release_execution(self, session_id: str, *, tenant_id: str, execution_id: str) -> SessionRecord:
        return await self._admission(
            session_id, tenant_id=tenant_id, execution_id=execution_id, expected=None, release=True
        )

    async def _admission(
        self, session_id: str, *, tenant_id: str, execution_id: str, expected: ConversationCursor | None, release: bool
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
    ) -> SessionRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> SessionRecord:
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
        current = await super().get(identity, tenant_id=tenant_id)
        if current is None or current.status is not expected_status:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await _cas_value(self, identity, current, next_record)


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

    async def reserve_start(self, reservation: ExecutionStartReservation) -> ExecutionStartReservationResult:
        _require_tenant(reservation.execution, self._tenant_id)
        _require_tenant(reservation.idempotency, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionStartReservationResult:
            identity = self._idempotency._identity_key(
                reservation.idempotency.scope, reservation.idempotency.idempotency_key_digest
            )
            id_key = self._idempotency._key("idempotency", identity)
            execution_key = self._key("execution", reservation.execution.execution_id)
            existing_id = await transaction.get_record(id_key)
            existing_execution = await transaction.get_record(execution_key)
            if existing_id is not None or existing_execution is not None:
                if existing_id is None or existing_execution is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_value = await self._decode(existing_execution, ExecutionRecord)
                existing_idempotency = await self._idempotency._decode(existing_id, IdempotencyRecord)
                if not _same_idempotency(existing_idempotency, reservation.idempotency):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                if not _execution_replay_matches(existing_value, reservation.execution):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return ExecutionStartReservationResult(existing_value, existing_idempotency, False)
            await transaction.insert_record(
                self._stored(
                    "execution",
                    reservation.execution.execution_id,
                    reservation.execution,
                    state=reservation.execution.status.value,
                )
            )
            await transaction.insert_record(
                self._idempotency._stored(
                    "idempotency", identity, reservation.idempotency, state=reservation.idempotency.status.value
                )
            )
            return ExecutionStartReservationResult(reservation.execution, reservation.idempotency, True)

        return await self._store.mutate(mutate)

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        current = await self._required(claim.execution_id, claim.tenant_id)
        if (
            current.revision != claim.expected_revision
            or current.event_sequence != claim.expected_event_sequence
            or current.status is not ExecutionStatus.PENDING_START
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            status=ExecutionStatus.STARTED,
            revision=current.revision + 1,
            event_sequence=current.event_sequence + 1,
            updated_at=claim.started_at,
        )
        return await self._transition_with_fact(current, next_value, ExecutionEventType.EXECUTION_STARTED, {})

    async def claim_next_agent_run(
        self, execution_id: str, *, tenant_id: str, expected_revision: int, expected_agent_run_sequence: int
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            record = await transaction.get_record(self._key("execution", execution_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ExecutionRecord)
            if current.revision != expected_revision or current.agent_run_sequence != expected_agent_run_sequence:
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

        return await self._store.mutate(mutate)

    async def mark_start_unknown(self, commit: ExecutionStartUnknownCommit) -> ExecutionRecord:
        current = await self._required(commit.execution_id, commit.tenant_id)
        if current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            status=ExecutionStatus.START_UNKNOWN,
            revision=current.revision + 1,
            event_sequence=current.event_sequence + 1,
            updated_at=commit.occurred_at,
        )
        return await self._transition_with_fact(current, next_value, ExecutionEventType.EXECUTION_START_UNKNOWN, {})

    async def request_cancel(self, commit: ExecutionCancelRequestCommit) -> ExecutionRecord:
        current = await self._required(commit.execution_id, commit.tenant_id)
        if current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            status=ExecutionStatus.CANCELLING,
            revision=current.revision + 1,
            event_sequence=current.event_sequence + 1,
            updated_at=commit.requested_at,
        )
        return await self._transition_with_fact(
            current, next_value, ExecutionEventType.CANCEL_REQUESTED, {"operation_id": commit.operation_id}
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

    async def _required(self, execution_id: str, tenant_id: str) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        value = await self.get(execution_id, tenant_id=tenant_id)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return value

    async def _transition_with_fact(
        self,
        current: ExecutionRecord,
        next_value: ExecutionRecord,
        event_type: ExecutionEventType,
        payload: dict[str, object],
    ) -> ExecutionRecord:
        key = self._key("execution", current.execution_id)
        stream = stream_digest(self._namespace, self._tenant_id, self._domain.value, "execution", current.execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if stored_value.revision != current.revision or stored_value.event_sequence != current.event_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            candidate = _projected_record(self, stored, next_value)
            await _replace_checked(transaction, candidate, stored.storage_version)
            await transaction.insert_fact(
                StoredFact(
                    stream,
                    next_value.event_sequence,
                    key,
                    event_type.value,
                    subject_digest(current.execution_id),
                    None,
                    payload,
                )
            )
            return next_value

        return await self._store.mutate(mutate)

    async def commit_terminal(self, commit: ExecutionTerminalCommit) -> ExecutionTerminalCommitResult:
        current = await self._required(commit.execution.execution_id, commit.execution.tenant_id)
        if current.revision != commit.expected_revision or current.event_sequence != commit.expected_event_sequence:
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        key = self._key("execution", current.execution_id)
        stream = stream_digest(self._namespace, self._tenant_id, self._domain.value, "execution", current.execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionTerminalCommitResult:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != commit.expected_revision
                or stored_value.event_sequence != commit.expected_event_sequence
            ):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            next_execution = replace(
                commit.execution,
                revision=stored_value.revision + 1,
                event_sequence=stored_value.event_sequence + 1,
                result=commit.result,
            )
            candidate = _projected_record(self, stored, next_execution)
            await _replace_checked(transaction, candidate, stored.storage_version)
            await transaction.insert_fact(
                StoredFact(
                    stream,
                    next_execution.event_sequence,
                    key,
                    commit.terminal_event_type.value,
                    subject_digest(current.execution_id),
                    None,
                    commit.terminal_event_payload,
                )
            )
            if commit.idempotency is not None:
                identity = self._idempotency._identity_key(
                    commit.idempotency.scope,
                    commit.idempotency.idempotency_key_digest,
                )
                id_key = self._idempotency._key("idempotency", identity)
                id_record = await transaction.get_record(id_key)
                if id_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                id_value = await self._idempotency._decode(id_record, IdempotencyRecord)
                next_id = replace(
                    id_value,
                    status=commit.idempotency.next_status,
                    request_digest=commit.idempotency.request_digest,
                    result_digest=commit.idempotency.result_digest,
                    error_code=commit.idempotency.error_code,
                    updated_at=await transaction.now(),
                )
                await _replace_checked(
                    transaction,
                    _projected_record(self._idempotency, id_record, next_id),
                    id_record.storage_version,
                )
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
                next_operation = replace(
                    current_operation,
                    status=commit.operation.next_status,
                    result_ref=commit.operation.result_ref,
                    result_digest=commit.operation.result_digest,
                    error_code=commit.operation.error_code,
                    updated_at=await transaction.now(),
                )
                if not await transaction.replace_operation(
                    _stored_from_operation(next_operation, operation_record),
                    expected_state=commit.operation.expected_status.value,
                ):
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            return ExecutionTerminalCommitResult(next_execution, commit.result)

        return await self._store.mutate(mutate)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return None if execution is None else execution.result


class EventRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.EXECUTION)

    async def append(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_sequence: int,
        event_type: ExecutionEventType,
        payload: object,
    ) -> ExecutionEventRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = self._key("execution", execution_id)
        stream = stream_digest(self._namespace, self._tenant_id, self._domain.value, "execution", execution_id)

        async def mutate(transaction: StateTransaction) -> ExecutionEventRecord:
            current = await transaction.get_record(key)
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            execution = decode_domain(_domain_payload(current), ExecutionRecord)
            if execution.event_sequence != expected_sequence:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            next_execution = replace(
                execution,
                event_sequence=expected_sequence + 1,
                revision=execution.revision + 1,
                updated_at=await transaction.now(),
            )
            candidate = _projected_record(self, current, next_execution)
            await _replace_checked(transaction, candidate, current.storage_version)
            data = payload if isinstance(payload, dict) else {"value": payload}
            await transaction.insert_fact(
                StoredFact(
                    stream, expected_sequence + 1, key, event_type.value, subject_digest(execution_id), None, data
                )
            )
            return ExecutionEventRecord(execution_id, tenant_id, expected_sequence + 1, event_type, payload)

        return await self._store.mutate(mutate)

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
        current = await self.get(approval_id, tenant_id=tenant_id)
        if current is None or current.status is not expected_status:
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
        await self._replace_value(await self._required_record("approval", approval_id), value)
        return value

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "approval",
            scope=self._scope("approval", "execution", execution_id),
            states=frozenset({ApprovalStatus.PENDING.value}),
        )
        return tuple([await self._decode(record, ApprovalRecord) for record in records])

    async def _required_record(self, kind: str, identity: str) -> StoredRecord:
        record = await self._record(self._key(kind, identity))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return record


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
        current = await self.get(call_id, tenant_id=tenant_id)
        if current is None or current.status is not expected_status:
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        value = replace(
            current,
            status=ExternalCallStatus.SUPPLIED,
            idempotency_key_digest=idempotency_key_digest,
            object_ref=object_ref,
            payload_digest=payload_digest,
            supplied_at=supplied_at,
        )
        await self._replace_value(await self._required_record("external_call", call_id), value)
        return value

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ExternalCallRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "external_call",
            scope=self._scope("external_call", "execution", execution_id),
            states=frozenset({ExternalCallStatus.PENDING.value}),
        )
        return tuple([await self._decode(record, ExternalCallRecord) for record in records])

    async def _required_record(self, kind: str, identity: str) -> StoredRecord:
        record = await self._record(self._key(kind, identity))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return record


class RecoveryCheckpointRepositoryImpl(_ResourceRepository[RecoveryCheckpoint]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
            kind="recovery_checkpoint",
            resource_kind=ResourceKind.EXECUTION,
            value_type=RecoveryCheckpoint,
            identity_field="execution_id",
        )

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        if tenant_id != self._tenant_id:
            return ()
        return await self.list_values()


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
            await transaction.insert_record(self._stored("task_graph", graph.graph_id, view, state=view.status.value))
            for node in graph.nodes:
                status = TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
                node_view = TaskNodeView(
                    graph.graph_id, node.node_id, node.dependencies, status, None, 0, None, None, None, None
                )
                await transaction.insert_record(
                    self._stored(
                        "task_node",
                        [graph.graph_id, node.node_id],
                        node_view,
                        parent=self._parent("task_node", "graph", graph.graph_id),
                        state=status.value,
                    )
                )
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
                RecordQuery(parent_digest=self._parent("task_node", "graph", graph_id))
            )
            current_nodes = {node.node_id: node for node in await self._decode_many(node_records)}
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
            if not changed_nodes:
                return TaskGraphView(graph.graph_id, _graph_status(tuple(next_nodes.values())), graph.nodes)
            next_graph = replace(graph, status=_graph_status(tuple(next_nodes.values())))
            await _replace_checked(
                transaction,
                _projected_record(self, graph_record, next_graph),
                graph_record.storage_version,
            )
            for current, value in changed_nodes:
                node_record = await transaction.get_record(self._node_key(graph_id, current.node_id))
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                stored_node = await self._decode(node_record, TaskNodeView)
                if stored_node != current:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await _replace_checked(
                    transaction,
                    _projected_record(self, node_record, value),
                    node_record.storage_version,
                )
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
                RecordQuery(parent_digest=self._parent("task_node", "graph", graph_id))
            )
            nodes = await self._decode_many(node_records)
            changed = [
                (node, replace(node, status=TaskStatus.CANCELLED, owner=None, lease_expires_at=None))
                for node in nodes
                if node.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            ]
            if not changed:
                return TaskGraphView(graph.graph_id, _graph_status(tuple(nodes)), graph.nodes)
            changed_ids = {current.node_id for current, _ in changed}
            next_nodes = tuple(value for _, value in changed) + tuple(
                node for node in nodes if node.node_id not in changed_ids
            )
            next_graph = replace(graph, status=_graph_status(next_nodes))
            await _replace_checked(
                transaction,
                _projected_record(self, graph_record, next_graph),
                graph_record.storage_version,
            )
            for current, value in changed:
                node_record = await transaction.get_record(self._node_key(graph_id, current.node_id))
                if node_record is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await _replace_checked(
                    transaction,
                    _projected_record(self, node_record, value),
                    node_record.storage_version,
                )
            _logger.info("task graph cancelled atomically: graph_id=%s changed_nodes=%s", graph_id, len(changed))
            return TaskGraphView(graph.graph_id, next_graph.status, graph.nodes)

        return await self._store.mutate(mutate)

    async def claim(self, graph_id: str, node_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> TaskLease:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskLease:
            record = await transaction.get_record(self._node_key(graph_id, node_id))
            if record is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            node = await self._decode(record, TaskNodeView)
            now = await transaction.now()
            expired = node.lease_expires_at is not None and node.lease_expires_at <= now
            if node.status is TaskStatus.RUNNING and node.owner not in {None, owner} and not expired:
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT)
            if node.status is not TaskStatus.READY and not (node.status is TaskStatus.RUNNING and expired):
                raise AIError(ErrorCode.TASK_NOT_READY)
            fence = node.fence + 1
            expires = now + timedelta(seconds=lease_seconds)
            value = replace(node, status=TaskStatus.RUNNING, owner=owner, fence=fence, lease_expires_at=expires)
            await self._update_node_in_transaction(transaction, node, value)
            return TaskLease(graph_id, node_id, tenant_id, owner, fence, expires)

        return await self._store.mutate(mutate)

    async def renew(self, lease: TaskLease, *, tenant_id: str, lease_seconds: int) -> TaskLease:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> TaskLease:
            graph_record = await transaction.get_record(self._graph_key(lease.graph_id))
            node_record = await transaction.get_record(self._node_key(lease.graph_id, lease.node_id))
            if graph_record is None or node_record is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            node = await self._decode(node_record, TaskNodeView)
            now = await transaction.now()
            _require_live_task_lease(node, lease, now)
            expires = now + timedelta(seconds=lease_seconds)
            if await transaction.guard_record(
                graph_record.key_digest,
                expected_storage_version=graph_record.storage_version,
            ) is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
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
        node_record = await transaction.get_record(self._node_key(current.graph_id, current.node_id))
        if graph_record is None or node_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        stored_node = await self._decode(node_record, TaskNodeView)
        if stored_node != current:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        graph = await self._decode(graph_record, TaskGraphView)
        node_records = await transaction.list_records(
            RecordQuery(parent_digest=self._parent("task_node", "graph", current.graph_id))
        )
        node_values = [await self._decode(item, TaskNodeView) for item in node_records]
        node_values = [value if item.node_id == current.node_id else item for item in node_values]
        graph_value = replace(graph, status=_graph_status(tuple(node_values)))
        graph_candidate = _projected_record(self, graph_record, graph_value)
        node_candidate = _projected_record(self, node_record, value)
        await _replace_checked(transaction, graph_candidate, graph_record.storage_version)
        await _replace_checked(transaction, node_candidate, node_record.storage_version)


class ToolRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.RECOVERY)

    def _tool_key(self, identity: str) -> bytes:
        return self._key("tool_operation", identity)

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
                    await transaction.insert_alias(StoredAlias(replay_alias, existing_key))
                return existing
            stored = self._stored(
                "tool_operation",
                record.tool_operation_id,
                record,
                scope=self._scope("tool_operation", "step_run", record.step_run_id),
                state=record.status.value,
            )
            await transaction.insert_record(stored)
            await transaction.insert_alias(StoredAlias(replay_alias, key))
            return record

        return await self._store.mutate(mutate)

    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> ToolOperationRecord | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._tool_key(tool_operation_id))
        return None if record is None else await self._decode(record, ToolOperationRecord)

    async def claim(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

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
                return current
            now = await transaction.now()
            expired = current.lease_expires_at is not None and current.lease_expires_at <= now
            if current.status is ToolOperationStatus.CLAIMED and expired and not current.replay_safe:
                unknown = replace(
                    current, status=ToolOperationStatus.EFFECT_UNKNOWN, lease_expires_at=None, updated_at=now
                )
                await self._replace_tool_in_transaction(transaction, record, unknown)
                return unknown
            if current.owner not in {None, owner} and not expired:
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

    async def complete(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, result_object_ref: ObjectRef | None
    ) -> ToolOperationRecord:
        return await self._finish_tool(
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            terminal_status=ToolOperationStatus.COMPLETED,
            requested_result=result_object_ref,
            value=lambda current, now: replace(
                current,
                status=ToolOperationStatus.COMPLETED,
                result_object_ref=result_object_ref,
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

    async def _required(self, identity: str, tenant_id: str) -> ToolOperationRecord:
        value = await self.get_operation(identity, tenant_id=tenant_id)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return value

    async def _finish_tool(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        terminal_status: ToolOperationStatus,
        requested_result: ObjectRef | None = None,
        requested_error: str | None = None,
        value: object,
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def mutate(transaction: StateTransaction) -> ToolOperationRecord:
            record = await transaction.get_record(self._tool_key(tool_operation_id))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ToolOperationRecord)
            now = await transaction.now()
            if current.status is ToolOperationStatus.COMPLETED:
                if terminal_status is ToolOperationStatus.COMPLETED and current.result_object_ref == requested_result:
                    return current
                if terminal_status is ToolOperationStatus.COMPLETED:
                    raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is ToolOperationStatus.FAILED:
                if terminal_status is ToolOperationStatus.FAILED and current.error_code == requested_error:
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

        return await self._store.mutate(mutate)

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
        values["sessions"] = SessionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
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


def _domain_data(value: object) -> dict[str, object]:
    payload = encode_domain(value)
    if isinstance(value, (TaskNodeView, ToolOperationRecord)) and isinstance(payload, Mapping):
        fields = payload.get("fields")
        if isinstance(fields, Mapping):
            payload = dict(payload)
            payload["fields"] = {
                key: item for key, item in fields.items() if key not in {"owner", "fence", "lease_expires_at"}
            }
    return encode_envelope({"type": value.__class__.__name__, "payload": payload})


def _domain_payload(record: StoredRecord) -> object:
    payload = decode_envelope(record.data).get("payload")
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return payload


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
    identity = _canonical_record_identity(current.kind, value)
    projected = repository._stored(current.kind, identity, value, state=_record_state(value))
    return replace(projected, storage_version=current.storage_version + 1)


async def _cas_value(repository: _ResourceRepository[ValueT], identity: str, current: ValueT, value: ValueT) -> ValueT:
    record = await repository._record(repository._key(repository._kind, identity))
    if record is None:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
    await repository._replace_value(record, value)
    return value


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
    payload = decode_envelope(value.data).get("payload")
    if payload is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    candidate = decode_domain(payload, OperationLedgerInput)
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
        left.tenant_id == right.tenant_id
        and left.runtime_domain is right.runtime_domain
        and left.scope == right.scope
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.request_digest == right.request_digest
        and left.resource_kind is right.resource_kind
        and left.resource_id == right.resource_id
    )


def _tool_replay_matches(left: ToolOperationRecord, right: ToolOperationRecord) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.step_run_id == right.step_run_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and left.binding_fingerprint == right.binding_fingerprint
        and left.replay_safe == right.replay_safe
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


def _require_live_task_lease(node: TaskNodeView, lease: TaskLease, now: datetime) -> None:
    if (
        node.status is not TaskStatus.RUNNING
        or node.owner != lease.owner
        or node.fence != lease.fence
        or node.lease_expires_at is None
        or node.lease_expires_at <= now
    ):
        raise AIError(ErrorCode.TASK_FENCE_STALE)


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


def _graph_status(nodes: tuple[TaskNodeView, ...]) -> TaskStatus:
    statuses = {node.status for node in nodes}
    if statuses and statuses <= {TaskStatus.SUCCEEDED}:
        return TaskStatus.SUCCEEDED
    if TaskStatus.FAILED in statuses:
        return TaskStatus.FAILED
    if statuses and statuses <= {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED}:
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
    "EventRepositoryImpl",
    "EvaluationRepositoryImpl",
    "ExecutionRepositoryImpl",
    "ExternalCallRepositoryImpl",
    "IdempotencyRepositoryImpl",
    "MemoryRepositoryImpl",
    "OperationLedgerRepository",
    "RecoveryCheckpointRepositoryImpl",
    "SessionRepositoryImpl",
    "TaskRepositoryImpl",
    "ToolRepositoryImpl",
    "build_repository_bundle",
]
