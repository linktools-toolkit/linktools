#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared backend-neutral repository primitives."""

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Generic, TypeVar
from linktools.core import environ
from ...core import OperationLedgerInput, OperationLedgerRecord, OperationStatus, ResourceKind, ResourceRef, canonical_json_bytes, operation_replay_matches
from ...errors import AIError, ErrorCode
from ...task import TaskGraphView, TaskNodeView
from ._contracts import ToolOperationRecord
from ._codec import _decode_enveloped_domain, _encode_persisted_domain, encode_envelope, wire_type_id
from ._contracts import ApprovalRecord, ArtifactRecord, ConversationHistoryRecord, EvaluationRecord, ExecutionRecord, ExternalCallRecord, IdempotencyRecord, MemoryRecord, RecoveryCheckpoint, SessionRecord
from ._plan import RuntimeDomain
from ._store import OperationQuery, RecordQuery, StateStore, StateTransaction, StoredOperation, StoredRecord, operation_key, parent_digest, partition_digest, record_key_digest, scope_digest, sequence_key, sortable_identity, stream_digest

_logger = environ.get_logger("ai.runtime.state.repositories")

ValueT = TypeVar("ValueT")


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
        if isinstance(value, ToolOperationRecord):
            return self._parent(kind, "execution", value.execution_id)
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
        sort_key_prefix: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[StoredRecord, ...]:
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1001
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)
        query_limit = None if limit is None else min(limit, 1000)

        async def read(transaction: StateTransaction) -> tuple[StoredRecord, ...]:
            records = await transaction.list_records(
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
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=query_limit,
                )
            )
            if limit != 1001 or len(records) != 1000:
                return records
            last = records[-1]
            probe = await transaction.list_records(
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
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=last.sort_key,
                    after_key_digest=last.key_digest,
                    limit=1,
                )
            )
            if probe and (
                probe[0].sort_key,
                probe[0].key_digest,
            ) > (last.sort_key, last.key_digest):
                return (*records, probe[0])
            return records

        return await self._store.read(read)

    async def _has_records(
        self,
        kind: str,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        states: frozenset[str] | None = None,
        sort_key_prefix: str | None = None,
        cursor: str | None = None,
    ) -> bool:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> bool:
            records = await transaction.list_records(
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
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=1,
                )
            )
            return bool(records)

        return await self._store.read(read)

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
    ) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
        self._kind = kind
        self._resource_kind = resource_kind
        self._value_type = value_type

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

    async def create_in_transaction(
        self,
        transaction: StateTransaction,
        value: ValueT,
    ) -> ValueT:
        _require_tenant(value, self._tenant_id)
        identity = self._identity(value)
        key = self._key(self._kind, identity)
        current = await transaction.get_record(key)
        if current is not None:
            existing = await self._decode(current, self._value_type)
            if existing != value:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return existing
        await transaction.insert_record(
            self._stored(self._kind, identity, value, state=_record_state(value))
        )
        _logger.debug(
            "created Runtime record in group: kind=%s id=%s",
            self._kind,
            identity,
        )
        return value

    async def get(self, identity: str, *, tenant_id: str) -> ValueT | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key(self._kind, identity))
        return None if record is None else await self._decode(record, self._value_type)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        identity: str,
        *,
        tenant_id: str,
    ) -> ValueT | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key(self._kind, identity))
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
            return await self.append_in_transaction(transaction, value)

        return await self._store.mutate(mutate)

    async def append_in_transaction(
        self,
        transaction: StateTransaction,
        value: OperationLedgerInput,
    ) -> OperationLedgerRecord:
        _require_tenant(value, self._tenant_id)
        key = operation_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            value.operation_id,
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
    payload = _encode_persisted_domain(value)
    if isinstance(value, (TaskNodeView, ToolOperationRecord)) and isinstance(
        payload, Mapping
    ):
        fields = payload.get("fields")
        if isinstance(fields, Mapping):
            payload = dict(payload)
            payload["fields"] = {
                key: item
                for key, item in fields.items()
                if key not in {"owner", "fence", "lease_expires_at"}
            }
    return encode_envelope({"type": wire_type_id(value), "payload": payload})


def _restore_lease_fields(payload: object, target: type[ValueT]) -> object:
    if target not in {TaskNodeView, ToolOperationRecord} or not isinstance(
        payload, Mapping
    ):
        return payload
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        return payload
    restored = dict(payload)
    restored["fields"] = {
        **fields,
        "owner": None,
        "fence": 1,
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
        value,
        (
            SessionRecord,
            ExecutionRecord,
            IdempotencyRecord,
            MemoryRecord,
            EvaluationRecord,
            RecoveryCheckpoint,
        ),
    ):
        return value.revision
    return 0


def _status_value(value: object) -> str | None:
    if isinstance(value, RecoveryCheckpoint):
        return value.state.value
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


async def _replace_checked(
    transaction: StateTransaction, candidate: StoredRecord, expected: int
) -> None:
    if not await transaction.replace_record(
        candidate, expected_storage_version=expected
    ):
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _projected_record(
    repository: _RepositoryBase,
    current: StoredRecord,
    value: object,
) -> StoredRecord:
    _require_tenant(value, repository._tenant_id)
    if current.kind == "session" and isinstance(value, SessionRecord):
        _require_session_identity(
            _decode_enveloped_domain(current.data, SessionRecord),
            value,
        )
    identity = _canonical_record_identity(current.kind, value)
    projected = repository._stored(
        current.kind, identity, value, state=_record_state(value)
    )
    return replace(projected, storage_version=current.storage_version + 1)


def _require_session_identity(
    current: SessionRecord,
    candidate: SessionRecord,
) -> None:
    if (
        candidate.session_id,
        candidate.tenant_id,
        candidate.owner_principal_id,
        candidate.agent_id,
        candidate.history_id,
        candidate.created_at,
    ) != (
        current.session_id,
        current.tenant_id,
        current.owner_principal_id,
        current.agent_id,
        current.history_id,
        current.created_at,
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _append_operation(
    transaction: StateTransaction,
    repository: _RepositoryBase,
    value: OperationLedgerInput,
) -> tuple[OperationLedgerRecord, bool]:
    operation, replayed = await _reserve_operation(transaction, repository, value)
    if operation is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if replayed:
        return operation, True
    await _insert_operation(
        transaction,
        repository,
        value,
        operation.sequence,
    )
    return operation, False


async def _reserve_operation(
    transaction: StateTransaction,
    repository: _RepositoryBase,
    value: OperationLedgerInput,
) -> tuple[OperationLedgerRecord | None, bool]:
    key = operation_key(
        repository._namespace,
        repository._tenant_id,
        repository._domain.value,
        value.operation_id,
    )
    existing = await transaction.get_operation(key)
    if existing is not None:
        current = _decode_operation(existing)
        if _operation_matches(current, value):
            return current, True
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    sequence = await transaction.next_sequence(
        sequence_key(
            repository._namespace,
            repository._tenant_id,
            repository._domain.value,
            "operation",
            [value.resource_kind.value, value.resource_id],
        )
    )
    return _operation_record(value, sequence), False


async def _insert_operation(
    transaction: StateTransaction,
    repository: _RepositoryBase,
    value: OperationLedgerInput,
    sequence: int,
) -> None:
    key = operation_key(
        repository._namespace,
        repository._tenant_id,
        repository._domain.value,
        value.operation_id,
    )
    await transaction.insert_operation(
        StoredOperation(
            key,
            stream_digest(
                repository._namespace,
                repository._tenant_id,
                repository._domain.value,
                "operation",
                [value.resource_kind.value, value.resource_id],
            ),
            sequence,
            value.status.value,
            value.compactable,
            _domain_data(value),
        )
    )


def _decode_operation(value: StoredOperation) -> OperationLedgerRecord:
    candidate = _decode_enveloped_domain(value.data, OperationLedgerInput)
    return _operation_record(candidate, value.sequence)  # type: ignore[arg-type]


def _validate_page_limit(limit: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 1000
    ):
        raise AIError(ErrorCode.PAGE_LIMIT_INVALID)


def _operation_record(
    value: OperationLedgerInput, sequence: int
) -> OperationLedgerRecord:
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


def _stored_from_operation(
    value: OperationLedgerRecord, current: StoredOperation
) -> StoredOperation:
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
    return replace(
        current,
        state=value.status.value,
        compactable=value.compactable,
        data=_domain_data(candidate),
    )


def _operation_matches(
    current: OperationLedgerRecord, candidate: OperationLedgerInput
) -> bool:
    return operation_replay_matches(current, candidate)


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
    except (
        TypeError,
        ValueError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


RepositoryBase = _RepositoryBase
ResourceRepository = _ResourceRepository
append_operation = _append_operation
decode_operation = _decode_operation
decode_record_cursor = _decode_record_cursor
domain_data = _domain_data
insert_operation = _insert_operation
projected_record = _projected_record
record_cursor = _record_cursor
record_state = _record_state
replace_checked = _replace_checked
require_repository_tenant = _require_repository_tenant
require_tenant = _require_tenant
reserve_operation = _reserve_operation
stored_from_operation = _stored_from_operation
validate_page_limit = _validate_page_limit


__all__ = [
    "OperationLedgerRepository",
    "RepositoryBase",
    "ResourceRepository",
    "append_operation",
    "decode_operation",
    "decode_record_cursor",
    "domain_data",
    "insert_operation",
    "projected_record",
    "record_cursor",
    "record_state",
    "replace_checked",
    "require_repository_tenant",
    "require_tenant",
    "reserve_operation",
    "stored_from_operation",
    "validate_page_limit",
]
