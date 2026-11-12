#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared backend-neutral repository primitives."""

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Generic, TypeVar
from linktools.core import environ
from ..core import (
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    ResourceRef,
    operation_cas_immutable_matches,
    operation_replay_matches,
)
from ..errors import AIError, ErrorCode
from ..evaluation import EvidenceBundle, EvaluationReport, ComparisonReport
from ..task import TaskGraphView, TaskNodeView
from .contracts import ToolOperationRecord
from ._message_root import NativeTextReference
from .codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
    wire_type_id,
)
from .contracts import (
    ApprovalRecord,
    ArtifactRecord,
    EvaluationRecord,
    ExecutionRecord,
    RecoveryHandoffPhase,
    ExternalCallRecord,
    IdempotencyRecord,
    MemoryRecord,
    SessionRecord,
)
from ._execution_content import ExecutionContent
from ._evaluation_records import (
    EvaluationCaseRecord,
    EvaluationDatasetRecord,
    EvaluationTombstone,
    EvaluationContentTombstone,
    EvaluationCleanupRecord,
)
from ._plan import PersistenceDomain
from .state import (
    OperationQuery,
    RECORD_KEY_MAX_BYTES,
    validate_record_key,
    validate_sort_key,
    RecordQuery,
    StateStore,
    StateTransaction,
    StoredOperation,
    StoredRecord,
    operation_key,
    parent_key,
    record_key,
    scope_key,
    stream_key,
)

_logger = environ.get_logger("ai.runtime.state.repositories")

ValueT = TypeVar("ValueT")


class _RepositoryBase:
    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        domain: PersistenceDomain,
    ) -> None:
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._domain = domain
        self._execution_content = ExecutionContent(store, namespace=namespace, tenant_id=tenant_id)

    async def initialize(self) -> None:
        return None

    async def initialize_content(self, *, create: bool, deferred: bool = False) -> None:
        """Opt into shared binding writes before entering repository transactions.

        Standalone repositories otherwise retain complete inline bindings. A
        low-level callback reading shared Runtime data must also call this
        operation before entry; callbacks cannot enlist new archive stores.
        """
        if self._domain is PersistenceDomain.EXECUTION:
            if deferred:
                if create:
                    self._execution_content.enable_sharing()
            else:
                await self._execution_content.initialize(create=create)

    async def close(self) -> None:
        return None

    @property
    def state_store(self) -> StateStore:
        return self._store

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def namespace(self) -> str:
        return self._namespace

    def _key(self, kind: str, identity: Sequence[str]) -> bytes:
        return record_key(
            self._namespace, self._tenant_id, self._domain.value, kind, identity
        )

    def _scope(self, kind: str, relation: str, identity: Sequence[str]) -> bytes:
        return scope_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            kind,
            relation,
            identity,
        )

    def _parent(self, kind: str, relation: str, identity: Sequence[str]) -> bytes:
        return parent_key(
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
        identity: Sequence[str],
        value: object,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        state: str | None = None,
    ) -> StoredRecord:
        return project_record(
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain,
            kind=kind,
            identity=identity,
            value=value,
            scope=scope,
            parent=parent,
            state=state,
        )

    async def _stored_content(
        self, kind: str, identity: Sequence[str], value: object, *, scope: bytes | None = None,
        parent: bytes | None = None, state: str | None = None,
        previous_value: ExecutionRecord | None = None,
    ) -> StoredRecord:
        if isinstance(value, ExecutionRecord):
            previous = None if previous_value is None else self._execution_content.projection(previous_value)
            encoded = await self._execution_content.encode(value, previous=previous,
                observed_value=previous_value if previous is not None else None)
            record = project_record(
                namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain,
                kind=kind, identity=identity, value=value, scope=scope, parent=parent,
                state=state, encoded=encoded,
            )
            self._execution_content.remember_projection(value, record)
            return record
        return self._stored(kind, identity, value, scope=scope, parent=parent, state=state)

    async def _projected_content(self, current: StoredRecord, value: object, *, result_source: NativeTextReference | None = None) -> StoredRecord:
        if isinstance(value, ExecutionRecord):
            _require_tenant(value, self._tenant_id)
            encoded = await self._execution_content.encode(value, previous=current, result_source=result_source)
            record = project_record(namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain,
                kind=current.kind, identity=(value.execution_id,), value=value, state=_record_state(value),
                storage_revision=current.storage_revision + 1, encoded=encoded)
            self._execution_content.remember_projection(value, record)
            return record
        return _projected_record(self, current, value)

    async def _record(self, key: bytes) -> StoredRecord | None:
        return await self._store.read(lambda transaction: transaction.get_record(key))

    async def _records(
        self,
        kind: str,
        *,
        scope: bytes | None = None,
        parent: bytes | None = None,
        states: frozenset[str] | None = None,
        sort_key_prefix: bytes | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[StoredRecord, ...]:
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1001
        ):
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        after_sort_key, after_key = _decode_record_cursor(cursor)
        query_limit = None if limit is None else min(limit, 1000)

        async def read(transaction: StateTransaction) -> tuple[StoredRecord, ...]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_key=scope,
                    parent_key=parent,
                    kind=kind,
                    states=states,
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=after_sort_key,
                    after_key=after_key,
                    limit=query_limit,
                )
            )
            if limit != 1001 or len(records) != 1000:
                return records
            last = records[-1]
            probe = await transaction.list_records(
                RecordQuery(
                    scope_key=scope,
                    parent_key=parent,
                    kind=kind,
                    states=states,
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=last.sort_key,
                    after_key=last.key,
                    limit=1,
                )
            )
            if probe and (
                probe[0].sort_key,
                probe[0].key,
            ) > (last.sort_key, last.key):
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
        sort_key_prefix: bytes | None = None,
        cursor: str | None = None,
    ) -> bool:
        after_sort_key, after_key = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> bool:
            records = await transaction.list_records(
                RecordQuery(
                    scope_key=scope,
                    parent_key=parent,
                    kind=kind,
                    states=states,
                    sort_key_prefix=sort_key_prefix,
                    after_sort_key=after_sort_key,
                    after_key=after_key,
                    limit=1,
                )
            )
            return bool(records)

        return await self._store.read(read)

    async def _refresh_records(self, keys: Sequence[bytes]) -> Mapping[bytes, StoredRecord]:
        # Native stores can repair their ordinary cache after a source fallback.
        # Custom stores retain the existing mandatory StateStore contract.
        from ._memory import InMemoryStateStore
        from ._sql import SqlStateStore

        if isinstance(self._store, (InMemoryStateStore, SqlStateStore)):
            return await self._store.refresh_records(keys)
        return await self._store.read(lambda transaction: transaction.get_records(keys))

    async def _insert(self, record: StoredRecord) -> None:
        async def insert(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
        await self._store.mutate(insert)

    async def _decode(self, record: StoredRecord, target: type[ValueT]) -> ValueT:
        if target is ExecutionRecord:
            return await self._execution_content.decode(record)
        value = _decode_enveloped_domain(
            record.data,
            target,
            payload_transform=lambda payload: _restore_lease_fields(payload, target),
        )
        if isinstance(value, ToolOperationRecord):
            if (record.subject_id != value.tool_call_id
                    or record.key != self._key("tool_operation", (value.tool_operation_id,))):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replace(value, owner=record.lease_owner, fence=record.lease_fence)
        if isinstance(value, TaskNodeView):
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
        domain: PersistenceDomain,
        kind: str,
        resource_kind: ResourceKind,
        value_type: type[ValueT],
    ) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
        self._kind = kind
        self._resource_kind = resource_kind
        self._value_type = value_type

    async def initialize(self) -> None:
        await super().initialize()

    def _identity(self, value: object) -> tuple[str, ...]:
        return _canonical_record_identity(self._kind, value)

    async def _decode(self, record: StoredRecord, target: type[ValueT]) -> ValueT:
        value = await super()._decode(record, target)
        if target is self._value_type:
            self._validate_record_identity(record, value)
        return value

    def _validate_record_identity(self, record: StoredRecord, value: ValueT) -> None:
        if record.kind != self._kind or record.key != self._key(self._kind, self._identity(value)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def create(self, value: ValueT) -> ValueT:
        _require_tenant(value, self._tenant_id)
        identity = self._identity(value)
        await self._insert(
            self._stored(self._kind, identity, value, state=_record_state(value))
        )
        _logger.debug("created Runtime record: kind=%s", self._kind)
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
            await self._stored_content(self._kind, identity, value, state=_record_state(value))
        )
        _logger.debug(
            "created Runtime record in group: kind=%s",
            self._kind,
        )
        return value

    async def get(self, identity: str, *, tenant_id: str) -> ValueT | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key(self._kind, (identity,))
        record = (
            await self._store.read_record(key)
            if self._kind in {"session", "execution"}
            else await self._record(key)
        )
        return None if record is None else await self._decode(record, self._value_type)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        identity: str,
        *,
        tenant_id: str,
    ) -> ValueT | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key(self._kind, (identity,)))
        return None if record is None else await self._decode(record, self._value_type)

    async def get_fresh(self, identity: str, *, tenant_id: str) -> ValueT | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key(self._kind, (identity,))
        records = await self._refresh_records((key,))
        record = records.get(key)
        return None if record is None else await self._decode(record, self._value_type)

    async def get_header_fresh(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        value = await self.get_fresh(identity, tenant_id=tenant_id)
        return None if value is None else self._header(value, self._resource_kind, identity)

    async def get_header(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        if tenant_id != self._tenant_id:
            return None
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
            current = await transaction.get_record(self._key(self._kind, (identity,)))
            if current is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            value = await self._decode(current, self._value_type)
            if _domain_revision(value) != expected_revision:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                _projected_record(self, current, next_record),
                current.storage_revision,
            )
            return next_record

        return await self._store.mutate(mutate)

    async def list_values(self, *, scope: bytes | None = None) -> tuple[ValueT, ...]:
        records = await self._records(self._kind, scope=scope)
        values = [await self._decode(record, self._value_type) for record in records]
        return tuple(values)


def operation_identity_matches(
    current: OperationLedgerRecord,
    candidate: OperationLedgerInput,
) -> bool:
    """Match a receipt independently of its outcome or producer claim."""
    return (
        current.operation_id == candidate.operation_id
        and current.tenant_id == candidate.tenant_id
        and current.resource_kind is candidate.resource_kind
        and current.resource_id == candidate.resource_id
        and current.execution_id == candidate.execution_id
        and current.operation_kind is candidate.operation_kind
        and current.request_digest == candidate.request_digest
        and current.request_digest_contract == candidate.request_digest_contract
        and current.compactable == candidate.compactable
    )


def validate_recovery_operation(
    operation: OperationLedgerInput,
    *,
    execution_id: str,
    tenant_id: str,
    producer: bool,
) -> None:
    if (
        operation.tenant_id != tenant_id
        or operation.resource_kind is not ResourceKind.EXECUTION
        or operation.resource_id != execution_id
        or operation.execution_id != execution_id
        or operation.operation_kind is not OperationKind.EXECUTION_RECOVER
        or operation.status is not OperationStatus.RUNNING
        or operation.compactable
        or operation.result_digest is not None
        or operation.error_code is not None
        or (
            not isinstance(operation.result_ref, str) or not operation.result_ref
            if producer else operation.result_ref is not None
        )
    ):
        raise ValueError("execution recovery operation is invalid")


class OperationLedgerRepositoryImpl(_RepositoryBase):
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
        key = operation_key(self._namespace, self._tenant_id, self._domain.value, value.operation_id)
        existing = await transaction.get_operation(key)
        if existing is not None:
            current = _decode_operation(existing, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain)
            if _operation_matches(current, value):
                return current
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        stream = stream_key(
            self._namespace, self._tenant_id, self._domain.value,
            "operation", [value.resource_kind.value, value.resource_id],
        )
        sequence = await transaction.next_sequence(stream)
        await transaction.insert_operation(StoredOperation(
            key, stream, sequence, value.status.value, value.compactable, _domain_data(value),
        ))
        return _operation_record(value, sequence)

    async def get(
        self, operation_id: str, *, tenant_id: str
    ) -> OperationLedgerRecord | None:
        if tenant_id != self._tenant_id:
            return None
        key = operation_key(self._namespace, self._tenant_id, self._domain.value, operation_id)
        stored = await self._store.read(
            lambda transaction: transaction.get_operation(key)
        )
        return None if stored is None else _decode_operation(stored, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain)

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
        return None if stored is None else _decode_operation(stored, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain)

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
                in {
                    OperationStatus.SUCCEEDED.value,
                    OperationStatus.FAILED.value,
                    OperationStatus.CANCELLED.value,
                }
                and next_record.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not operation_cas_immutable_matches(
                _decode_operation(current, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain), next_record
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
        states: frozenset[OperationStatus] | None = None,
        after_sequence: int | None = None,
    ) -> tuple[OperationLedgerRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "operation",
            [resource_kind.value, resource_id],
        )
        values = await self._store.read(
            lambda transaction: transaction.list_operations(
                OperationQuery(
                    stream_key=stream,
                    states=(
                        frozenset({"PENDING", "RUNNING"})
                        if states is None
                        else frozenset(status.value for status in states)
                    ),
                    limit=limit,
                    after_sequence=after_sequence,
                )
            )
        )
        return tuple(_decode_operation(value, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain) for value in values)

    async def compact_terminal(
        self,
        resource_kind: ResourceKind,
        resource_id: str,
        *,
        tenant_id: str,
        through_sequence: int,
    ) -> None:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "operation",
            [resource_kind.value, resource_id],
        )
        async def compact(
            transaction: StateTransaction,
        ) -> None:
            query = OperationQuery(
                stream_key=stream,
                states=frozenset({"SUCCEEDED", "FAILED", "CANCELLED"}),
                through_sequence=through_sequence,
                compactable=True,
            )
            candidates = await transaction.list_operations(query)
            if len(candidates) <= 1:
                return
            anchor = max(
                candidates,
                key=lambda value: (value.sequence, value.key),
            )
            await transaction.delete_operations(
                OperationQuery(
                    stream_key=stream,
                    states=frozenset({"SUCCEEDED", "FAILED", "CANCELLED"}),
                    through_sequence=anchor.sequence - 1,
                    compactable=True,
                )
            )

        await self._store.mutate(compact)


def project_record(
    *,
    namespace: str,
    tenant_id: str,
    domain: PersistenceDomain,
    kind: str,
    identity: Sequence[str],
    value: object,
    scope: bytes | None = None,
    parent: bytes | None = None,
    state: str | None = None,
    sort_key: bytes | None = None,
    storage_revision: int = 0,
    encoded: JsonValue | None = None,
) -> StoredRecord:
    """Project one logical value into its canonical physical Runtime record."""
    if scope is None:
        scope = _default_record_scope(
            namespace,
            tenant_id,
            domain,
            kind,
            value,
        )
    if parent is None:
        parent = _default_record_parent(
            namespace,
            tenant_id,
            domain,
            kind,
            value,
        )
    if isinstance(value, ExecutionRecord):
        state = _record_state(value)
    lease_owner, lease_fence, lease_expires_at = _record_lease(value)
    key = record_key(namespace, tenant_id, domain.value, kind, identity)
    return StoredRecord(
        key,
        scope,
        parent,
        kind,
        key if sort_key is None else sort_key,
        state,
        storage_revision,
        lease_owner,
        lease_fence,
        lease_expires_at,
        _domain_data(value) if encoded is None else encode_envelope({"type": wire_type_id(value), "payload": encoded}),
        subject_id=value.tool_call_id if isinstance(value, ToolOperationRecord) else None,
    )


def _memory_scope_key(
    namespace: str,
    tenant_id: str,
    memory_scope: str,
    transient_execution_id: str | None,
) -> bytes:
    """Keep Memory's query scope compact so its index can retain full paths."""
    return hashlib.sha256(scope_key(
        namespace, tenant_id, PersistenceDomain.MEMORY.value, "memory", "memory_scope",
        MemoryRecord.scope_components(memory_scope, transient_execution_id),
    )).digest()


def _default_record_scope(
    namespace: str,
    tenant_id: str,
    domain: PersistenceDomain,
    kind: str,
    value: object,
) -> bytes | None:
    if isinstance(value, EvaluationReport):
        return scope_key(namespace, tenant_id, domain.value, kind, "experiment", (value.experiment_id,))
    if isinstance(value, SessionRecord):
        return scope_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "owner",
            (value.owner_principal_id,),
        )
    if isinstance(value, ExecutionRecord) and value.session_id is not None:
        return scope_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "session",
            (value.session_id,),
        )
    if isinstance(value, IdempotencyRecord):
        return scope_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "resource",
            [value.resource_kind.value, value.resource_id],
        )
    if isinstance(
        value,
        (ArtifactRecord, ApprovalRecord, ExternalCallRecord),
    ):
        return scope_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "execution",
            (value.execution_id,),
        )
    if isinstance(value, MemoryRecord):
        return _memory_scope_key(namespace, tenant_id, value.memory_scope, value.transient_execution_id)
    if isinstance(value, ToolOperationRecord):
        return scope_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "agent_run",
            (value.agent_run_id,),
        )
    return None


def _default_record_parent(
    namespace: str,
    tenant_id: str,
    domain: PersistenceDomain,
    kind: str,
    value: object,
) -> bytes | None:
    if isinstance(value, TaskNodeView):
        return parent_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "graph",
            (value.graph_id,),
        )
    if isinstance(value, ExecutionRecord) and value.parent_execution_id is not None:
        return parent_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "execution",
            (value.parent_execution_id,),
        )
    if isinstance(value, ToolOperationRecord):
        return parent_key(
            namespace,
            tenant_id,
            domain.value,
            kind,
            "execution",
            (value.execution_id,),
        )
    return None


def _canonical_record_identity(kind: str, value: object) -> tuple[str, ...]:
    if isinstance(value, IdempotencyRecord):
        return (value.scope, value.idempotency_key)
    if isinstance(value, TaskGraphView):
        return (value.graph_id,)
    if isinstance(value, TaskNodeView):
        return (value.graph_id, value.node_id)
    if isinstance(value, ToolOperationRecord):
        return (value.tool_operation_id,)
    if isinstance(value, SessionRecord):
        return (value.session_id,)
    if isinstance(value, ExecutionRecord):
        return (value.execution_id,)
    if isinstance(value, MemoryRecord):
        return (value.memory_id,)
    if isinstance(value, ArtifactRecord):
        return (value.artifact_id,)
    if isinstance(value, EvaluationRecord):
        return (value.experiment_id,)
    if isinstance(value, (EvaluationTombstone, EvaluationCleanupRecord)):
        return (value.evaluation_id,)
    if isinstance(value, EvaluationContentTombstone):
        return (value.kind, *value.record_identity)
    if isinstance(value, EvaluationDatasetRecord):
        return (value.contract.ref.id, str(value.contract.ref.revision))
    if isinstance(value, EvaluationCaseRecord):
        return (value.contract.ref.dataset_id, value.contract.ref.case_id, str(value.contract.ref.revision))
    if isinstance(value, EvidenceBundle):
        return (value.ref.evidence_id,)
    if isinstance(value, (EvaluationReport, ComparisonReport)):
        return (value.report_id,)
    if isinstance(value, ApprovalRecord):
        return (value.approval_id,)
    if isinstance(value, ExternalCallRecord):
        return (value.call_id,)
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
        **({"lease_expires_at": None} if target is TaskNodeView else {}),
    }
    return restored


def _record_lease(value: object) -> tuple[str | None, int, datetime | None]:
    if isinstance(value, ToolOperationRecord):
        return value.owner, value.fence, None
    if isinstance(value, TaskNodeView):
        return value.owner, value.fence, value.lease_expires_at
    return None, 0, None


def _require_tenant(value: object, tenant_id: str) -> None:
    if isinstance(value, (OperationLedgerRecord, OperationLedgerInput)) and (
        value.tenant_id != tenant_id
    ):
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
        ),
    ):
        return value.revision
    return 0


def _status_value(value: object) -> str | None:
    if isinstance(value, ExecutionRecord) and value.recovery is not None and value.recovery.handoff_phase is RecoveryHandoffPhase.PREPARED:
        return "execution_handoff"
    if isinstance(
        value,
        (
            SessionRecord,
            ExecutionRecord,
            IdempotencyRecord,
            ApprovalRecord,
            ExternalCallRecord,
            TaskGraphView,
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
        candidate, expected_storage_revision=expected
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
    return replace(projected, storage_revision=current.storage_revision + 1)


def _require_session_identity(
    current: SessionRecord,
    candidate: SessionRecord,
) -> None:
    if (
        candidate.session_id,
        candidate.owner_principal_id,
        candidate.agent_id,
        candidate.history_id,
        candidate.created_at,
    ) != (
        current.session_id,
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
    key = operation_key(repository._namespace, repository._tenant_id, repository._domain.value, value.operation_id)
    existing = await transaction.get_operation(key)
    if existing is not None:
        current = _decode_operation(existing, namespace=repository._namespace, tenant_id=repository._tenant_id, domain=repository._domain)
        if _operation_matches(current, value):
            return current, True
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    sequence = await transaction.next_sequence(
        stream_key(
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
    key = operation_key(repository._namespace, repository._tenant_id, repository._domain.value, value.operation_id)
    await transaction.insert_operation(
        StoredOperation(
            key,
            stream_key(
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


def _decode_operation(
    value: StoredOperation,
    *,
    namespace: str,
    tenant_id: str,
    domain: PersistenceDomain,
) -> OperationLedgerRecord:
    candidate = _decode_enveloped_domain(value.data, OperationLedgerInput)
    try:
        expected_key = operation_key(namespace, tenant_id, domain.value, candidate.operation_id)
        expected_stream = stream_key(
            namespace, tenant_id, domain.value, "operation",
            (candidate.resource_kind.value, candidate.resource_id),
        )
    except (AIError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if (candidate.tenant_id != tenant_id or value.key != expected_key
            or value.stream_key != expected_stream or value.state != candidate.status.value
            or value.compactable != candidate.compactable):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        value.request_digest_contract,
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
        value.request_digest_contract,
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
    sort_key = validate_sort_key(record.sort_key)
    key = validate_record_key(record.key)
    payload = len(key).to_bytes(2, "big") + key + sort_key
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _decode_record_cursor(cursor: str | None) -> tuple[bytes | None, bytes | None]:
    if cursor is None:
        return None, None
    try:
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("cursor length")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        key_size = int.from_bytes(raw[:2], "big")
        if not 1 <= key_size <= RECORD_KEY_MAX_BYTES or len(raw) <= 2 + key_size:
            raise ValueError("cursor identity")
        if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != cursor:
            raise ValueError("cursor encoding")
        return validate_sort_key(raw[2 + key_size:]), validate_record_key(raw[2:2 + key_size])
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


RepositoryBase = _RepositoryBase
ResourceRepository = _ResourceRepository
append_operation = _append_operation
decode_operation = _decode_operation
decode_record_cursor = _decode_record_cursor
canonical_record_identity = _canonical_record_identity
domain_data = _domain_data
memory_scope_key = _memory_scope_key
insert_operation = _insert_operation
projected_record = _projected_record
record_cursor = _record_cursor
record_state = _record_state
restore_lease_fields = _restore_lease_fields
replace_checked = _replace_checked
require_repository_tenant = _require_repository_tenant
require_tenant = _require_tenant
reserve_operation = _reserve_operation
stored_from_operation = _stored_from_operation
validate_page_limit = _validate_page_limit


__all__ = [
    "OperationLedgerRepositoryImpl",
    "RepositoryBase",
    "ResourceRepository",
    "append_operation",
    "operation_identity_matches",
    "validate_recovery_operation",
    "decode_operation",
    "decode_record_cursor",
    "canonical_record_identity",
    "domain_data",
    "memory_scope_key",
    "insert_operation",
    "projected_record",
    "record_cursor",
    "record_state",
    "restore_lease_fields",
    "project_record",
    "replace_checked",
    "require_repository_tenant",
    "require_tenant",
    "reserve_operation",
    "stored_from_operation",
    "validate_page_limit",
]
