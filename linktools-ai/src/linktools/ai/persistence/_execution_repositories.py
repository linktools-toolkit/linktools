#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-domain repository implementations."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import TypeVar
from ._message_root import NativeTextReference
from linktools.core import environ
from ..core import (
    ExecutionEventType,
    ExecutionStatus,
    IdempotencyStatus,
    JsonValue,
    OperationLedgerRecord,
    Page,
    ResourceKind,
    validate_resource_id,
    validate_idempotency_key,
    ResourceRef,
)
from ..errors import AIError, ErrorCode
from ..task import TaskBindingContract
from ..execution import ExecutionMetadataRecord, ExecutionControlRecord, ExecutionResultRecord, ExecutionRetentionRecord, ExecutionTimelineRecord, ExecutionView
from .codec import decode_execution_metadata, decode_execution_control, decode_execution_retention, encode_envelope, encode_domain, decode_execution_producer_control, decode_execution_timeline, decode_execution_header, decode_execution_view, decode_envelope, decode_execution_content
from ._archive import ContentArchive, PreparedArchive
from ._execution_content import _ContentReadRequired
from ._budget import BudgetRepositoryImpl
from .contracts import (
    ExecutionCandidate,
    ExecutionCandidatePage,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionEventRecord,
    ExecutionInsertionReceipt,
    ExecutionHistorySealRecord,
    ExecutionRecord,
    ExecutionRecoveryDetail,
    ExecutionRecoveryState,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartUnknownCommit,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    IdempotencyRecord,
    IdempotencyTerminalUpdate,
    ResultRecord,
)
from .contracts import RuntimePayloadRef
from ._plan import PersistenceDomain
from .state import (
    FactQuery,
    Observed,
    RecordQuery,
    RecordReplacement,
    StateStore,
    StateGroupCallback,
    active_state_scope,
    StateTransaction,
    StoredFact,
    StoredRecord,
    operation_key,
    stream_key,
)
from ._repository_common import (
    RepositoryBase as _RepositoryBase,
    ResourceRepository as _ResourceRepository,
    decode_operation as _decode_operation,
    projected_record as _projected_record,
    record_cursor as _record_cursor,
    replace_checked as _replace_checked,
    require_repository_tenant as _require_repository_tenant,
    require_tenant as _require_tenant,
    stored_from_operation as _stored_from_operation,
    validate_page_limit as _validate_page_limit,
)

_logger = environ.get_logger("ai.runtime.state.repositories")


class IdempotencyRepositoryImpl(_ResourceRepository[IdempotencyRecord]):
    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        domain: PersistenceDomain,
    ) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=domain,
            kind="idempotency",
            resource_kind=ResourceKind.EXECUTION
            if domain is PersistenceDomain.EXECUTION
            else ResourceKind.EVALUATION,
            value_type=IdempotencyRecord,
        )

    def _require_resource_kind(self, record: IdempotencyRecord) -> None:
        if record.resource_kind is not self._resource_kind:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _identity_key(self, scope: str, key: str) -> tuple[str, str]:
        return validate_resource_id(scope), validate_idempotency_key(key)

    async def reserve(self, record: IdempotencyRecord) -> IdempotencyRecord:
        _require_tenant(record, self._tenant_id)
        self._require_resource_kind(record)
        identity = self._identity_key(record.scope, record.idempotency_key)
        try:
            await self._insert(
                await self._stored_content("idempotency", identity, record, state=record.status.value)
            )
            return record
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            stored = await self._record(self._key(self._kind, identity))
            existing = None if stored is None else await self._decode(stored, IdempotencyRecord)
            if existing is not None and _same_idempotency(existing, record):
                return existing
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def get(
        self,
        scope: str,
        idempotency_key: str,
        *,
        tenant_id: str,
    ) -> IdempotencyRecord | None:  # type: ignore[override]
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key(self._kind, self._identity_key(scope, idempotency_key)))
        return None if record is None else await self._decode(record, IdempotencyRecord)

    async def list_by_resource(
        self, resource_kind: ResourceKind, resource_id: str, *, tenant_id: str
    ) -> tuple[IdempotencyRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            self._kind,
            scope=self._scope(
                "idempotency", "resource", [resource_kind.value, resource_id]
            ),
        )
        values = [await self._decode(record, self._value_type) for record in records]
        return tuple(values)

    async def compare_and_swap(
        self,
        scope: str,
        idempotency_key: str,
        *,
        tenant_id: str,
        expected_status: IdempotencyStatus,
        next_record: IdempotencyRecord,
    ) -> IdempotencyRecord:  # type: ignore[override]
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_tenant(next_record, self._tenant_id)
        self._require_resource_kind(next_record)
        if next_record.scope != scope or next_record.idempotency_key != idempotency_key:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        identity = self._identity_key(scope, idempotency_key)

        async def mutate(transaction: StateTransaction) -> IdempotencyRecord:
            current_record = await transaction.get_record(
                self._key(self._kind, identity)
            )
            if current_record is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(current_record, IdempotencyRecord)
            if current.status is not expected_status:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await _replace_checked(
                transaction,
                await self._projected_content(current_record, next_record),
                current_record.storage_revision,
            )
            return next_record

        return await self._store.mutate(mutate)


_ContentValueT = TypeVar("_ContentValueT")


class ExecutionRepositoryImpl(_ResourceRepository[ExecutionRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=PersistenceDomain.EXECUTION,
            kind="execution",
            resource_kind=ResourceKind.EXECUTION,
            value_type=ExecutionRecord,
        )
        self._budgets = BudgetRepositoryImpl(
            store, namespace=namespace, tenant_id=tenant_id,
        )
        self._idempotency = IdempotencyRepositoryImpl(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=PersistenceDomain.EXECUTION,
        )

    async def read_content(
        self, capture: StateGroupCallback[_ContentValueT], *, stores: Sequence[StateStore] = (),
    ) -> _ContentValueT:
        """Read owners and declared content in one native group snapshot.

        A cold content dependency restarts this read-only capture once after
        opening its archive. Callbacks must rebuild their observations on entry.
        Inline-only captures do not discover or open archive storage.
        """
        group = self._store.storage_group
        members = tuple(dict.fromkeys((self._store, *stores)))
        try:
            return await group.read_many(members, capture)
        except _ContentReadRequired as required:
            if required.group is not group or active_state_scope() is not None:
                raise
        await self._execution_content.initialize(create=False, share_content=False)
        return await group.read_many(members, capture)

    async def get(self, identity: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self.read_content(lambda _group: super(ExecutionRepositoryImpl, self).get(identity, tenant_id=tenant_id))

    async def get_fresh(self, identity: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self.read_content(lambda _group: super(ExecutionRepositoryImpl, self).get_fresh(identity, tenant_id=tenant_id))

    async def _decode_many(self, records: Sequence[StoredRecord]) -> tuple[ExecutionRecord, ...]:
        values = await self._execution_content.decode_many(records)
        for record, value in zip(records, values, strict=True):
            self._validate_record_identity(record, value)
        return values

    async def get_metadata_record(self, execution_id: str, *, tenant_id: str,
                                  fresh: bool = False) -> ExecutionMetadataRecord | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key("execution", (execution_id,))
        async def read():
            record = ((await self._refresh_records((key,))).get(key) if fresh else
                      await self._store.read_record(key))
            if record is None:
                return None
            self._decode_header(record, execution_id)
            metadata = decode_execution_metadata(record.data, tenant_id=tenant_id)
            if fresh and metadata.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                payloads = await self._execution_content.read_payloads((record,), result_only=True)
                metadata = replace(metadata, result_record=decode_execution_content(record.data, payloads, result=True))
            return metadata
        return await self.read_content(lambda _group: read()) if fresh else await read()

    async def get_result_record(self, execution_id: str, *, tenant_id: str,
                                fresh: bool = False) -> ExecutionResultRecord | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key("execution", (execution_id,))
        async def read(_transactions):
            record = ((await self._refresh_records((key,))).get(key) if fresh else
                      await self._store.read_record(key))
            if record is None:
                return None
            self._decode_header(record, execution_id)
            payloads = await self._execution_content.read_payloads((record,), result_only=True)
            return decode_execution_content(record.data, payloads, result=True)
        return await self.read_content(read)

    async def get_retention_in_transaction(self, transaction: StateTransaction, execution_id: str,
                                          *, tenant_id: str) -> ExecutionRetentionRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", (execution_id,)))
        if record is None:
            return None
        self._decode_header(record, execution_id)
        return decode_execution_retention(record.data)

    async def get_header(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._store.read_record(self._key("execution", (identity,)))
        return None if record is None else self._decode_header(record, identity)

    async def get_header_fresh(self, identity: str, *, tenant_id: str) -> ResourceRef | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key("execution", (identity,))
        record = (await self._refresh_records((key,))).get(key)
        return None if record is None else self._decode_header(record, identity)

    def _decode_header(self, record: StoredRecord, identity: str) -> ResourceRef:
        header = decode_execution_header(record.data, tenant_id=self._tenant_id)
        if (record.kind != "execution" or record.key != self._key("execution", (identity,))
                or header.id != identity):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return header

    async def get_view(self, execution_id: str, *, tenant_id: str) -> tuple[ResourceRef, ExecutionView] | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._store.read_record(self._key("execution", (execution_id,)))
        return None if record is None else await self._decode_view(record, execution_id)

    async def get_view_fresh(self, execution_id: str, *, tenant_id: str) -> tuple[ResourceRef, ExecutionView] | None:
        if tenant_id != self._tenant_id:
            return None
        key = self._key("execution", (execution_id,))
        record = (await self._refresh_records((key,))).get(key)
        return None if record is None else await self._decode_view(record, execution_id)

    async def _decode_view(self, record: StoredRecord, execution_id: str) -> tuple[ResourceRef, ExecutionView]:
        self._decode_header(record, execution_id)
        return decode_execution_view(record.data, tenant_id=self._tenant_id)

    async def list_child_views(self, execution_id: str, *, tenant_id: str) -> tuple[ExecutionView, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records("execution", parent=self._parent("execution", "execution", (execution_id,)))
        values = []
        for record in records:
            header = decode_execution_header(record.data, tenant_id=tenant_id)
            self._decode_header(record, header.id)
            _owner, view = decode_execution_view(record.data, tenant_id=tenant_id)
            if view.parent_execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(view)
        return tuple(values)

    async def get_view_with_children(
        self, execution_id: str, *, tenant_id: str,
    ) -> tuple[tuple[ResourceRef, ExecutionView] | None, tuple[ExecutionView, ...]]:
        """Capture parent and child view projections in the same read scope."""
        if tenant_id != self._tenant_id:
            return None, ()

        async def capture(transaction: StateTransaction) -> tuple[StoredRecord | None, tuple[StoredRecord, ...]]:
            parent = await transaction.get_record(self._key("execution", (execution_id,)))
            if parent is None:
                return None, ()
            children = await self._records(
                "execution", parent=self._parent("execution", "execution", (execution_id,)),
            )
            return parent, tuple(children)

        parent, children = await self._store.read(capture)
        if parent is None:
            return None, ()
        owner = await self._decode_view(parent, execution_id)
        views = []
        for record in children:
            header = decode_execution_header(record.data, tenant_id=tenant_id)
            _principal, view = await self._decode_view(record, header.id)
            if view.parent_execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            views.append(view)
        return owner, tuple(views)

    async def _projected_content(self, current: StoredRecord, value: object, *,
                                 result_source: NativeTextReference | None = None) -> StoredRecord:
        candidate = await super()._projected_content(current, value, result_source=result_source)
        if current.state in {ExecutionStatus.SUCCEEDED.value, ExecutionStatus.FAILED.value,
                             ExecutionStatus.CANCELLED.value, "execution_handoff"}:
            if not isinstance(value, ExecutionRecord) or not self._execution_content.terminal_matches(current, candidate, value):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        return candidate

    async def get_timeline_many(
        self, execution_ids: Sequence[str], *, tenant_id: str,
    ) -> Mapping[str, ExecutionTimelineRecord]:
        async def read(_group):
            _require_repository_tenant(tenant_id, self._tenant_id)
            if isinstance(execution_ids, (str, bytes)):
                raise TypeError("execution_ids must be a sequence of strings")
            ordered = tuple(dict.fromkeys(execution_ids))
            if any(not isinstance(value, str) or not value for value in ordered):
                raise ValueError("execution_ids must contain non-empty strings")
            keys = {value: self._key("execution", (value,)) for value in ordered}
            records = await self._store.read_records(tuple(keys.values())) if keys else {}
            payloads = await self._execution_content.read_payloads(tuple(records.values()))
            values: dict[str, ExecutionTimelineRecord] = {}
            for execution_id, key in keys.items():
                record = records.get(key)
                if record is None:
                    continue
                value = decode_execution_content(record.data, payloads, timeline=True)
                if record.kind != "execution" or record.key != key or value.execution_id != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[execution_id] = value
            return values
        return await self.read_content(read)

    async def get_recovery(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return execution if execution is not None and execution.recovery is not None else None

    async def get_recovery_in_transaction(
        self, transaction: StateTransaction, execution_id: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        execution = await self.get_in_transaction(transaction, execution_id, tenant_id=tenant_id)
        return execution if execution is not None and execution.recovery is not None else None

    async def get_observed(self, execution_id: str, *, tenant_id: str) -> Observed[ExecutionRecord] | None:
        async def read(_group):
            _require_repository_tenant(tenant_id, self._tenant_id)
            record = await self._record(self._key("execution", (execution_id,)))
            return None if record is None else Observed(await self._decode(record, ExecutionRecord), record.storage_revision)
        return await self.read_content(read)

    async def get_observed_in_transaction(
        self, transaction: StateTransaction, execution_id: str, *, tenant_id: str,
    ) -> Observed[ExecutionRecord] | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", (execution_id,)))
        return None if record is None else Observed(await self._decode(record, ExecutionRecord), record.storage_revision)

    async def list_recoverable_page(
        self, *, tenant_id: str, cursor: str | None, limit: int,
    ) -> Page[ExecutionRecord]:
        async def read(_group):
            if tenant_id != self._tenant_id:
                return Page(())
            _validate_page_limit(limit)
            states = frozenset(status.value for status in (
                ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.RECOVERY_REQUIRED,
                ExecutionStatus.CANCELLING, ExecutionStatus.FINALIZING, ExecutionStatus.WAITING_DEFERRED,
            )) | {"execution_handoff"}
            selected: list[tuple[StoredRecord, ExecutionRecord]] = []
            next_cursor = cursor
            while len(selected) <= limit:
                records = await self._records("execution", states=states, cursor=next_cursor, limit=limit + 1)
                if not records:
                    break
                for record, value in zip(records, await self._decode_many(records), strict=True):
                    next_cursor = _record_cursor(record)
                    if value.recovery is not None and value.recovery_state is not ExecutionRecoveryState.COMPLETED:
                        selected.append((record, value))
                        if len(selected) > limit:
                            return Page(tuple(item[1] for item in selected[:limit]), _record_cursor(selected[limit - 1][0]))
                if len(records) < limit + 1:
                    break
            return Page(tuple(item[1] for item in selected))
        return await self.read_content(read)

    async def compare_and_swap(
        self, execution_id: str, *, tenant_id: str,
        expected_revision: int, next_record: ExecutionRecord,
    ) -> ExecutionRecord:
        await self._execution_content.prepare_value(next_record)
        return await self._store.mutate(lambda transaction: self.compare_and_swap_in_transaction(
            transaction, execution_id, tenant_id=tenant_id,
            expected_revision=expected_revision, next_record=next_record,
        ))

    async def compare_and_swap_in_transaction(
        self, transaction: StateTransaction, execution_id: str, *, tenant_id: str,
        expected_revision: int, next_record: ExecutionRecord,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        stored = await transaction.get_record(self._key("execution", (execution_id,)))
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        self._decode_header(stored, execution_id)
        fields = decode_envelope(stored.data).value["payload"]["fields"]
        if type(fields.get("revision")) is not int:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if fields["revision"] != expected_revision or next_record.execution_id != execution_id or next_record.revision != expected_revision + 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await _replace_checked(transaction, await self._projected_content(stored, next_record), stored.storage_revision)
        return next_record

    async def commit_boundary_in_transaction(
        self, transaction: StateTransaction, observed: Observed[ExecutionRecord], *,
        next_record: ExecutionRecord, expected_producer_generation: int,
        expected_producer_claim_id: str, allow_cancelling: bool = False,
    ) -> Observed[ExecutionRecord]:
        """Publish a prepared boundary using its confirmed control receipt."""
        current = observed.value
        if (current.execution_id != next_record.execution_id
                or not (current.status is ExecutionStatus.STARTED
                        or allow_cancelling and current.status is ExecutionStatus.CANCELLING
                        and next_record == current)
                or current.producer_generation != expected_producer_generation
                or current.producer_claim_id != expected_producer_claim_id
                or next_record.producer_generation != current.producer_generation
                or next_record.producer_claim_id != current.producer_claim_id
                or (next_record != current and next_record.revision != current.revision + 1)):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if next_record == current:
            revision = await transaction.guard_record_revision(
                self._key("execution", (current.execution_id,)), expected_storage_revision=observed.storage_revision,
            )
            if revision is None or revision != observed.storage_revision + 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return Observed(current, revision)
        candidate = replace(await self._stored_content("execution", (next_record.execution_id,), next_record,
                                         state=next_record.status.value, previous_value=current),
                            storage_revision=observed.storage_revision + 1)
        await _replace_checked(transaction, candidate, observed.storage_revision)
        return Observed(next_record, candidate.storage_revision)

    async def get_many(
        self,
        execution_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        return await self._get_many(execution_ids, tenant_id=tenant_id, refresh=False)

    async def get_many_fresh(
        self, execution_ids: Sequence[str], *, tenant_id: str,
    ) -> Mapping[str, ExecutionRecord]:
        return await self._get_many(execution_ids, tenant_id=tenant_id, refresh=True)

    async def _get_many(
        self, execution_ids: Sequence[str], *, tenant_id: str, refresh: bool,
    ) -> Mapping[str, ExecutionRecord]:
        async def read(_group):
            if tenant_id != self._tenant_id:
                return {}
            if isinstance(execution_ids, (str, bytes)):
                raise TypeError("execution_ids must be a sequence of strings")
            ordered = tuple(dict.fromkeys(execution_ids))
            if any(not isinstance(execution_id, str) or not execution_id for execution_id in ordered):
                raise ValueError("execution_ids must contain non-empty strings")
            if not ordered:
                return {}
            keys = {
                execution_id: self._key("execution", (execution_id,))
                for execution_id in ordered
            }

            records = (await self._refresh_records(tuple(keys.values())) if refresh else
                       await self._store.read_records(tuple(keys.values())))
            decoded = {record.key: value for record, value in zip(records.values(),
                await self._decode_many(tuple(records.values())), strict=True)}
            values: dict[str, ExecutionRecord] = {}
            for execution_id, key in keys.items():
                record = records.get(key)
                if record is None:
                    continue
                value = decoded[key]
                if value.execution_id != execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[execution_id] = value
            return values
        return await self.read_content(read)

    def execution_record_key(self, execution_id: str) -> bytes:
        """Identify the canonical record for a transaction-local read receipt."""
        return self._key("execution", (execution_id,))

    def producer_record_keys(self, execution_id: str) -> tuple[bytes, ...]:
        """Identify the source control records checked by producer publication."""
        return (self._key("execution", (execution_id,)),)

    def terminal_record_keys(
        self, execution: ExecutionRecord, *, include_history_seal: bool,
    ) -> tuple[bytes, ...]:
        """Identify control records checked by terminal publication."""
        _require_tenant(execution, self._tenant_id)
        kinds = ("execution",)
        if include_history_seal:
            kinds = (*kinds, "execution_history_seal")
        return tuple(self._key(kind, (execution.execution_id,)) for kind in kinds)

    async def prepare_content(self, value: ExecutionRecord) -> None:
        """Enlist immutable content storage before entering an owner transaction."""
        _require_tenant(value, self._tenant_id)
        await self._execution_content.prepare_value(value)

    async def create(self, value: ExecutionRecord) -> ExecutionRecord:
        return await self.create_execution(value)

    async def create_execution(self, execution: ExecutionRecord) -> ExecutionRecord:
        _require_tenant(execution, self._tenant_id)
        await self.prepare_content(execution)
        return await self._store.mutate(
            lambda transaction: self.create_execution_in_transaction(transaction, execution)
        )

    async def create_execution_in_transaction(
        self, transaction: StateTransaction, execution: ExecutionRecord,
    ) -> ExecutionRecord:
        _require_tenant(execution, self._tenant_id)
        if execution.input_context is not None:
            execution = replace(execution, repository_instructions=None)
        current = await transaction.get_record(self._key("execution", (execution.execution_id,)))
        if current is not None:
            existing = await self._decode(current, ExecutionRecord)
            if existing != execution:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return existing
        await transaction.insert_record(await self._stored_content(
            "execution", (execution.execution_id,), execution, state=execution.status.value,
        ))
        return execution

    async def list_by_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        statuses: frozenset[ExecutionStatus] | None = None,
    ) -> tuple[ExecutionRecord, ...]:
        async def read(_group):
            if tenant_id != self._tenant_id:
                return ()
            records = await self._records(
                "execution",
                scope=self._scope("execution", "session", (session_id,)),
                states=None
                if statuses is None
                else frozenset(status.value for status in statuses) | {"execution_handoff"},
            )
            values = await self._decode_many(records)
            return tuple(value for value in values if statuses is None or value.status in statuses)
        return await self.read_content(read)

    async def list_children(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ExecutionRecord, ...]:
        async def read(_group):
            if tenant_id != self._tenant_id:
                return ()
            records = await self._records(
                "execution",
                parent=self._parent("execution", "execution", (execution_id,)),
            )
            return await self._decode_many(records)
        return await self.read_content(read)

    async def list_candidates(
        self,
        *,
        tenant_id: str,
        session_id: str | None,
        parent_execution_id: str | None,
        cursor: str | None,
        limit: int,
    ) -> ExecutionCandidatePage:
        async def read(_group):
            if tenant_id != self._tenant_id:
                return ExecutionCandidatePage((), False)
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= 1000
            ):
                raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
            scope = (
                None
                if session_id is None or parent_execution_id is not None
                else self._scope("execution", "session", (session_id,))
            )
            parent = (
                None
                if parent_execution_id is None
                else self._parent("execution", "execution", (parent_execution_id,))
            )
            records = await self._records(
                "execution",
                scope=scope,
                parent=parent,
                cursor=cursor,
                limit=limit,
            )
            selected = records[:limit]
            candidates: list[ExecutionCandidate] = []
            for record, value in zip(selected, await self._decode_many(selected), strict=True):
                candidates.append(
                    ExecutionCandidate(
                        value,
                        _record_cursor(record),
                    )
                )
            has_more = False
            if len(records) == limit:
                has_more = await self._has_records(
                    "execution",
                    scope=scope,
                    parent=parent,
                    cursor=_record_cursor(records[-1]),
                )
            return ExecutionCandidatePage(
                tuple(candidates),
                has_more,
            )
        return await self.read_content(read)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", (execution_id,)))
        return None if record is None else await self._decode(record, ExecutionRecord)

    async def guard_recovery_in_transaction(
        self,
        transaction: StateTransaction,
        execution: ExecutionRecord,
    ) -> None:
        """Fence a recovery control admission without changing execution truth."""
        key = self._key("execution", (execution.execution_id,))
        stored = await transaction.get_record(key)
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ExecutionRecord)
        if (
            current.revision != execution.revision
            or current.event_seq != execution.event_seq
            or current.status is not execution.status
            or current.status not in {
                ExecutionStatus.PENDING_START,
                ExecutionStatus.STARTED,
                ExecutionStatus.RECOVERY_REQUIRED,
                ExecutionStatus.CANCELLING,
                ExecutionStatus.FINALIZING,
            }
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        guarded = await transaction.guard_record(
            key, expected_storage_revision=stored.storage_revision,
        )
        if guarded is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def record_recovery_cancel_in_transaction(
        self,
        transaction: StateTransaction,
        execution: ExecutionRecord,
        *,
        operation_id: str,
        admit_operation: Callable[[], Awaitable[OperationLedgerRecord]],
    ) -> OperationLedgerRecord:
        """Admit the receipt in this group transaction, then publish its cancel intent."""
        key = self._key("execution", (execution.execution_id,))
        stored = await transaction.get_record(key)
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ExecutionRecord)
        if (
            current.status is not ExecutionStatus.RECOVERY_REQUIRED
            or current.revision != execution.revision
            or current.event_seq != execution.event_seq
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        admitted = await admit_operation()
        updated = replace(
            current,
            revision=current.revision + 1,
            event_seq=current.event_seq + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            await self._projected_content(stored, updated),
            stored.storage_revision,
        )
        stream = stream_key(
            self._namespace, self._tenant_id, self._domain.value,
            "execution", (execution.execution_id,),
        )
        await transaction.insert_facts((
            StoredFact(
                stream, updated.event_seq, key,
                ExecutionEventType.CANCEL_REQUESTED.value,
                None, None, {"operation_id": operation_id},
            ),
        ))
        return admitted

    async def transition_recovery_in_transaction(
        self,
        transaction: StateTransaction,
        execution: ExecutionRecord,
        *,
        next_status: ExecutionStatus,
        event_type: ExecutionEventType,
        event_payload: Mapping[str, JsonValue],
        next_error_code: str | None,
        next_safe_error_details: Mapping[str, JsonValue],
        audit_events: Sequence[ExecutionEventAppend] = (),
        producer_generation: int | None = None,
        agent_run_seq: int | None = None,
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        """Commit recovery status, producer fencing and its ordered events."""
        events = (*audit_events, ExecutionEventAppend(event_type, event_payload))
        if producer_generation is not None:
            await self.require_execution_producer_in_transaction(
                transaction, execution.execution_id,
                expected_producer_generation=producer_generation,
            )
        key = self._key("execution", (execution.execution_id,))
        stored = await transaction.get_record(key)
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ExecutionRecord)
        if (
            current.revision != execution.revision
            or current.event_seq != execution.event_seq
            or current.status is not execution.status
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if agent_run_seq is not None and (
            event_type is not ExecutionEventType.EXECUTION_RESUMED
            or agent_run_seq != current.agent_run_seq + 1
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if (event_type is ExecutionEventType.EXECUTION_RESUMED
                and execution.revision + len(events) <= current.producer_generation):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        updated = replace(
            current,
            status=next_status,
            producer_generation=(execution.revision + len(events)
                if event_type is ExecutionEventType.EXECUTION_RESUMED else current.producer_generation),
            producer_claim_id=(event_payload["producer_claim_id"]
                if event_type is ExecutionEventType.EXECUTION_RESUMED else current.producer_claim_id),
            agent_run_seq=current.agent_run_seq if agent_run_seq is None else agent_run_seq,
            recovery=current.recovery if recovery is None else recovery,
            revision=execution.revision + len(events),
            event_seq=execution.event_seq + len(events),
            error_code=next_error_code,
            safe_error_details=dict(next_safe_error_details),
            error_diagnostics=None,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            await self._projected_content(stored, updated),
            stored.storage_revision,
        )
        stream = stream_key(
            self._namespace, self._tenant_id, self._domain.value,
            "execution", (execution.execution_id,),
        )
        await transaction.insert_facts(tuple(
            StoredFact(
                stream, execution.event_seq + index, key,
                event.event_type, None, None, event.payload,
            )
            for index, event in enumerate(events, 1)
        ))
        return updated

    async def get_start_idempotency(
        self,
        claim: ExecutionStartClaim,
    ) -> IdempotencyRecord | None:
        return await self._idempotency.get(
            claim.scope,
            claim.idempotency_key,
            tenant_id=self._tenant_id,
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

    async def reserve_start(
        self, reservation: ExecutionStartReservation, *,
        input_context_write: tuple[ContentArchive, PreparedArchive] | None = None,
    ) -> ExecutionStartReservationResult:
        _require_tenant(reservation.execution, self._tenant_id)
        _require_tenant(reservation.idempotency, self._tenant_id)
        self._idempotency._require_resource_kind(reservation.idempotency)
        if reservation.execution.input_context is not None:
            reservation = replace(reservation, execution=replace(reservation.execution, repository_instructions=None))
        await self.prepare_content(reservation.execution)
        if reservation.budget is not None and (
            reservation.execution.budget_scope_id != "execution:" + reservation.execution.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def mutate(
            transaction: StateTransaction,
        ) -> ExecutionStartReservationResult:
            identity = self._idempotency._identity_key(
                reservation.idempotency.scope,
                reservation.idempotency.idempotency_key,
            )
            id_key = self._idempotency._key("idempotency", identity)
            execution_key = self._key("execution", (reservation.execution.execution_id,))
            records = await transaction.get_records((id_key, execution_key))
            existing_id = records.get(id_key)
            if existing_id is not None:
                existing_idempotency = await self._idempotency._decode(
                    existing_id, IdempotencyRecord
                )
                if not _same_idempotency_identity(
                    existing_idempotency, reservation.idempotency
                ):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                winner_key = self._key("execution", (existing_idempotency.resource_id,))
                if winner_key == execution_key:
                    winner_records = records
                else:
                    winner_records = await transaction.get_records((winner_key,))
                winner_record = winner_records.get(winner_key)
                if winner_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_value = await self._decode(winner_record, ExecutionRecord)
                _logger.debug(
                    "execution start reservation replayed: execution=%s",
                    existing_value.execution_id,
                )
                return ExecutionStartReservationResult(
                    existing_value, existing_idempotency, False
                )
            if execution_key in records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if reservation.execution.parent_execution_id is not None:
                if reservation.parent_producer_generation is None:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                await self._guard_subagent_parent_in_transaction(
                    transaction, reservation.execution,
                    parent_producer_generation=reservation.parent_producer_generation,
                )
            scope_id = reservation.execution.budget_scope_id
            if scope_id is not None:
                if reservation.budget is None:
                    await self._budgets.read_in_transaction(transaction, scope_id)
                else:
                    await self._budgets.ensure_in_transaction(
                        transaction, scope_id, reservation.budget,
                    )
            if input_context_write is not None:
                archive, prepared = input_context_write
                if archive.store.storage_group is not self._store.storage_group:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
            await transaction.insert_records(
                (
                    await self._stored_content(
                        "execution",
                        (reservation.execution.execution_id,),
                        reservation.execution,
                        state=reservation.execution.status.value,
                    ),
                    self._idempotency._stored(
                        "idempotency",
                        identity,
                        reservation.idempotency,
                        state=reservation.idempotency.status.value,
                    ),
                )
            )
            _logger.debug(
                "execution start reserved: execution=%s",
                reservation.execution.execution_id,
            )
            return ExecutionStartReservationResult(
                reservation.execution, reservation.idempotency, True
            )

        return await self._store.mutate(mutate)

    async def _guard_subagent_parent_in_transaction(
        self, transaction: StateTransaction, execution: ExecutionRecord, *,
        parent_producer_generation: int,
    ) -> None:
        """Order child admission against parent cancellation and producer takeover."""
        _require_tenant(execution, self._tenant_id)
        if (execution.parent_execution_id is None
                or isinstance(parent_producer_generation, bool)
                or not isinstance(parent_producer_generation, int) or parent_producer_generation < 1):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        parent = await self.get_in_transaction(
            transaction, execution.parent_execution_id, tenant_id=self._tenant_id,
        )
        if parent is None or parent.root_execution_id != execution.root_execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if parent.status is ExecutionStatus.CANCELLING:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        if parent.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self.require_execution_producer_in_transaction(
            transaction, parent.execution_id,
            expected_producer_generation=parent_producer_generation,
            allow_cancelling=False,
        )

    async def claim_start(self, claim: ExecutionStartClaim) -> ExecutionRecord:
        return await self._store.mutate(
            lambda transaction: self.claim_start_in_transaction(transaction, claim)
        )

    async def claim_start_in_transaction(
        self,
        transaction: StateTransaction,
        claim: ExecutionStartClaim,
    ) -> ExecutionRecord:
        execution_key = self._key("execution", (claim.execution_id,))
        identity = self._idempotency._identity_key(
            claim.scope,
            claim.idempotency_key,
        )
        idempotency_key = self._idempotency._key("idempotency", identity)
        stored = await transaction.get_records((execution_key, idempotency_key))
        execution_record = stored.get(execution_key)
        if execution_record is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        current = await self._decode(execution_record, ExecutionRecord)
        if (
            current.revision != claim.expected_revision
            or current.event_seq != claim.expected_event_seq
            or current.status is not ExecutionStatus.PENDING_START
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        idempotency_record = stored.get(idempotency_key)
        idempotency_created = idempotency_record is None
        if idempotency_record is None:
            idempotency = IdempotencyRecord(
                scope=claim.scope,
                idempotency_key=claim.idempotency_key,
                request_digest=claim.request_digest,
                resource_kind=ResourceKind.EXECUTION,
                resource_id=claim.execution_id,
                status=IdempotencyStatus.STARTED,
                result_digest=None,
                error_code=None,
                created_at=claim.started_at,
                updated_at=claim.started_at,
                request_identity_version=claim.request_identity_version,
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
                        scope=claim.scope,
                        idempotency_key=claim.idempotency_key,
                        request_digest=claim.request_digest,
                        resource_kind=ResourceKind.EXECUTION,
                        resource_id=claim.execution_id,
                        status=idempotency.status,
                        result_digest=idempotency.result_digest,
                        error_code=idempotency.error_code,
                        created_at=idempotency.created_at,
                        updated_at=idempotency.updated_at,
                        request_identity_version=idempotency.request_identity_version,
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
            event_seq=current.event_seq + 1,
            updated_at=now,
            started_at=current.started_at or now,
            recovery=current.recovery or (ExecutionRecoveryDetail() if current.binding_kind == "agent" else None),
        )
        execution_replacement = RecordReplacement(
            await self._projected_content(execution_record, next_execution),
            execution_record.storage_revision,
        )
        if idempotency_created:
            await transaction.insert_records((idempotency_record,))
            await _replace_checked(
                transaction,
                execution_replacement.record,
                execution_replacement.expected_storage_revision,
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
                    idempotency_record.storage_revision,
                )
            )
            await transaction.replace_records(tuple(replacements))
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            (claim.execution_id,),
        )
        await transaction.insert_facts(
            (
                StoredFact(
                    stream,
                    next_execution.event_seq,
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
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_seq: int,
        producer_claim_id: str | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            return await self.claim_next_agent_run_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                expected_agent_run_seq=expected_agent_run_seq,
                producer_claim_id=producer_claim_id,
            )

        return await self._store.mutate(mutate)

    async def claim_next_agent_run_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_agent_run_seq: int,
        producer_claim_id: str | None = None,
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", (execution_id,)))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ExecutionRecord)
        if (
            current.status is not ExecutionStatus.STARTED
            or current.revision != expected_revision
            or current.agent_run_seq != expected_agent_run_seq
            or current.revision + 1 <= current.producer_generation
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            agent_run_seq=current.agent_run_seq + 1,
            recovery=current.recovery if recovery is None else recovery,
            producer_generation=current.revision + 1,
            producer_claim_id=producer_claim_id,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            await self._projected_content(record, next_value),
            record.storage_revision,
        )
        return next_value

    async def claim_prepared_agent_run_in_transaction(
        self, transaction: StateTransaction, execution_id: str, *, tenant_id: str,
        expected_revision: int, expected_agent_run_seq: int,
        input_context: RuntimePayloadRef, producer_claim_id: str,
        recovery: ExecutionRecoveryDetail | None = None,
        observed: Observed[ExecutionRecord] | None = None,
    ) -> ExecutionRecord:
        """Claim the first attempt and its already materialized input together."""
        _require_repository_tenant(tenant_id, self._tenant_id)
        if observed is None:
            record = await transaction.get_record(self._key("execution", (execution_id,)))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ExecutionRecord)
        else:
            current = observed.value
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            record = replace(await self._stored_content("execution", (execution_id,), current, state=current.status.value, previous_value=current),
                             storage_revision=observed.storage_revision)
        if (current.status is not ExecutionStatus.STARTED
                or current.revision != expected_revision
                or current.agent_run_seq != expected_agent_run_seq
                or expected_agent_run_seq != 0 or current.input_context is not None
                or current.revision + 1 <= current.producer_generation):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        admitted = replace(current, agent_run_seq=1, revision=current.revision + 1,
                           producer_generation=current.revision + 1, producer_claim_id=producer_claim_id,
                           updated_at=await transaction.now(), recovery=current.recovery if recovery is None else recovery)
        final = replace(admitted, input_context=input_context, repository_instructions=None,
                        revision=admitted.revision + 1)
        await _replace_checked(transaction, await self._projected_content(record, final), record.storage_revision)
        return final

    async def enter_deferred_wait_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_seq: int,
        expected_agent_run_seq: int,
        audit_events: Sequence[ExecutionEventAppend] = (),
        deferred_events: Sequence[ExecutionEventAppend],
        occurred_at: datetime,
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        if not deferred_events or any(
            event.event_type
            not in {
                ExecutionEventType.APPROVAL_REQUESTED,
                ExecutionEventType.EXTERNAL_REQUESTED,
            }
            for event in deferred_events
        ):
            raise ValueError("deferred wait requires deferred events")
        if occurred_at.tzinfo is None:
            raise ValueError("deferred wait timestamp must be timezone-aware")
        current = await self.get_in_transaction(
            transaction,
            execution_id,
            tenant_id=tenant_id,
        )
        if (
            current is None
            or current.status is not ExecutionStatus.STARTED
            or current.revision != expected_revision
            or current.event_seq != expected_event_seq
            or current.agent_run_seq != expected_agent_run_seq
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        last = deferred_events[-1]
        updated = await self._transition_execution(
            execution_id,
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            expected_event_seq=expected_event_seq,
            expected_status=ExecutionStatus.STARTED,
            next_status=ExecutionStatus.WAITING_DEFERRED,
            pending_events=(*audit_events, *deferred_events[:-1]),
            event_type=last.event_type,
            payload=last.payload,
            updated_at=occurred_at,
            transaction=transaction,
            recovery=recovery,
        )
        if updated.agent_run_seq != expected_agent_run_seq:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return updated

    async def claim_deferred_resume_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_seq: int,
        expected_agent_run_seq: int,
        producer_claim_id: str | None = None,
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._key("execution", (execution_id,)))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ExecutionRecord)
        if (
            current.status is not ExecutionStatus.WAITING_DEFERRED
            or current.revision != expected_revision
            or current.event_seq != expected_event_seq
            or current.agent_run_seq != expected_agent_run_seq
            or current.revision + 1 <= current.producer_generation
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_value = replace(
            current,
            status=ExecutionStatus.STARTED,
            agent_run_seq=current.agent_run_seq + 1,
            recovery=current.recovery if recovery is None else recovery,
            producer_generation=current.revision + 1,
            producer_claim_id=producer_claim_id,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        await _replace_checked(
            transaction,
            await self._projected_content(record, next_value),
            record.storage_revision,
        )
        return next_value

    async def transition_task_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_seq: int,
        expected_status: ExecutionStatus,
        next_status: ExecutionStatus,
        task_attempt: int,
        task_deadline_at: datetime | None,
        task_next_attempt_at: datetime | None,
        error_code: str | None,
        safe_error_details: Mapping[str, JsonValue],
        event_type: str,
        payload: Mapping[str, JsonValue],
        occurred_at: datetime,
    ) -> ExecutionRecord:
        if (
            isinstance(task_attempt, bool)
            or not isinstance(task_attempt, int)
            or task_attempt < 0
            or occurred_at.tzinfo is None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._transition_execution(
            execution_id,
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            expected_event_seq=expected_event_seq,
            expected_status=expected_status,
            next_status=next_status,
            event_type=event_type,
            payload=payload,
            updated_at=occurred_at,
            task_state=(
                task_attempt,
                task_deadline_at,
                task_next_attempt_at,
                error_code,
                safe_error_details,
            ),
        )

    async def mark_start_unknown(
        self, commit: ExecutionStartUnknownCommit
    ) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=self._tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_seq=commit.expected_event_seq,
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
        return await self._store.mutate(
            lambda transaction: self.request_cancel_in_transaction(
                transaction,
                commit,
                expected_status=None,
                pending_events=pending_events,
            )
        )

    async def request_cancel_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus | None = None,
        pending_events: Sequence[ExecutionEventAppend] = (),
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        return await self._transition_execution(
            commit.execution_id,
            tenant_id=self._tenant_id,
            expected_revision=commit.expected_revision,
            expected_event_seq=commit.expected_event_seq,
            expected_status=expected_status,
            next_status=ExecutionStatus.CANCELLING,
            event_type=ExecutionEventType.CANCEL_REQUESTED,
            payload={"operation_id": commit.operation_id},
            updated_at=commit.requested_at,
            pending_events=pending_events,
            transaction=transaction,
            recovery=recovery,
        )

    async def advance_event_seq(
        self, execution_id: str, *, tenant_id: str, expected_event_seq: int
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            record = await transaction.get_record(self._key("execution", (execution_id,)))
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(record, ExecutionRecord)
            if current.event_seq != expected_event_seq:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            next_value = replace(
                current,
                event_seq=current.event_seq + 1,
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                await self._projected_content(record, next_value),
                record.storage_revision,
            )
            return next_value

        return await self._store.mutate(mutate)

    async def _transition_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        expected_event_seq: int,
        next_status: ExecutionStatus,
        expected_status: ExecutionStatus | None = None,
        event_type: str,
        payload: Mapping[str, object],
        updated_at: datetime,
        pending_events: Sequence[ExecutionEventAppend] = (),
        task_state: "tuple[int, datetime | None, datetime | None, str | None, Mapping[str, object]] | None" = None,
        transaction: StateTransaction | None = None,
        recovery: ExecutionRecoveryDetail | None = None,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if any(
            not isinstance(event.event_type, str)
            for event in pending_events
        ):
            raise TypeError("pending execution events require a string event type")
        key = self._key("execution", (execution_id,))
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            (execution_id,),
        )

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored_value = await self._decode(stored, ExecutionRecord)
            if (
                stored_value.revision != expected_revision
                or stored_value.event_seq != expected_event_seq
                or expected_status is not None
                and stored_value.status is not expected_status
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            event_count = len(pending_events) + 1
            task_updates: dict[str, object] = {}
            if task_state is not None:
                if not isinstance(stored_value.binding, TaskBindingContract):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                (
                    task_attempt,
                    task_deadline_at,
                    task_next_attempt_at,
                    error_code,
                    safe_error_details,
                ) = task_state
                task_updates = {
                    "task_attempt": task_attempt,
                    "task_deadline_at": task_deadline_at,
                    "task_next_attempt_at": task_next_attempt_at,
                    "error_code": error_code,
                    "safe_error_details": dict(safe_error_details),
                }
            next_value = replace(
                stored_value,
                status=next_status,
                recovery=stored_value.recovery if recovery is None else recovery,
                revision=stored_value.revision + event_count,
                event_seq=stored_value.event_seq + event_count,
                updated_at=updated_at,
                **task_updates,
            )
            candidate = await self._projected_content(stored, next_value)
            await _replace_checked(transaction, candidate, stored.storage_revision)
            first_sequence = stored_value.event_seq + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_value.event_seq,
                    key,
                    str(event_type),
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

    def start_idempotency_record_key(self, identity: IdempotencyRecord) -> bytes:
        """Physical key of the request identity confirmed by admission."""
        return self._idempotency._key("idempotency", self._idempotency._identity_key(
            identity.scope, identity.idempotency_key,
        ))

    async def terminal_idempotency_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
        *, identity: IdempotencyRecord | None = None,
    ) -> IdempotencyTerminalUpdate | None:
        """Build the terminal idempotency update from the active transaction."""
        if identity is None:
            scope = self._idempotency._scope(
                "idempotency", "resource",
                [ResourceKind.EXECUTION.value, commit.execution.execution_id],
            )
            records = await transaction.list_records(RecordQuery(scope_key=scope, kind="idempotency"))
            if len(records) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if not records:
                return None
            identity = await self._idempotency._decode(records[0], IdempotencyRecord)
        elif (identity.resource_kind is not ResourceKind.EXECUTION
              or identity.resource_id != commit.execution.execution_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        next_status = (
            IdempotencyStatus.COMPLETED
            if commit.execution.status is ExecutionStatus.SUCCEEDED
            else IdempotencyStatus.CANCELLED
            if commit.execution.status is ExecutionStatus.CANCELLED
            else IdempotencyStatus.FAILED
        )
        return IdempotencyTerminalUpdate(
            identity.scope,
            identity.idempotency_key,
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
        await self._execution_content.prepare_value(replace(commit.execution, result=commit.result))
        return await self._commit_terminal(commit, pending_events=pending_events)

    async def _commit_terminal(
        self,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
        transaction: StateTransaction | None = None,
        result_source: NativeTextReference | None = None,
    ) -> ExecutionTerminalCommitResult:
        key = self._key("execution", (commit.execution.execution_id,))
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            (commit.execution.execution_id,),
        )

        async def mutate(
            transaction: StateTransaction,
        ) -> ExecutionTerminalCommitResult:
            if any(
                not isinstance(event.event_type, str)
                for event in pending_events
            ):
                raise TypeError("pending execution events require a string event type")
            record_keys = [key]
            id_key = None
            if commit.idempotency is not None:
                identity = self._idempotency._identity_key(
                    commit.idempotency.scope,
                    commit.idempotency.idempotency_key,
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
                or stored_value.event_seq != commit.expected_event_seq
                or stored_value.native_step_source != commit.execution.native_step_source
                or stored_value.restart_checkpoints != commit.execution.restart_checkpoints
                or stored_value.durable_trace != commit.execution.durable_trace
                or stored_value.producer_generation != commit.execution.producer_generation
                or stored_value.producer_claim_id != commit.execution.producer_claim_id
                or (commit.producer_generation is not None
                    and stored_value.producer_generation != commit.producer_generation)
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
                if (id_value.resource_kind is not ResourceKind.EXECUTION
                        or id_value.resource_id != commit.execution.execution_id
                        or id_value.scope != commit.idempotency.scope
                        or id_value.idempotency_key != commit.idempotency.idempotency_key
                        or id_value.request_digest != commit.idempotency.request_digest):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if id_value.status is not commit.idempotency.expected_status:
                    raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            operation_record = None
            current_operation = None
            if commit.operation is not None:
                operation_key_value = operation_key(self._namespace, self._tenant_id, self._domain.value, commit.operation.operation_id)
                operation_record = await transaction.get_operation(operation_key_value)
                if operation_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current_operation = _decode_operation(operation_record, namespace=self._namespace, tenant_id=self._tenant_id, domain=self._domain)
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
                event_seq=stored_value.event_seq + len(pending_events) + 1,
                result=commit.result,
            ).finalize_terminal(commit.result)
            if result_source is not None:
                await self._execution_content.verify_result_source(result_source, commit.result.output)
            replacements = [
                RecordReplacement(
                    (await self._projected_content(stored, next_execution) if result_source is None else
                     await self._projected_content(stored, next_execution, result_source=result_source)),
                    stored.storage_revision,
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
                        id_record.storage_revision,
                    )
                )
            first_sequence = stored_value.event_seq + 1
            facts = [
                StoredFact(
                    stream,
                    first_sequence + index,
                    key,
                    event.event_type,
                    None,
                    None,
                    event.payload,
                )
                for index, event in enumerate(pending_events)
            ]
            facts.append(
                StoredFact(
                    stream,
                    next_execution.event_seq,
                    key,
                    str(commit.terminal_event_type),
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
        result_source: NativeTextReference | None = None,
    ) -> ExecutionTerminalCommitResult:
        return await self._commit_terminal(
            commit,
            pending_events=pending_events,
            transaction=transaction, result_source=result_source,
        )

    async def commit_closed_terminal_in_transaction(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
        *,
        pending_events: Sequence[ExecutionEventAppend] = (),
        result_source: NativeTextReference | None = None,
    ) -> ExecutionTerminalCommitResult:
        if commit.execution.durable_trace:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._commit_terminal(commit, pending_events=pending_events, transaction=transaction, result_source=result_source)

    async def pending_turn_result_in_transaction(
        self, transaction: StateTransaction, reservation: ExecutionStartReservation,
    ) -> ExecutionTerminalCommitResult | None:
        identity = self._idempotency._identity_key(
            reservation.idempotency.scope, reservation.idempotency.idempotency_key,
        )
        stored = await transaction.get_record(self._idempotency._key("idempotency", identity))
        if stored is None:
            return None
        current = await self._idempotency._decode(stored, IdempotencyRecord)
        if not _same_idempotency_identity(current, reservation.idempotency):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        execution = await self.get(current.resource_id, tenant_id=self._tenant_id)
        terminal_statuses = {
            ExecutionStatus.SUCCEEDED: IdempotencyStatus.COMPLETED,
            ExecutionStatus.FAILED: IdempotencyStatus.FAILED,
            ExecutionStatus.CANCELLED: IdempotencyStatus.CANCELLED,
        }
        if (execution is None or execution.status not in terminal_statuses
                or current.status is not terminal_statuses[execution.status]
                or execution.result is None
                or execution.durable_trace != reservation.execution.durable_trace
                or (execution.history_seal_digest is not None) != execution.durable_trace
                or (execution.result.output is not None) != (execution.status is ExecutionStatus.SUCCEEDED)
                or current.result_digest != (None if execution.result.output is None else execution.result.output.digest)
                or current.error_code != execution.error_code):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        return ExecutionTerminalCommitResult(execution, execution.result)

    def pending_turn_record_keys(self, reservation: ExecutionStartReservation) -> tuple[bytes, ...]:
        _require_tenant(reservation.idempotency, self._tenant_id)
        identity = self._idempotency._identity_key(
            reservation.idempotency.scope, reservation.idempotency.idempotency_key,
        )
        return (self._idempotency._key("idempotency", identity),)

    async def insert_pending_turn_in_transaction(
        self, transaction: StateTransaction, reservation: ExecutionStartReservation,
        commit: ExecutionTerminalCommit, *, audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> tuple[ExecutionTerminalCommitResult, ExecutionInsertionReceipt]:
        _require_tenant(reservation.execution, self._tenant_id)
        _require_tenant(reservation.idempotency, self._tenant_id)
        _require_tenant(commit.execution, self._tenant_id)
        self._idempotency._require_resource_kind(reservation.idempotency)
        execution_id = reservation.execution.execution_id
        terminal_states = {
            ExecutionStatus.SUCCEEDED: (ExecutionEventType.EXECUTION_SUCCEEDED, IdempotencyStatus.COMPLETED),
            ExecutionStatus.FAILED: (ExecutionEventType.EXECUTION_FAILED, IdempotencyStatus.FAILED),
            ExecutionStatus.CANCELLED: (ExecutionEventType.EXECUTION_CANCELLED, IdempotencyStatus.CANCELLED),
        }
        if (not reservation.execution.conversation or not commit.execution.conversation
                or commit.execution.execution_id != execution_id
                or reservation.idempotency.resource_id != execution_id
                or reservation.idempotency.status is not IdempotencyStatus.RESERVED
                or commit.execution.status not in terminal_states
                or commit.terminal_event_type != terminal_states[commit.execution.status][0]
                or commit.operation is not None
                or (commit.result.output is not None) != (commit.execution.status is ExecutionStatus.SUCCEEDED)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for name in ("session_id", "binding_digest", "principal_id", "principal_kind", "stored_user_input", "binding",
                     "parent_execution_id", "parent_invocation_id", "root_execution_id", "previous_execution_id",
                     "fork_base_execution_id", "lineage_kind", "created_at", "mode", "planning", "thinking",
                     "memory_scope", "repository_instructions", "context_imported", "correlation",
                     "requires_task_invocation_capture", "budget_scope_id", "native_step_source", "restart_checkpoints", "durable_trace", "tool_effects", "session_context",
                     "conversation"):
            if getattr(reservation.execution, name) != getattr(commit.execution, name):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = (ExecutionEventAppend(ExecutionEventType.EXECUTION_STARTED, {}), *audit_events,
                  ExecutionEventAppend(commit.terminal_event_type, commit.terminal_event_payload))
        if any(event.event_type not in {ExecutionEventType.MODEL_REQUEST_STARTED,
                                        ExecutionEventType.MODEL_REQUEST_FINISHED,
                                        ExecutionEventType.ASSISTANT_PART_COMPLETED,
                                        ExecutionEventType.TOOL_CALL_STARTED,
                                        ExecutionEventType.TOOL_CALL_FINISHED} for event in audit_events):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        # Pending turns opt out of the post-terminal dependency admission window.
        final = replace(commit.execution, result=commit.result, event_seq=len(events), retention_closed=True,
                        revision=commit.expected_revision + len(audit_events) + 1).finalize_terminal(commit.result)
        identity = self._idempotency._identity_key(
            reservation.idempotency.scope, reservation.idempotency.idempotency_key,
        )
        completed = replace(reservation.idempotency, status=terminal_states[commit.execution.status][1],
                            result_digest=None if commit.result.output is None else commit.result.output.digest,
                            error_code=commit.execution.error_code,
                            updated_at=await transaction.now())
        execution_record = await self._stored_content("execution", (execution_id,), final, state=final.status.value)
        await transaction.insert_records((
            execution_record,
            self._idempotency._stored("idempotency", identity, completed, state=completed.status.value),
        ))
        stream = stream_key(self._namespace, self._tenant_id, self._domain.value, "execution", (execution_id,))
        await transaction.insert_facts(tuple(
            StoredFact(stream, index, self._key("execution", (execution_id,)), str(event.event_type),
                       None, None, event.payload)
            for index, event in enumerate(events, 1)
        ))
        return (ExecutionTerminalCommitResult(final, commit.result),
                ExecutionInsertionReceipt(transaction, final, execution_record))

    async def get_result(
        self, execution_id: str, *, tenant_id: str
    ) -> ResultRecord | None:
        execution = await self.get(execution_id, tenant_id=tenant_id)
        return None if execution is None else execution.result

    async def acquire_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> bool:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not isinstance(hold_id, str) or not hold_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = self._key("execution", (execution_id,))

        async def mutate(transaction: StateTransaction) -> bool:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(stored, ExecutionRecord)
            if current.retention_closed:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if hold_id in current.dependency_hold_ids:
                return False
            next_value = replace(
                current,
                dependency_hold_ids=tuple(
                    sorted((*current.dependency_hold_ids, hold_id))
                ),
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                await self._projected_content(stored, next_value),
                stored.storage_revision,
            )
            return True

        return await self._store.mutate(mutate)

    async def release_dependency_hold(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        hold_id: str,
    ) -> ExecutionRecord:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not isinstance(hold_id, str) or not hold_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        key = self._key("execution", (execution_id,))

        async def mutate(transaction: StateTransaction) -> ExecutionRecord:
            stored = await transaction.get_record(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = await self._decode(stored, ExecutionRecord)
            if hold_id not in current.dependency_hold_ids:
                return current
            next_value = replace(
                current,
                dependency_hold_ids=tuple(
                    value
                    for value in current.dependency_hold_ids
                    if value != hold_id
                ),
                revision=current.revision + 1,
                updated_at=await transaction.now(),
            )
            await _replace_checked(
                transaction,
                await self._projected_content(stored, next_value),
                stored.storage_revision,
            )
            return next_value

        return await self._store.mutate(mutate)

    async def close_retention(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> bool:
        released, _execution = await self._store.mutate(
            lambda transaction: self.close_retention_in_transaction(
                transaction, execution_id, tenant_id=tenant_id,
            )
        )
        return released

    async def close_retention_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        observed: Observed[ExecutionRecord] | None = None,
    ) -> tuple[bool, ExecutionRetentionRecord | None]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = self._key("execution", (execution_id,))
        if observed is None:
            stored = await transaction.get_record(key)
            if stored is None:
                return True, None
            self._decode_header(stored, execution_id)
            current = decode_execution_retention(stored.data)
        else:
            current = observed.value
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored = replace(await self._stored_content("execution", (execution_id,), current, state=current.status.value, previous_value=current),
                             storage_revision=observed.storage_revision)
            current = decode_execution_retention(stored.data)
        if current.status not in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        } or current.dependency_hold_ids or current.recovery_handoff:
            return False, current
        if current.retention_closed:
            return True, current
        next_value = replace(
            current,
            retention_closed=True,
            revision=current.revision + 1,
            updated_at=await transaction.now(),
        )
        envelope = decode_envelope(stored.data)
        payload = envelope.value["payload"]
        data = encode_envelope({**envelope.value, "payload": {**payload, "fields": {
            **payload["fields"], "retention_closed": True, "revision": next_value.revision,
            "updated_at": encode_domain(next_value.updated_at),
        }}})
        await _replace_checked(transaction, replace(stored, data=data,
            storage_revision=stored.storage_revision + 1), stored.storage_revision)
        return True, next_value

    async def require_active_producer(
        self, execution_id: str, *, tenant_id: str, expected_producer_generation: int,
        allow_cancelling: bool = True, fresh: bool = False,
    ) -> None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        key = self._key("execution", (execution_id,))
        record = ((await self._refresh_records((key,))).get(key) if fresh else
                  await self._store.read_record(key))
        if record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._decode_header(record, execution_id)
        generation, status, _durable_trace, _seal = decode_execution_producer_control(record.data)
        if generation != expected_producer_generation:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if status is ExecutionStatus.CANCELLING and not allow_cancelling:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        allowed = ({ExecutionStatus.STARTED, ExecutionStatus.CANCELLING, ExecutionStatus.FINALIZING}
                   if allow_cancelling else {ExecutionStatus.STARTED})
        if status not in allowed:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def require_revoked_producer(
        self, execution_id: str, *, tenant_id: str, producer_generation: int,
    ) -> None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        async def read(transaction: StateTransaction) -> None:
            record = await transaction.get_record(self._key("execution", (execution_id,)))
            if record is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._decode_header(record, execution_id)
            generation, status, durable_trace, seal_digest = decode_execution_producer_control(record.data)
            if generation > producer_generation:
                return
            if generation == producer_generation and status is ExecutionStatus.RECOVERY_REQUIRED:
                return
            if (generation < producer_generation or status not in {
                    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if not durable_trace:
                return
            seal = await self._get_history_seal_in_transaction(transaction, execution_id)
            if seal is None or seal.seal_digest != seal_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._store.read(read)

    async def get_history_seal(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionHistorySealRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)

        async def read(
            transaction: StateTransaction,
        ) -> ExecutionHistorySealRecord | None:
            return await self._get_history_seal_in_transaction(
                transaction, execution_id
            )

        return await self._store.read(read)

    async def _get_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
    ) -> ExecutionHistorySealRecord | None:
        record = await transaction.get_record(
            self._key("execution_history_seal", (execution_id,))
        )
        if record is None:
            return None
        if record.kind != "execution_history_seal":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._decode(record, ExecutionHistorySealRecord)
        if value.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def require_execution_producer_in_transaction(
        self, transaction: StateTransaction, execution_id: str, *,
        expected_producer_generation: int, allow_cancelling: bool = True,
    ) -> tuple[ExecutionControlRecord, StoredRecord]:
        """Fence publication against the sole execution control owner."""
        if (isinstance(expected_producer_generation, bool)
                or not isinstance(expected_producer_generation, int)
                or expected_producer_generation < 1):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        key = self._key("execution", (execution_id,))
        record = await transaction.get_record(key)
        if record is None or record.kind != "execution":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = decode_execution_control(record.data)
        if execution.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.producer_generation != expected_producer_generation:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if execution.status is ExecutionStatus.CANCELLING and not allow_cancelling:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        allowed = ({ExecutionStatus.STARTED, ExecutionStatus.CANCELLING, ExecutionStatus.FINALIZING}
                   if allow_cancelling else {ExecutionStatus.STARTED})
        if execution.status not in allowed:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        guarded = await transaction.guard_record(
            key, expected_storage_revision=record.storage_revision,
        )
        if guarded is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return execution, guarded

    async def require_execution_history_write_in_transaction(
        self, transaction: StateTransaction, execution_id: str,
    ) -> tuple[ExecutionControlRecord, StoredRecord]:
        key = self._key("execution", (execution_id,))
        record = await transaction.get_record(key)
        if record is None or record.kind != "execution":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution = decode_execution_control(record.data)
        if execution.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.history_seal_digest is not None or execution.status in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        guarded = await transaction.guard_record(key, expected_storage_revision=record.storage_revision)
        if guarded is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return execution, guarded

    async def put_history_seal_in_transaction(
        self,
        transaction: StateTransaction,
        seal: ExecutionHistorySealRecord,
    ) -> ExecutionHistorySealRecord:
        key = self._key("execution_history_seal", (seal.execution_id,))
        current = await transaction.get_record(key)
        if current is not None:
            existing = await self._decode(current, ExecutionHistorySealRecord)
            if existing != seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        await transaction.insert_record(
            await self._stored_content(
                "execution_history_seal",
                (seal.execution_id,),
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
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=PersistenceDomain.EXECUTION,
        )

    async def append_many(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_event_seq: int | None = None,
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
                expected_event_seq=expected_event_seq,
            )
        )
        _logger.debug(
            "execution events appended: execution=%s count=%s last_sequence=%s",
            execution_id,
            len(events),
            result[-1].event_seq,
        )
        return result

    async def append_many_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
        events: Sequence[ExecutionEventAppend],
        expected_event_seq: int | None = None,
    ) -> tuple[ExecutionEventRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        if not events:
            return ()
        if any(
            not isinstance(event.event_type, str) for event in events
        ):
            raise TypeError("event repository requires a string event type")
        if any(not isinstance(event.payload, Mapping) for event in events):
            raise TypeError("event payload must be a mapping")
        key = self._key("execution", (execution_id,))
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            (execution_id,),
        )
        current = await transaction.get_record(key)
        if current is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        execution = await self._decode(current, ExecutionRecord)
        if (
            expected_event_seq is not None
            and execution.event_seq != expected_event_seq
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        now = await transaction.now()
        first_sequence = execution.event_seq + 1
        last_sequence = execution.event_seq + len(events)
        next_execution = replace(
            execution,
            event_seq=last_sequence,
            revision=execution.revision + len(events),
            updated_at=now,
        )
        await _replace_checked(
            transaction,
            await self._projected_content(current, next_execution),
            current.storage_revision,
        )
        facts = tuple(
            StoredFact(
                stream,
                first_sequence + index,
                key,
                event.event_type,
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
        event_type: str,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_event_seq=None,
        )
        return values[0]

    async def append_expected(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_event_seq: int,
        event_type: str,
        payload: object,
    ) -> ExecutionEventRecord:
        values = await self.append_many(
            execution_id,
            tenant_id=tenant_id,
            events=(ExecutionEventAppend(event_type, _event_payload(payload)),),
            expected_event_seq=expected_event_seq,
        )
        return values[0]

    async def list(
        self, execution_id: str, *, tenant_id: str, after_event_seq: int, limit: int
    ) -> Page[ExecutionEventRecord]:
        if tenant_id != self._tenant_id:
            return Page(())
        _validate_page_limit(limit)
        stream = stream_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            "execution",
            (execution_id,),
        )
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    stream,
                    after_sequence=after_event_seq,
                    limit=min(limit + 1, 1000),
                )
            )
        )
        if limit == 1000 and len(values) == 1000:
            extra = await self._store.read(
                lambda transaction: transaction.list_facts(
                    FactQuery(
                        stream,
                        after_sequence=values[-1].sequence,
                        limit=1,
                    )
                )
            )
            if extra:
                values = (*values, extra[0])
        items = tuple(
            ExecutionEventRecord(
                execution_id,
                value.sequence,
                value.kind,
                value.data,
            )
            for value in values[:limit]
        )
        return Page(
            items, str(items[-1].event_seq) if len(values) > limit and items else None
        )


def _event_payload(value: object) -> Mapping[str, JsonValue]:
    if isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return {"value": value}  # type: ignore[dict-item]


def _same_idempotency(left: IdempotencyRecord, right: IdempotencyRecord) -> bool:
    return (
        _same_idempotency_identity(left, right)
        and left.resource_id == right.resource_id
    )


def _same_idempotency_identity(
    left: IdempotencyRecord, right: IdempotencyRecord
) -> bool:
    """Compare only the immutable request identity, never a candidate resource id."""
    return (
        left.scope == right.scope
        and left.idempotency_key == right.idempotency_key
        and left.request_digest == right.request_digest
        and left.request_identity_version == right.request_identity_version
        and left.resource_kind is right.resource_kind
    )


__all__ = [
    "EventRepositoryImpl",
    "ExecutionRepositoryImpl",
    "IdempotencyRepositoryImpl",
]
