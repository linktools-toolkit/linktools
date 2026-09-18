#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery-domain repository implementations."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from linktools.core import environ
from ...core import ApprovalDecision, ApprovalStatus, ExternalCallStatus, JsonValue, Page, ResourceKind, ToolOperationStatus, validate_lease_owner, validate_lease_seconds
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from ._contracts import ToolOperationRecord
from ._contracts import ApprovalRecord, ExternalCallRecord, RecoveryCheckpoint, RecoveryCheckpointState, ToolOperationAdmission, validate_tool_operation_failure
from ._plan import RuntimeDomain
from ._store import (
    RecordReplacement,
    StateStore,
    StateTransaction,
    StoredAlias,
    StoredRecord,
    alias_digest,
)
from ._repository_common import (
    OperationLedgerRepository,
    RepositoryBase as _RepositoryBase,
    ResourceRepository as _ResourceRepository,
    projected_record as _projected_record,
    record_cursor as _record_cursor,
    record_state as _record_state,
    replace_checked as _replace_checked,
    require_repository_tenant as _require_repository_tenant,
    require_tenant as _require_tenant,
    validate_page_limit as _validate_page_limit,
)

_logger = environ.get_logger("ai.runtime.state.repositories")


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
        decision_message: str | None = None,
        resolution_metadata: Mapping[str, JsonValue] | None = None,
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
                status=ApprovalStatus.APPROVED
                if decision is ApprovalDecision.APPROVE
                else ApprovalStatus.DENIED,
                idempotency_key_digest=idempotency_key_digest,
                decision=decision,
                decided_by=principal_id,
                decision_digest=decision_digest,
                decided_at=decided_at,
                decision_message=decision_message,
                resolution_metadata=({} if resolution_metadata is None else resolution_metadata),
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, value),
                record.storage_version,
            )
            return value

        return await self._store.mutate(mutate)

    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        approval_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        decided_at: datetime,
    ) -> tuple[ApprovalRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        values: list[ApprovalRecord] = []
        for approval_id in approval_ids:
            record = await self.get_in_transaction(
                transaction,
                approval_id,
                tenant_id=tenant_id,
            )
            if record is None or record.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored = await transaction.get_record(self._key("approval", approval_id))
            if stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is not ApprovalStatus.PENDING:
                values.append(record)
                continue
            value = replace(record, status=ApprovalStatus.CANCELLED, decided_at=decided_at)
            await _replace_checked(
                transaction,
                _projected_record(self, stored, value),
                stored.storage_version,
            )
            values.append(value)
        return tuple(values)

    async def list_pending(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ApprovalRecord, ...]:
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
        resolution_kind: str,
        result_payload: StoredPayload | None,
        resolution_metadata: Mapping[str, JsonValue],
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
                resolution_kind=resolution_kind,
                result_payload=result_payload,
                resolution_metadata=resolution_metadata,
                supplied_at=supplied_at,
            )
            await _replace_checked(
                transaction,
                _projected_record(self, record, value),
                record.storage_version,
            )
            return value

        return await self._store.mutate(mutate)

    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        call_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        cancelled_at: datetime,
    ) -> tuple[ExternalCallRecord, ...]:
        _require_repository_tenant(tenant_id, self._tenant_id)
        values: list[ExternalCallRecord] = []
        for call_id in call_ids:
            record = await self.get_in_transaction(
                transaction,
                call_id,
                tenant_id=tenant_id,
            )
            if record is None or record.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            stored = await transaction.get_record(self._key("external_call", call_id))
            if stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is not ExternalCallStatus.PENDING:
                values.append(record)
                continue
            value = replace(record, status=ExternalCallStatus.CANCELLED, supplied_at=cancelled_at)
            await _replace_checked(
                transaction,
                _projected_record(self, stored, value),
                stored.storage_version,
            )
            values.append(value)
        return tuple(values)

    async def list_pending(
        self, execution_id: str, *, tenant_id: str
    ) -> tuple[ExternalCallRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "external_call",
            scope=self._scope("external_call", "execution", execution_id),
            states=frozenset({ExternalCallStatus.PENDING.value}),
        )
        return tuple(
            [await self._decode(record, ExternalCallRecord) for record in records]
        )


class _RecoveryCheckpointRepository(_ResourceRepository[RecoveryCheckpoint]):
    """Store one immutable recovery frontier per execution."""

    _KIND = "recovery_checkpoint"

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
            kind=self._KIND,
            resource_kind=ResourceKind.EXECUTION,
            value_type=RecoveryCheckpoint,
        )

    async def list(self, *, tenant_id: str) -> tuple[RecoveryCheckpoint, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(self._KIND)
        return tuple(
            [await self._decode(record, RecoveryCheckpoint) for record in records]
        )

    async def list_recoverable_page(
        self,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[RecoveryCheckpoint]:
        if tenant_id != self._tenant_id:
            return Page(())
        _validate_page_limit(limit)
        records = await self._records(
            self._KIND,
            states=frozenset(
                {
                    RecoveryCheckpointState.ADMITTED.value,
                    RecoveryCheckpointState.ACTIVE.value,
                    RecoveryCheckpointState.WAITING.value,
                    RecoveryCheckpointState.HANDOFF.value,
                }
            ),
            cursor=cursor,
            limit=min(limit + 1, 1001),
        )
        selected = records[:limit]
        values = tuple(
            [await self._decode(record, RecoveryCheckpoint) for record in selected]
        )
        next_cursor = _record_cursor(selected[-1]) if len(records) > limit else None
        return Page(values, next_cursor)

    async def get(
        self, execution_id: str, *, tenant_id: str
    ) -> RecoveryCheckpoint | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._key(self._KIND, execution_id))
        return None if record is None else await self._decode(record, RecoveryCheckpoint)

    async def get_in_transaction(
        self,
        transaction: StateTransaction,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None:
        if tenant_id != self._tenant_id:
            return None
        record = await transaction.get_record(self._key(self._KIND, execution_id))
        return None if record is None else await self._decode(record, RecoveryCheckpoint)

    async def create(self, record: RecoveryCheckpoint) -> RecoveryCheckpoint:
        return await self._store.mutate(
            lambda transaction: self.admit_in_transaction(transaction, record)
        )

    async def admit_in_transaction(
        self,
        transaction: StateTransaction,
        record: RecoveryCheckpoint,
    ) -> RecoveryCheckpoint:
        _require_tenant(record, self._tenant_id)
        key = self._key(self._KIND, record.execution_id)
        current = await transaction.get_record(key)
        if current is not None:
            existing = await self._decode(current, RecoveryCheckpoint)
            if existing != record:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return existing
        await transaction.insert_record(
            self._stored(
                self._KIND,
                record.execution_id,
                record,
                state=_record_state(record),
            )
        )
        return record

    async def compare_and_swap(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: RecoveryCheckpoint,
    ) -> RecoveryCheckpoint:
        return await self._store.mutate(
            lambda transaction: self.compare_and_swap_in_transaction(
                transaction,
                execution_id,
                tenant_id=tenant_id,
                expected_revision=expected_revision,
                next_record=next_record,
            )
        )

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
        key = self._key(self._KIND, execution_id)
        current_record = await transaction.get_record(key)
        if current_record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(current_record, RecoveryCheckpoint)
        if current.revision != expected_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if next_record.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await _replace_checked(
            transaction,
            replace(
                self._stored(self._KIND, execution_id, next_record),
                storage_version=current_record.storage_version + 1,
            ),
            current_record.storage_version,
        )
        return next_record


class ToolRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.RECOVERY,
        )

    def _tool_key(self, identity: str) -> bytes:
        return self._key("tool_operation", identity)

    async def _retry_storage_conflict(
        self,
        operation: Callable[[], Awaitable[ToolOperationRecord]],
    ) -> ToolOperationRecord:
        while True:
            try:
                return await operation()
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
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

        return await self._retry_storage_conflict(attempt)

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
            record_keys = tuple(
                dict.fromkeys(
                    (*resolved.values(), self._tool_key(request.tool_operation_id))
                )
            )
            records = await transaction.get_records(record_keys)
            resolved_keys = tuple(dict.fromkeys(resolved.values()))
            if len(resolved_keys) > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            candidate_key = self._tool_key(request.tool_operation_id)
            candidate_record = records.get(candidate_key)
            if (
                resolved_keys
                and candidate_record is not None
                and candidate_key not in resolved_keys
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            existing_key = resolved_keys[0] if resolved_keys else candidate_key
            record = records.get(existing_key)
            if record is None:
                now = await transaction.now()
                value = ToolOperationRecord(
                    tool_operation_id=request.tool_operation_id,
                    tenant_id=self._tenant_id,
                    execution_id=request.execution_id,
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
                    arguments_payload=request.arguments_payload,
                )
                await transaction.insert_record(
                    self._stored(
                        "tool_operation",
                        request.tool_operation_id,
                        value,
                        scope=self._scope(
                            "tool_operation", "step_run", request.step_run_id
                        ),
                        state=value.status.value,
                    )
                )
                await transaction.insert_aliases(
                    tuple(
                        StoredAlias(alias, self._tool_key(request.tool_operation_id))
                        for alias in aliases
                    )
                )
                return value
            current = await self._decode(record, ToolOperationRecord)
            if not _tool_admission_matches(current, request):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            missing_aliases = tuple(
                alias for alias in aliases if resolved.get(alias) is None
            )
            if missing_aliases:
                guarded = await transaction.guard_record(
                    record.key_digest,
                    expected_storage_version=record.storage_version,
                )
                if guarded is None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                record = guarded
                await transaction.insert_aliases(
                    tuple(
                        StoredAlias(alias, record.key_digest)
                        for alias in missing_aliases
                    )
                )
            now = (
                await transaction.now()
                if current.status
                in {
                    ToolOperationStatus.PENDING,
                    ToolOperationStatus.CLAIMED,
                }
                else None
            )
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
            expired = (
                current.lease_expires_at is not None and current.lease_expires_at <= now
            )
            if current.status is ToolOperationStatus.CLAIMED and not expired:
                if current.owner == request.owner:
                    return current
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if (
                current.status is ToolOperationStatus.CLAIMED
                and not current.replay_safe
            ):
                value = replace(
                    current,
                    status=ToolOperationStatus.EFFECT_UNKNOWN,
                    lease_expires_at=None,
                    updated_at=now,
                )
            elif current.status in {
                ToolOperationStatus.PENDING,
                ToolOperationStatus.CLAIMED,
            }:
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
        if record.status is ToolOperationStatus.FAILED:
            validate_tool_operation_failure(record.error_code, record.error_payload)

        async def attempt() -> ToolOperationRecord:
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
                        await transaction.insert_aliases(
                            (StoredAlias(replay_alias, existing_key),)
                        )
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

        return await self._retry_storage_conflict(attempt)

    async def get_operation(
        self, tool_operation_id: str, *, tenant_id: str
    ) -> ToolOperationRecord | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._tool_key(tool_operation_id))
        return (
            None if record is None else await self._decode(record, ToolOperationRecord)
        )

    async def list_by_execution(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ToolOperationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        records = await self._records(
            "tool_operation",
            parent=self._parent("tool_operation", "execution", execution_id),
        )
        values = tuple(
            [await self._decode(record, ToolOperationRecord) for record in records]
        )
        if any(value.execution_id != execution_id for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def has_by_step_run(
        self,
        step_run_id: str,
        *,
        tenant_id: str,
    ) -> bool:
        if tenant_id != self._tenant_id:
            return False
        records = await self._records(
            "tool_operation",
            scope=self._scope("tool_operation", "step_run", step_run_id),
            limit=1,
        )
        return bool(records)

    async def existing_call_ids(
        self,
        step_run_id: str,
        tool_call_ids: Sequence[str],
        *,
        tenant_id: str,
    ) -> frozenset[str]:
        if tenant_id != self._tenant_id:
            return frozenset()
        if not isinstance(step_run_id, str) or not step_run_id:
            raise ValueError("step_run_id must be a non-empty string")
        if not isinstance(tool_call_ids, Sequence) or isinstance(
            tool_call_ids, (str, bytes)
        ):
            raise TypeError("tool_call_ids must be a sequence")
        ordered = tuple(dict.fromkeys(tool_call_ids))
        if any(not isinstance(value, str) or not value for value in ordered):
            raise ValueError("tool_call_ids must contain non-empty strings")
        if not ordered:
            return frozenset()
        aliases = tuple(
            alias_digest(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                "tool_call",
                [step_run_id, tool_call_id],
            )
            for tool_call_id in ordered
        )

        async def read(transaction: StateTransaction) -> frozenset[str]:
            resolved = await transaction.resolve_aliases(aliases)
            record_keys = tuple(
                dict.fromkeys(key for key in resolved.values() if key is not None)
            )
            records = await transaction.get_records(record_keys) if record_keys else {}
            decoded: dict[bytes, ToolOperationRecord] = {}
            present: set[str] = set()
            for tool_call_id, alias in zip(ordered, aliases, strict=True):
                key = resolved.get(alias)
                if key is None:
                    continue
                stored = records.get(key)
                if stored is None or stored.kind != "tool_operation":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                value = decoded.get(key)
                if value is None:
                    value = await self._decode(stored, ToolOperationRecord)
                    decoded[key] = value
                if (
                    value.tenant_id != self._tenant_id
                    or value.tool_call_id != tool_call_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                present.add(tool_call_id)
            return frozenset(present)

        return await self._store.read(read)

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
            return (
                None
                if stored is None
                else await self._decode(stored, ToolOperationRecord)
            )

        return await self._store.read(read)

    async def list_by_step_run(
        self,
        step_run_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ToolOperationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        if not isinstance(step_run_id, str) or not step_run_id:
            raise ValueError("step_run_id must be a non-empty string")
        records = await self._records(
            "tool_operation",
            scope=self._scope("tool_operation", "step_run", step_run_id),
        )
        values = tuple(
            [await self._decode(record, ToolOperationRecord) for record in records]
        )
        if any(
            value.tenant_id != self._tenant_id or value.step_run_id != step_run_id
            for value in values
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values

    async def claim(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int
    ) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
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
                expired = (
                    current.lease_expires_at is not None
                    and current.lease_expires_at <= now
                )
                if current.status is ToolOperationStatus.CLAIMED and not expired:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                if (
                    current.status is ToolOperationStatus.CLAIMED
                    and not current.replay_safe
                ):
                    unknown = replace(
                        current,
                        status=ToolOperationStatus.EFFECT_UNKNOWN,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                    await self._replace_tool_in_transaction(
                        transaction, record, unknown
                    )
                    return unknown
                if (
                    current.status is not ToolOperationStatus.PENDING
                    and current.status is not ToolOperationStatus.CLAIMED
                ):
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

        return await self._retry_storage_conflict(attempt)

    async def renew(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        lease_seconds: int,
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
        async def attempt() -> ToolOperationRecord:
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

        return await self._retry_storage_conflict(attempt)

    async def fail_payload(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str,
        error_payload: StoredPayload,
    ) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
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

        return await self._retry_storage_conflict(attempt)

    async def defer(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
            return await self._store.mutate(
                lambda transaction: self.defer_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                )
            )

        return await self._retry_storage_conflict(attempt)

    async def defer_in_transaction(
        self,
        transaction: StateTransaction,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        record = await transaction.get_record(self._tool_key(tool_operation_id))
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(record, ToolOperationRecord)
        if current.status is ToolOperationStatus.PENDING:
            if (
                current.owner is None
                and current.lease_expires_at is None
                and current.fence == fence
            ):
                return current
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        now = await transaction.now()
        _require_live_tool_lease(current, owner=owner, fence=fence, now=now)
        value = replace(
            current,
            status=ToolOperationStatus.PENDING,
            owner=None,
            lease_expires_at=None,
            error_code=None,
            error_payload=None,
            result_payload=None,
            updated_at=now,
        )
        await self._replace_tool_in_transaction(transaction, record, value)
        return value

    async def mark_effect_unknown(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        error_code: str | None,
    ) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
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
                    if (
                        current.owner == owner
                        and current.fence == fence
                        and current.error_code == error_code
                    ):
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

        return await self._retry_storage_conflict(attempt)

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
        if terminal_status is ToolOperationStatus.FAILED:
            validate_tool_operation_failure(
                requested_error,
                requested_error_payload,
            )

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
        error_payload: StoredPayload,
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


def _tool_replay_matches(left: ToolOperationRecord, right: ToolOperationRecord) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.execution_id == right.execution_id
        and left.step_run_id == right.step_run_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and _tool_argument_payloads_match(
            left.arguments_payload,
            right.arguments_payload,
        )
        and left.binding_digest == right.binding_digest
        and left.replay_safe == right.replay_safe
    )


def _tool_admission_matches(
    left: ToolOperationRecord, right: ToolOperationAdmission
) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.execution_id == right.execution_id
        and left.tool_operation_id == right.tool_operation_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and _tool_argument_payloads_match(
            left.arguments_payload,
            right.arguments_payload,
        )
        and left.binding_digest == right.binding_digest
        and left.replay_safe is right.replay_safe
        and left.step_run_id in {right.step_run_id, right.recovery_step_run_id}
    )


def _tool_argument_payloads_match(
    left: StoredPayload | None,
    right: StoredPayload | None,
) -> bool:
    if right is None:
        return left is None
    return left is not None and left.digest == right.digest and left.size == right.size


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


_ApprovalRepositoryImpl = ApprovalRepositoryImpl
_ExternalCallRepositoryImpl = ExternalCallRepositoryImpl


class RecoveryApprovalRepositoryImpl(_ApprovalRepositoryImpl):
    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        approval_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        decided_at: datetime,
    ) -> tuple[ApprovalRecord, ...]:
        _validate_cancel_request(approval_ids, tenant_id, self._tenant_id, decided_at)
        ordered = tuple(dict.fromkeys(approval_ids))
        if not ordered:
            return ()
        keys = tuple(self._key("approval", approval_id) for approval_id in ordered)
        stored_records = await transaction.get_records(keys)
        values: list[ApprovalRecord] = []
        replacements: list[RecordReplacement] = []
        for approval_id, key in zip(ordered, keys, strict=True):
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(stored, ApprovalRecord)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is not ApprovalStatus.PENDING:
                values.append(current)
                continue
            value = replace(
                current,
                status=ApprovalStatus.CANCELLED,
                decided_at=decided_at,
            )
            replacements.append(
                RecordReplacement(
                    replace(
                        self._stored(
                            "approval",
                            value.approval_id,
                            value,
                            state=value.status.value,
                        ),
                        storage_version=stored.storage_version + 1,
                    ),
                    stored.storage_version,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
        return tuple(values)


class RecoveryExternalCallRepositoryImpl(_ExternalCallRepositoryImpl):
    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        call_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        cancelled_at: datetime,
    ) -> tuple[ExternalCallRecord, ...]:
        _validate_cancel_request(call_ids, tenant_id, self._tenant_id, cancelled_at)
        ordered = tuple(dict.fromkeys(call_ids))
        if not ordered:
            return ()
        keys = tuple(self._key("external_call", call_id) for call_id in ordered)
        stored_records = await transaction.get_records(keys)
        values: list[ExternalCallRecord] = []
        replacements: list[RecordReplacement] = []
        for call_id, key in zip(ordered, keys, strict=True):
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(stored, ExternalCallRecord)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is not ExternalCallStatus.PENDING:
                values.append(current)
                continue
            value = replace(
                current,
                status=ExternalCallStatus.CANCELLED,
                supplied_at=cancelled_at,
            )
            replacements.append(
                RecordReplacement(
                    replace(
                        self._stored(
                            "external_call",
                            value.call_id,
                            value,
                            state=value.status.value,
                        ),
                        storage_version=stored.storage_version + 1,
                    ),
                    stored.storage_version,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
        return tuple(values)


def build_recovery_repository_bundle(
    store: StateStore,
    *,
    namespace: str,
    tenant_id: str,
) -> dict[str, _RepositoryBase]:
    return {
        "operations": OperationLedgerRepository(store, namespace=namespace, tenant_id=tenant_id, domain=RuntimeDomain.RECOVERY),
        "approvals": RecoveryApprovalRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        "external_calls": RecoveryExternalCallRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        "checkpoints": _RecoveryCheckpointRepository(store, namespace=namespace, tenant_id=tenant_id),
        "tools": ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
    }


def _validate_cancel_request(
    identities: Sequence[str],
    tenant_id: str,
    expected_tenant_id: str,
    timestamp: datetime,
) -> None:
    if tenant_id != expected_tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("identities must be a sequence")
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("identities must contain non-empty strings")
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")


RecoveryCheckpointRepositoryImpl = _RecoveryCheckpointRepository
tool_admission_matches = _tool_admission_matches


__all__ = [
    "ApprovalRepositoryImpl",
    "ExternalCallRepositoryImpl",
    "RecoveryApprovalRepositoryImpl",
    "RecoveryExternalCallRepositoryImpl",
    "RecoveryCheckpointRepositoryImpl",
    "ToolRepositoryImpl",
    "build_recovery_repository_bundle",
    "tool_admission_matches",
]
