#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint control repositories: attempts, approvals and tool results."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TypeVar
from pydantic_ai_harness.step_persistence import ToolEffectRecord
from linktools.core import environ
from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    ExternalCallStatus,
    JsonValue,
    OperationLedgerRecord,
    Page,
    ResourceKind,
    ToolOperationStatus,
    validate_lease_owner,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from .contracts import ToolOperationRecord
from .contracts import (
    ApprovalRecord,
    ExternalCallRecord,
    ToolOperationAdmission,
    decode_tool_operation_failure,
)
from ._plan import PersistenceDomain
from .state import (
    Observed,
    RecordReplacement,
    StateStore,
    StateTransaction,
    StoredAlias,
    StoredRecord,
    alias_digest,
)
from ._repository_common import (
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

_ToolValue = TypeVar("_ToolValue")


class ApprovalRepositoryImpl(_ResourceRepository[ApprovalRecord]):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=PersistenceDomain.CHECKPOINT,
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
                record.storage_revision,
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
                        storage_revision=stored.storage_revision + 1,
                    ),
                    stored.storage_revision,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
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
            domain=PersistenceDomain.CHECKPOINT,
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
                record.storage_revision,
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
                        storage_revision=stored.storage_revision + 1,
                    ),
                    stored.storage_revision,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
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


class ToolRepositoryImpl(_RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=PersistenceDomain.CHECKPOINT,
        )

    def _tool_key(self, identity: str) -> bytes:
        return self._key("tool_operation", identity)

    async def _retry_storage_conflict(
        self,
        operation: Callable[[], Awaitable[_ToolValue]],
    ) -> _ToolValue:
        while True:
            try:
                return await operation()
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                await asyncio.sleep(0)

    async def record_tool_effect(
        self, effect: ToolEffectRecord, *, execution_id: str, tool_operation_id: str,
        binding_digest: str, recovery_agent_run_id: str | None = None,
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
        """Annotate an admitted effect without creating or changing its outcome."""
        async def mutate(transaction: StateTransaction) -> Observed[ToolOperationRecord]:
            if observed is None:
                stored = await transaction.get_record(self._tool_key(tool_operation_id))
                if stored is None and recovery_agent_run_id is not None:
                    alias = alias_digest(self._namespace, self._tenant_id, self._domain.value,
                                         "tool_call", [effect.run_id, effect.tool_call_id])
                    resolved = await transaction.resolve_alias(alias)
                    stored = None if resolved is None else await transaction.get_record(resolved)
                if stored is None:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                current = await self._decode(stored, ToolOperationRecord)
            else:
                current = observed.value
                stored = replace(self._stored("tool_operation", current.tool_operation_id, current,
                                              state=current.status.value), storage_revision=observed.storage_revision)
            if (current.execution_id != execution_id or current.tool_call_id != effect.tool_call_id
                    or current.tool_name != effect.tool_name or current.binding_digest != binding_digest
                    or current.agent_run_id not in {effect.run_id, recovery_agent_run_id}
                    or current.to_tool_effect(effect.run_id).status != effect.status):
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            value = replace(current,
                effect_started_at=current.effect_started_at or effect.started_at,
                effect_idempotency_key=effect.idempotency_key if effect.idempotency_key is not None else current.effect_idempotency_key,
                effect_summary=effect.effect_summary if effect.effect_summary is not None else current.effect_summary,
            )
            if value == current:
                return Observed(current, stored.storage_revision)
            value = replace(value, updated_at=datetime.now(timezone.utc))
            await self._replace_tool_in_transaction(transaction, stored, value)
            return Observed(value, stored.storage_revision + 1)
        return await self._store.mutate(mutate)

    async def get_tool_effect(
        self, *, run_id: str, tool_call_id: str, source_run_id: str | None = None,
    ) -> ToolEffectRecord | None:
        value = await self.get_by_call(run_id, tool_call_id, tenant_id=self._tenant_id)
        if value is None and source_run_id is not None and source_run_id != run_id:
            value = await self.get_by_call(source_run_id, tool_call_id, tenant_id=self._tenant_id)
        return None if value is None else value.to_tool_effect(run_id)

    async def list_unresolved_tool_effects(
        self, *, execution_id: str, run_id: str, source_run_id: str | None = None,
    ) -> list[ToolEffectRecord]:
        values = await self._list_effect_records(execution_id, run_id, source_run_id)
        return [effect for value in values
                if (effect := value.to_tool_effect(run_id)).status == "started"]

    async def _list_effect_records(
        self, execution_id: str, run_id: str, source_run_id: str | None,
    ) -> tuple[ToolOperationRecord, ...]:
        records = await self._records(
            "tool_operation", parent=self._parent("tool_operation", "execution", execution_id),
        )
        values = [await self._decode(stored, ToolOperationRecord) for stored in records]
        if any(value.execution_id != execution_id for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        direct = {run_id, source_run_id} - {None}
        selected = {stored.key_digest for stored, value in zip(records, values, strict=True)
                    if value.agent_run_id in direct}
        remaining = [(stored, value) for stored, value in zip(records, values, strict=True)
                     if stored.key_digest not in selected]
        if remaining:
            aliases = tuple(alias_digest(
                self._namespace, self._tenant_id, self._domain.value,
                "tool_call", [alias_run_id, value.tool_call_id],
            ) for _stored, value in remaining for alias_run_id in direct)
            resolved = await self._store.read(lambda transaction: transaction.resolve_aliases(aliases))
            selected.update(resolved.values())
        return tuple(value for stored, value in zip(records, values, strict=True)
                     if stored.key_digest in selected)

    async def admit(self, request: ToolOperationAdmission) -> ToolOperationRecord:
        async def attempt() -> ToolOperationRecord:
            result = await self._store.mutate(
                lambda transaction: self.admit_in_transaction(transaction, request)
            )
            result = result.value
            _logger.debug(
                "tool operation admitted: operation=%s status=%s fence=%s",
                result.tool_operation_id,
                result.status.value,
                result.fence,
            )
            if result.status is ToolOperationStatus.EFFECT_UNKNOWN:
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
            return result

        return await self._retry_storage_conflict(attempt)

    async def admit_in_transaction(
        self,
        transaction: StateTransaction,
        request: ToolOperationAdmission,
        *, observed: Observed[ToolOperationRecord] | None = None, known_new: bool = False,
    ) -> Observed[ToolOperationRecord]:
        validate_lease_owner(request.owner)
        if known_new and (observed is not None or request.recovery_agent_run_id is not None):
            raise ValueError("known-new admission cannot carry recovery evidence")
        aliases = ((alias_digest(
            self._namespace, self._tenant_id, self._domain.value, "tool_call",
            [request.agent_run_id, request.tool_call_id],
        ),) if request.recovery_agent_run_id is not None
            and request.recovery_agent_run_id != request.agent_run_id else ())

        async def mutate(transaction: StateTransaction) -> Observed[ToolOperationRecord]:
            if known_new:
                record = None
                resolved = {}
            elif observed is not None:
                current = observed.value
                if (current.tool_operation_id != request.tool_operation_id
                        or current.execution_id != request.execution_id
                        or current.agent_run_id not in {request.agent_run_id, request.recovery_agent_run_id}
                        or current.tool_call_id != request.tool_call_id):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                record = replace(self._stored("tool_operation", current.tool_operation_id, current),
                                 storage_revision=observed.storage_revision)
                resolved = {}
            else:
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
                now = datetime.now(timezone.utc)
                value = ToolOperationRecord(
                    tool_operation_id=request.tool_operation_id,
                    execution_id=request.execution_id,
                    agent_run_id=request.agent_run_id,
                    tool_call_id=request.tool_call_id,
                    idempotency_key_digest=request.idempotency_key_digest,
                    tool_name=request.tool_name,
                    arguments_digest=request.arguments_digest,
                    binding_digest=request.binding_digest,
                    replay_safe=request.replay_safe,
                    status=ToolOperationStatus.CLAIMED,
                    owner=request.owner,
                    fence=1,
                    producer_generation=request.producer_generation,
                    error_code=None,
                    created_at=now,
                    updated_at=now,
                    arguments_payload=request.arguments_payload,
                    capability_ref=request.capability_ref,
                )
                await transaction.insert_record(
                    self._stored(
                        "tool_operation",
                        request.tool_operation_id,
                        value,
                        scope=self._scope(
                            "tool_operation", "agent_run", request.agent_run_id
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
                return Observed(value, 0)
            current = await self._decode(record, ToolOperationRecord)
            if not _tool_admission_matches(current, request):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            missing_aliases = tuple(
                alias for alias in aliases if resolved.get(alias) is None
            )
            if missing_aliases:
                guarded = await transaction.guard_record(
                    record.key_digest,
                    expected_storage_revision=record.storage_revision,
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
                datetime.now(timezone.utc)
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
                if current.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                return Observed(current, record.storage_revision)
            if now is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is ToolOperationStatus.CLAIMED:
                if current.owner == request.owner:
                    return Observed(current, record.storage_revision)
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is not ToolOperationStatus.PENDING:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            value = replace(current, status=ToolOperationStatus.CLAIMED,
                            owner=request.owner, fence=current.fence + 1,
                            producer_generation=request.producer_generation, updated_at=now)
            await self._replace_tool_in_transaction(transaction, record, value)
            return Observed(value, record.storage_revision + 1)

        return await mutate(transaction)

    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord:
        if record.status is ToolOperationStatus.FAILED:
            decode_tool_operation_failure(record.error_code, record.error_payload)

        async def attempt() -> ToolOperationRecord:
            _require_tenant(record, self._tenant_id)
            key = self._tool_key(record.tool_operation_id)
            replay_alias = alias_digest(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                "tool_call",
                [record.agent_run_id, record.tool_call_id],
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
                            expected_storage_revision=existing_record.storage_revision,
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
                    scope=self._scope("tool_operation", "agent_run", record.agent_run_id),
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
        value = None if record is None else await self._decode(record, ToolOperationRecord)
        return value

    async def settle_deferred_result_in_transaction(
        self, transaction: StateTransaction, expected: ToolOperationRecord, *,
        result_payload: StoredPayload | None, error_code: str | None,
        error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        stored = await transaction.get_record(self._tool_key(expected.tool_operation_id))
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ToolOperationRecord)
        if not _tool_replay_matches(current, expected) or current.fence != expected.fence:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        target = ToolOperationStatus.COMPLETED if result_payload is not None else ToolOperationStatus.FAILED
        if target is ToolOperationStatus.FAILED:
            decode_tool_operation_failure(error_code, error_payload)
        if current.status is target:
            if (current.result_payload == result_payload and current.error_code == error_code
                    and current.error_payload == error_payload):
                return current
            raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
        if current.status is not ToolOperationStatus.PENDING or current.owner is not None:
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        value = replace(current, status=target, result_payload=result_payload,
                        error_code=error_code, error_payload=error_payload,
                        updated_at=datetime.now(timezone.utc))
        await self._replace_tool_in_transaction(transaction, stored, value)
        return value

    async def reconcile_turn_claim_in_transaction(
        self, transaction: StateTransaction, tool_operation_id: str, *,
        execution_id: str, producer_generation: int, producer_claim_id: str,
        turn_stopping: bool,
    ) -> ToolOperationRecord:
        stored = await transaction.get_record(self._tool_key(tool_operation_id))
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ToolOperationRecord)
        if current.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if current.status is not ToolOperationStatus.CLAIMED:
            return current
        same_turn = (current.owner == producer_claim_id
                     and current.producer_generation == producer_generation)
        if same_turn and not turn_stopping:
            return current
        if (current.producer_generation is None
                or current.producer_generation > producer_generation
                or current.producer_generation == producer_generation and not same_turn):
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        retry = current.replay_safe and not turn_stopping
        value = replace(
            current,
            status=ToolOperationStatus.PENDING if retry else ToolOperationStatus.EFFECT_UNKNOWN,
            owner=None if retry else current.owner,
            error_code=None if retry else ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            updated_at=datetime.now(timezone.utc),
        )
        await self._replace_tool_in_transaction(transaction, stored, value)
        return value

    async def observe_operation(
        self, tool_operation_id: str, *, tenant_id: str,
    ) -> Observed[ToolOperationRecord] | None:
        if tenant_id != self._tenant_id:
            return None
        record = await self._record(self._tool_key(tool_operation_id))
        if record is None:
            return None
        return Observed(await self._decode(record, ToolOperationRecord), record.storage_revision)

    async def get_operation_in_transaction(
        self,
        transaction: StateTransaction,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord | None:
        _require_repository_tenant(tenant_id, self._tenant_id)
        record = await transaction.get_record(self._tool_key(tool_operation_id))
        value = None if record is None else await self._decode(record, ToolOperationRecord)
        return value

    async def resolve_effect_in_transaction(
        self,
        transaction: StateTransaction,
        tool_operation_id: str,
        *,
        execution_id: str,
        expected_fence: int,
        target_status: ToolOperationStatus,
        result_payload: StoredPayload | None,
        error_code: str | None,
        error_payload: StoredPayload | None,
        admit_operation: Callable[[], Awaitable[OperationLedgerRecord]],
    ) -> ToolOperationRecord:
        """Resolve an unknown effect after admitting its receipt in this group transaction."""
        stored = await transaction.get_record(self._tool_key(tool_operation_id))
        if stored is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        current = await self._decode(stored, ToolOperationRecord)
        if current.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            current.status is not ToolOperationStatus.EFFECT_UNKNOWN
            or current.fence != expected_fence
        ):
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        resolved_at = datetime.now(timezone.utc)
        if target_status is ToolOperationStatus.COMPLETED:
            updated = replace(
                current,
                status=ToolOperationStatus.COMPLETED,
                result_payload=result_payload,
                error_code=None,
                error_payload=None,

                updated_at=resolved_at,
            )
        elif target_status is ToolOperationStatus.PENDING:
            updated = replace(
                current,
                status=ToolOperationStatus.PENDING,
                owner=None,

                error_code=None,
                error_payload=None,
                result_payload=None,
                updated_at=resolved_at,
            )
        elif target_status is ToolOperationStatus.FAILED:
            updated = replace(
                current,
                status=ToolOperationStatus.FAILED,
                result_payload=None,
                error_code=error_code,
                error_payload=error_payload,
                updated_at=resolved_at,
            )
        else:
            raise ValueError("unsupported tool effect resolution target")
        await admit_operation()
        await self._replace_tool_in_transaction(transaction, stored, updated)
        return updated

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
        return tuple(values)

    async def has_by_agent_run(
        self,
        agent_run_id: str,
        *,
        tenant_id: str,
    ) -> bool:
        if tenant_id != self._tenant_id:
            return False
        records = await self._records(
            "tool_operation",
            scope=self._scope("tool_operation", "agent_run", agent_run_id),
            states=frozenset(status.value for status in ToolOperationStatus),
            limit=1,
        )
        return bool(records)

    async def existing_call_ids(
        self, agent_run_id: str, tool_call_ids: Sequence[str], *, tenant_id: str,
    ) -> frozenset[str]:
        values = await self.get_by_call_ids(agent_run_id, tool_call_ids, tenant_id=tenant_id)
        return frozenset(value.tool_call_id for value in values)

    async def get_by_call_ids(
        self, agent_run_id: str, tool_call_ids: Sequence[str], *, tenant_id: str,
    ) -> tuple[ToolOperationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        if not isinstance(agent_run_id, str) or not agent_run_id:
            raise ValueError("agent_run_id must be a non-empty string")
        if not isinstance(tool_call_ids, Sequence) or isinstance(tool_call_ids, (str, bytes)):
            raise TypeError("tool_call_ids must be a sequence")
        ordered = tuple(dict.fromkeys(tool_call_ids))
        if any(not isinstance(value, str) or not value for value in ordered):
            raise ValueError("tool_call_ids must contain non-empty strings")
        if not ordered:
            return ()
        records = await self._records("tool_operation", scope=self._scope("tool_operation", "agent_run", agent_run_id))
        values = [await self._decode(record, ToolOperationRecord) for record in records]
        selected = {value.tool_call_id: value for value in values
                    if value.agent_run_id == agent_run_id and value.tool_call_id in ordered}
        missing = tuple(call_id for call_id in ordered if call_id not in selected)
        if missing:
            aliases = tuple(alias_digest(self._namespace, self._tenant_id, self._domain.value,
                                         "tool_call", [agent_run_id, call_id]) for call_id in missing)
            async def read(transaction: StateTransaction) -> None:
                resolved = await transaction.resolve_aliases(aliases)
                keys = tuple(dict.fromkeys(resolved.values()))
                sources = await transaction.get_records(keys) if keys else {}
                for call_id, alias in zip(missing, aliases, strict=True):
                    key = resolved.get(alias)
                    if key is None:
                        continue
                    stored = sources.get(key)
                    if stored is None or stored.kind != "tool_operation":
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    value = await self._decode(stored, ToolOperationRecord)
                    if value.tool_call_id != call_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    selected[call_id] = value
            await self._store.read(read)
        return tuple(selected[call_id] for call_id in ordered if call_id in selected)

    async def get_by_call(
        self, agent_run_id: str, tool_call_id: str, *, tenant_id: str,
    ) -> ToolOperationRecord | None:
        values = await self.get_by_call_ids(agent_run_id, (tool_call_id,), tenant_id=tenant_id)
        return values[0] if values else None

    async def list_by_agent_run(
        self,
        agent_run_id: str,
        *,
        tenant_id: str,
        execution_id: str | None = None,
        recovery_agent_run_id: str | None = None,
    ) -> tuple[ToolOperationRecord, ...]:
        if tenant_id != self._tenant_id:
            return ()
        if not isinstance(agent_run_id, str) or not agent_run_id:
            raise ValueError("agent_run_id must be a non-empty string")
        if execution_id is not None:
            values = await self._list_effect_records(execution_id, agent_run_id, recovery_agent_run_id)
            return tuple(values)
        records = await self._records(
            "tool_operation",
            scope=self._scope("tool_operation", "agent_run", agent_run_id),
        )
        values = tuple(
            [await self._decode(record, ToolOperationRecord) for record in records]
        )
        if any(value.agent_run_id != agent_run_id for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(values)

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
            receipt = await self._finish_tool(
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
                    error_code=None,
                    updated_at=now,
                ),
            )
            return receipt.value

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
            receipt = await self._finish_tool(
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
                    updated_at=now,
                ),
            )
            return receipt.value

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
                and current.fence == fence
            ):
                return current
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        now = datetime.now(timezone.utc)
        _require_tool_claim(current, owner=owner, fence=fence)
        value = replace(
            current,
            status=ToolOperationStatus.PENDING,
            owner=None,

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
        write: Callable[[StateTransaction], Awaitable[None]] | None = None,
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
                now = datetime.now(timezone.utc)
                if current.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    if (
                        current.owner == owner
                        and current.fence == fence
                        and current.error_code == error_code
                    ):
                        if write is not None:
                            guarded = await transaction.guard_record_revision(
                                record.key_digest, expected_storage_revision=record.storage_revision,
                            )
                            if guarded is None:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            await write(transaction)
                        return current
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                _require_tool_claim(current, owner=owner, fence=fence)
                value = replace(
                    current,
                    status=ToolOperationStatus.EFFECT_UNKNOWN,
                    error_code=error_code,
                    updated_at=now,
                )
                await self._replace_tool_in_transaction(transaction, record, value)
                if write is not None:
                    await write(transaction)
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
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        validate_lease_owner(owner)
        if terminal_status is ToolOperationStatus.FAILED:
            decode_tool_operation_failure(
                requested_error,
                requested_error_payload,
            )

        async def mutate(transaction: StateTransaction) -> Observed[ToolOperationRecord]:
            record = (await transaction.get_record(self._tool_key(tool_operation_id))
                      if observed is None else replace(
                          self._stored("tool_operation", observed.value.tool_operation_id, observed.value),
                          storage_revision=observed.storage_revision))
            if observed is not None and observed.value.tool_operation_id != tool_operation_id:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current = observed.value if observed is not None else await self._decode(record, ToolOperationRecord)
            now = datetime.now(timezone.utc)
            if current.status is ToolOperationStatus.COMPLETED:
                if (
                    terminal_status is ToolOperationStatus.COMPLETED
                    and current.owner == owner
                    and current.fence == fence
                    and current.result_payload == requested_result_payload
                ):
                    return Observed(current, record.storage_revision)
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
                    return Observed(current, record.storage_revision)
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            if current.status is ToolOperationStatus.CANCELLED:
                raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
            _require_tool_claim(current, owner=owner, fence=fence, allow_unknown=True)
            next_value = value(current, now)
            await self._replace_tool_in_transaction(transaction, record, next_value)
            return Observed(next_value, record.storage_revision + 1)

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
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
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
                error_code=None,
                updated_at=now,
            ),
            transaction=transaction,
            observed=observed,
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
        observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
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
                updated_at=now,
            ),
            transaction=transaction,
            observed=observed,
        )

    async def _replace_tool_in_transaction(
        self,
        transaction: StateTransaction,
        record: StoredRecord,
        value: ToolOperationRecord,
    ) -> None:
        candidate = _projected_record(self, record, value)
        await _replace_checked(transaction, candidate, record.storage_revision)


def _tool_replay_matches(left: ToolOperationRecord, right: ToolOperationRecord) -> bool:
    return (
        left.execution_id == right.execution_id
        and left.agent_run_id == right.agent_run_id
        and left.tool_call_id == right.tool_call_id
        and left.idempotency_key_digest == right.idempotency_key_digest
        and left.tool_name == right.tool_name
        and left.arguments_digest == right.arguments_digest
        and _tool_argument_payloads_match(
            left.arguments_payload,
            right.arguments_payload,
        )
        and left.binding_digest == right.binding_digest
        and (left.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}
             or (left.replay_safe is right.replay_safe
                 and left.capability_ref
                 == right.capability_ref))
    )


def _tool_admission_matches(
    left: ToolOperationRecord, right: ToolOperationAdmission
) -> bool:
    return left.matches_admission(right)


def _tool_argument_payloads_match(
    left: StoredPayload | None,
    right: StoredPayload | None,
) -> bool:
    if right is None:
        return left is None
    return left is not None and left.digest == right.digest and left.size == right.size


def _require_tool_claim(
    current: ToolOperationRecord,
    *,
    owner: str,
    fence: int,
    allow_unknown: bool = False,
) -> None:
    allowed = {ToolOperationStatus.CLAIMED}
    if allow_unknown:
        allowed.add(ToolOperationStatus.EFFECT_UNKNOWN)
    if current.status not in allowed or current.owner != owner or current.fence != fence:
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)











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


tool_admission_matches = _tool_admission_matches


__all__ = [
    "ApprovalRepositoryImpl",
    "ExternalCallRepositoryImpl",
    "ToolRepositoryImpl",
    "tool_admission_matches",
]
