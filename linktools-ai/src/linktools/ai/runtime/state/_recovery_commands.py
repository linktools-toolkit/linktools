#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable commands for execution recovery and tool-effect resolution."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone

from ...core import (
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    ToolOperationStatus,
)
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._contracts import ExecutionEventAppend, ExecutionRecord
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._repositories import (
    EventRepositoryImpl,
    ExecutionRepositoryImpl,
    OperationLedgerRepository,
    ToolRepositoryImpl,
    _append_operation,
    _projected_record,
    _replace_checked,
)
from ._store import StateGroupTransaction, StateStore, StoredFact, stream_digest


class RuntimeRecoveryCommands:
    """Own the new recovery mutations without creating another state owner."""

    def __init__(
        self,
        execution: ExecutionRepositoryImpl,
        events: EventRepositoryImpl,
        operations: OperationLedgerRepository,
        tools: ToolRepositoryImpl,
        *,
        background_tasks: "set[asyncio.Task[object]]",
    ) -> None:
        self._execution = execution
        self._events = events
        self._operations = operations
        self._tools = tools
        self._background_tasks = background_tasks

    async def commit_recovery_required(
        self,
        execution: ExecutionRecord,
        *,
        error_code: str,
        safe_error_details: Mapping[str, JsonValue],
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        if execution.status not in {
            ExecutionStatus.STARTED,
            ExecutionStatus.CANCELLING,
        }:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self._commit_execution_transition(
            execution,
            next_status=ExecutionStatus.RECOVERY_REQUIRED,
            event_type=ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
            event_payload={
                "error_code": error_code,
                "safe_error_details": dict(safe_error_details),
            },
            next_error_code=error_code,
            next_safe_error_details=safe_error_details,
            audit_events=audit_events,
        )

    async def commit_resumed(self, execution: ExecutionRecord) -> ExecutionRecord:
        if execution.status is not ExecutionStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await self._commit_execution_transition(
            execution,
            next_status=ExecutionStatus.STARTED,
            event_type=ExecutionEventType.EXECUTION_RESUMED,
            event_payload={},
            next_error_code=None,
            next_safe_error_details={},
        )

    async def _commit_execution_transition(
        self,
        execution: ExecutionRecord,
        *,
        next_status: ExecutionStatus,
        event_type: ExecutionEventType,
        event_payload: Mapping[str, JsonValue],
        next_error_code: str | None,
        next_safe_error_details: Mapping[str, JsonValue],
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord:
        ordered_audit = tuple(audit_events)
        events = (*ordered_audit, ExecutionEventAppend(event_type, event_payload))
        event_count = len(events)
        target_revision = execution.revision + event_count
        target_sequence = execution.event_sequence + event_count
        store = self._execution.state_store
        key = self._execution._key("execution", execution.execution_id)
        stream = stream_digest(
            self._execution._namespace,
            execution.tenant_id,
            self._execution._domain.value,
            "execution",
            execution.execution_id,
        )

        async def operation() -> ExecutionRecord:
            async def mutate(transaction):
                stored = await transaction.get_record(key)
                if stored is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = await self._execution._decode(stored, ExecutionRecord)
                if (
                    current.revision != execution.revision
                    or current.event_sequence != execution.event_sequence
                    or current.status is not execution.status
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                now = await transaction.now()
                updated = replace(
                    current,
                    status=next_status,
                    revision=target_revision,
                    event_sequence=target_sequence,
                    error_code=next_error_code,
                    safe_error_details=dict(next_safe_error_details),
                    error_diagnostics=None,
                    updated_at=now,
                )
                await _replace_checked(
                    transaction,
                    _projected_record(self._execution, stored, updated),
                    stored.storage_version,
                )
                await transaction.insert_facts(
                    tuple(
                        StoredFact(
                            stream,
                            execution.event_sequence + index,
                            key,
                            event.event_type.value,
                            None,
                            None,
                            event.payload,
                        )
                        for index, event in enumerate(events, 1)
                    )
                )
                return updated

            return await store.mutate(mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            try:
                current = await self._execution.get(
                    execution.execution_id,
                    tenant_id=execution.tenant_id,
                )
                if current is None:
                    return _partial()
                page = await self._events.list(
                    execution.execution_id,
                    tenant_id=execution.tenant_id,
                    after_sequence=execution.event_sequence,
                    limit=event_count + 1,
                )
                prefix = page.items[:event_count]
                prefix_matches = (
                    len(prefix) == event_count
                    and all(
                        actual.sequence == execution.event_sequence + index
                        and actual.event_type is expected.event_type
                        and actual.payload == expected.payload
                        for index, (actual, expected) in enumerate(
                            zip(prefix, events, strict=True),
                            1,
                        )
                    )
                )
                target_matches = (
                    current.status is next_status
                    and current.revision >= target_revision
                    and current.event_sequence >= target_sequence
                    and current.error_code == next_error_code
                    and dict(current.safe_error_details)
                    == dict(next_safe_error_details)
                    and current.error_diagnostics is None
                    and prefix_matches
                )
                if target_matches:
                    return CommitObservation(DurableCommitState.COMMITTED, value=current)
                predecessor = (
                    current == execution
                    and not page.items
                )
                if predecessor:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return _partial()
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        return _require_committed(outcome)

    async def resolve_tool_effect(
        self,
        operation: OperationLedgerInput,
        *,
        expected_fence: int,
        target_status: ToolOperationStatus,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord:
        if (
            operation.resource_kind is not ResourceKind.TOOL_OPERATION
            or operation.operation_kind is not OperationKind.TOOL_EFFECT_RESOLVE
            or operation.status is not OperationStatus.SUCCEEDED
            or operation.execution_id is None
            or operation.resource_id == ""
            or expected_fence < 0
        ):
            raise ValueError("tool effect resolution operation is invalid")
        if target_status is ToolOperationStatus.COMPLETED:
            if result_payload is None or error_code is not None or error_payload is not None:
                raise ValueError("applied resolution requires only a result payload")
        elif target_status is ToolOperationStatus.PENDING:
            if result_payload is not None or error_code is not None or error_payload is not None:
                raise ValueError("not-applied resolution cannot carry a terminal payload")
        elif target_status is ToolOperationStatus.FAILED:
            if result_payload is not None or error_code is None:
                raise ValueError("failed resolution requires an error")
        else:
            raise ValueError("unsupported tool effect resolution target")

        stores = _dedupe_stores(
            (self._operations.state_store, self._tools.state_store)
        )
        if any(store.storage_group is not stores[0].storage_group for store in stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def durable_operation() -> ToolOperationRecord:
            async def mutate(group: StateGroupTransaction) -> ToolOperationRecord:
                operation_tx = group.transaction(self._operations.state_store)
                tool_tx = group.transaction(self._tools.state_store)
                existing_operation = await self._operations.get_in_transaction(
                    operation_tx,
                    operation.operation_id,
                    tenant_id=operation.tenant_id,
                )
                stored_tool = await tool_tx.get_record(
                    self._tools._tool_key(operation.resource_id)
                )
                if stored_tool is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                current = await self._tools._decode(stored_tool, ToolOperationRecord)
                if current.execution_id != operation.execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

                if existing_operation is not None:
                    if not _same_resolution_operation(existing_operation, operation):
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                    if not _resolved_tool_matches(
                        current,
                        expected_fence=expected_fence,
                        target_status=target_status,
                        result_payload=result_payload,
                        error_code=error_code,
                        error_payload=error_payload,
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return current

                if (
                    current.status is not ToolOperationStatus.EFFECT_UNKNOWN
                    or current.fence != expected_fence
                ):
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                now = await tool_tx.now()
                if target_status is ToolOperationStatus.COMPLETED:
                    updated = replace(
                        current,
                        status=ToolOperationStatus.COMPLETED,
                        result_payload=result_payload,
                        error_code=None,
                        error_payload=None,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                elif target_status is ToolOperationStatus.PENDING:
                    updated = replace(
                        current,
                        status=ToolOperationStatus.PENDING,
                        owner=None,
                        lease_expires_at=None,
                        error_code=None,
                        error_payload=None,
                        result_payload=None,
                        updated_at=now,
                    )
                else:
                    updated = replace(
                        current,
                        status=ToolOperationStatus.FAILED,
                        result_payload=None,
                        error_code=error_code,
                        error_payload=error_payload,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                await _append_operation(operation_tx, self._operations, operation)
                await self._tools._replace_tool_in_transaction(
                    tool_tx,
                    stored_tool,
                    updated,
                )
                return updated

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[ToolOperationRecord]:
            try:
                ledger = await self._operations.get(
                    operation.operation_id,
                    tenant_id=operation.tenant_id,
                )
                tool = await self._tools.get_operation(
                    operation.resource_id,
                    tenant_id=operation.tenant_id,
                )
                if tool is None:
                    return _partial()
                if ledger is not None:
                    if not _same_resolution_operation(ledger, operation):
                        return _partial()
                    if _resolved_tool_matches(
                        tool,
                        expected_fence=expected_fence,
                        target_status=target_status,
                        result_payload=result_payload,
                        error_code=error_code,
                        error_payload=error_payload,
                    ):
                        return CommitObservation(DurableCommitState.COMMITTED, value=tool)
                    return _partial()
                if (
                    tool.execution_id == operation.execution_id
                    and tool.status is ToolOperationStatus.EFFECT_UNKNOWN
                    and tool.fence == expected_fence
                ):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return _partial()
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return _partial(error)
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        outcome = await run_durable_commit(
            durable_operation,
            readback,
            background_tasks=self._background_tasks,
        )
        return _require_committed(outcome)


def _same_resolution_operation(
    current: OperationLedgerRecord,
    candidate: OperationLedgerInput,
) -> bool:
    return (
        current.operation_id == candidate.operation_id
        and current.tenant_id == candidate.tenant_id
        and current.resource_kind is candidate.resource_kind
        and current.resource_id == candidate.resource_id
        and current.execution_id == candidate.execution_id
        and current.operation_kind is candidate.operation_kind
        and current.status is candidate.status
        and current.request_digest == candidate.request_digest
        and current.result_ref == candidate.result_ref
        and current.result_digest == candidate.result_digest
        and current.error_code == candidate.error_code
        and current.compactable == candidate.compactable
    )


def _resolved_tool_matches(
    current: ToolOperationRecord,
    *,
    expected_fence: int,
    target_status: ToolOperationStatus,
    result_payload: StoredPayload | None,
    error_code: str | None,
    error_payload: StoredPayload | None,
) -> bool:
    if current.fence != expected_fence or current.status is not target_status:
        return False
    if target_status is ToolOperationStatus.COMPLETED:
        return (
            current.result_payload == result_payload
            and current.error_code is None
            and current.error_payload is None
        )
    if target_status is ToolOperationStatus.PENDING:
        return (
            current.owner is None
            and current.lease_expires_at is None
            and current.result_payload is None
            and current.error_code is None
            and current.error_payload is None
        )
    return (
        current.result_payload is None
        and current.error_code == error_code
        and current.error_payload == error_payload
    )


def _dedupe_stores(stores: Sequence[StateStore]) -> tuple[StateStore, ...]:
    result: list[StateStore] = []
    seen: set[int] = set()
    for store in stores:
        if id(store) in seen:
            continue
        result.append(store)
        seen.add(id(store))
    return tuple(result)


def _partial(error: BaseException | None = None):
    return CommitObservation(
        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
        error=error or AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
    )


def _require_committed(outcome):
    if outcome.state is DurableCommitState.COMMITTED:
        if outcome.value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if outcome.cancelled:
            raise asyncio.CancelledError
        return outcome.value
    if outcome.state is DurableCommitState.NOT_COMMITTED:
        if outcome.error is not None:
            raise outcome.error
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error


__all__ = ["RuntimeRecoveryCommands"]
