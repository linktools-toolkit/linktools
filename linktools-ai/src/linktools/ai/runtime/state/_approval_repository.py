#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Approval admission repository."""

import asyncio

from ...core import (
    ApprovalStatus,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    operation_replay_matches,
)
from ...errors import AIError, ErrorCode
from ._contracts import ApprovalRecord
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._repositories import ApprovalRepositoryImpl, OperationLedgerRepository
from ._store import StateStore, StateTransaction, operation_key, sequence_key


class ApprovalAdmissionRepositoryImpl(
    ApprovalRepositoryImpl,
    OperationLedgerRepository,
):
    """Persist Approval creation and its idempotency result as one checkpoint."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        ApprovalRepositoryImpl.__init__(
            self,
            store,
            namespace=namespace,
            tenant_id=tenant_id,
        )
        self._background_tasks: set[asyncio.Task[object]] = set()

    async def create_with_operation(
        self,
        record: ApprovalRecord,
        *,
        operation: OperationLedgerInput,
    ) -> tuple[ApprovalRecord, bool]:
        self._validate_create(record, operation)

        async def commit() -> tuple[ApprovalRecord, bool]:
            return await self._store.mutate(
                lambda transaction: self._create_in_transaction(
                    transaction,
                    record,
                    operation,
                )
            )

        async def readback() -> CommitObservation[tuple[ApprovalRecord, bool]]:
            return await self._store.read(
                lambda transaction: self._read_create(
                    transaction,
                    record,
                    operation,
                )
            )

        result = await run_durable_commit(
            commit,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            if result.cancelled:
                raise asyncio.CancelledError
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _create_in_transaction(
        self,
        transaction: StateTransaction,
        record: ApprovalRecord,
        operation: OperationLedgerInput,
    ) -> tuple[ApprovalRecord, bool]:
        operation_record = await OperationLedgerRepository.get_in_transaction(
            self,
            transaction,
            operation.operation_id,
            tenant_id=record.tenant_id,
        )
        stored = await transaction.get_record(
            self._key("approval", record.approval_id)
        )
        if operation_record is not None:
            if not operation_replay_matches(operation_record, operation):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing = await self._decode(stored, ApprovalRecord)
            if not _same_create_identity(existing, record):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing, True
        if stored is not None:
            raise AIError(ErrorCode.APPROVAL_CONFLICT)

        sequence = await transaction.next_sequence(
            sequence_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                "operation",
                [operation.resource_kind.value, operation.resource_id],
            )
        )
        await transaction.insert_operation(
            self._stored_operation(operation, sequence)
        )
        await transaction.insert_record(
            self._stored(
                "approval",
                record.approval_id,
                record,
                state=record.status.value,
            )
        )
        return record, False

    async def _read_create(
        self,
        transaction: StateTransaction,
        record: ApprovalRecord,
        operation: OperationLedgerInput,
    ) -> CommitObservation[tuple[ApprovalRecord, bool]]:
        stored_operation = await transaction.get_operation(
            operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                operation.operation_id,
            )
        )
        stored_approval = await transaction.get_record(
            self._key("approval", record.approval_id)
        )
        if stored_operation is None and stored_approval is None:
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        if stored_operation is None:
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.APPROVAL_CONFLICT),
            )
        operation_record = await OperationLedgerRepository.get_in_transaction(
            self,
            transaction,
            operation.operation_id,
            tenant_id=record.tenant_id,
        )
        if operation_record is None:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        if not operation_replay_matches(operation_record, operation):
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
            )
        if stored_approval is None:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        existing = await self._decode(stored_approval, ApprovalRecord)
        if not _same_create_identity(existing, record):
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        return CommitObservation(
            DurableCommitState.COMMITTED,
            (existing, True),
        )

    def _validate_create(
        self,
        record: ApprovalRecord,
        operation: OperationLedgerInput,
    ) -> None:
        if record.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (
            record.status is not ApprovalStatus.PENDING
            or record.idempotency_key_digest is not None
            or record.decision is not None
            or record.decided_by is not None
            or record.decision_digest is not None
            or record.decided_at is not None
        ):
            raise ValueError("approval create record must be pending")
        if (
            operation.tenant_id != record.tenant_id
            or operation.resource_kind is not ResourceKind.APPROVAL
            or operation.resource_id != record.approval_id
            or operation.execution_id != record.execution_id
            or operation.operation_kind is not OperationKind.APPROVAL
            or operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != record.approval_id
            or operation.result_digest is None
            or operation.error_code is not None
            or not operation.compactable
        ):
            raise ValueError("approval create operation identity is invalid")


def _same_create_identity(left: ApprovalRecord, right: ApprovalRecord) -> bool:
    return (
        left.approval_id == right.approval_id
        and left.execution_id == right.execution_id
        and left.tenant_id == right.tenant_id
        and left.operation_id == right.operation_id
    )


__all__ = ["ApprovalAdmissionRepositoryImpl"]
