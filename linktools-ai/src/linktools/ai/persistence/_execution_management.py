#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-owned storage operations for shared execution settlement."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from ..core import (
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    ResourceRef,
    RUNTIME_OBJECT_STORE_ID,
    PersistenceDomain,
)
from ..errors import AIError, ErrorCode
from ..execution import ExecutionDriver, ExecutionRecord, IdempotencyRecord, ResultRecord
from ..storage import ObjectRef, ObjectStore
from .contracts import ExecutionRepositories
from ._execution_repositories import ExecutionRepositoryImpl
from .objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from ._object_router import runtime_object_source_reads

class StoredExecutionManagementAdapter:
    """Bind execution queries and terminal results to their durable repositories."""

    @property
    def driver(self) -> ExecutionDriver | None:
        return None

    @property
    def has_pending_cancel_promotion(self) -> bool:
        return False

    async def promote_pending_cancel(self, execution_id: str) -> None:
        return None

    async def read_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        return await self.read_source_header(execution_id, tenant_id=tenant_id)

    def has_local_pending(self, execution_id: str, *, tenant_id: str) -> bool:
        return False

    async def read_remote_pending(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return None

    async def signal_remote_cancel(self, execution_id: str, *, tenant_id: str) -> None:
        return None

    def __init__(
        self, repositories: ExecutionRepositories, object_store: ObjectStore, *,
        object_key_factory: ObjectKeyFactory | None = None,
        object_reader: Callable[[PersistenceDomain, ObjectRef], Awaitable[bytes]] | None = None,
    ) -> None:
        self._repositories = repositories
        self._object_store = object_store
        self._object_key_factory = object_key_factory
        self._object_reader = object_reader

    @property
    def embedded_results(self) -> bool:
        return type(self._repositories.executions) is ExecutionRepositoryImpl

    def source_reads(self) -> AbstractContextManager[None]:
        return runtime_object_source_reads()

    async def write_output(self, content: bytes, *, tenant_id: str) -> ObjectRef:
        if self._object_key_factory is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await put_runtime_object(
            self._object_store, self._object_key_factory, PersistenceDomain.EXECUTION, tenant_id, content,
        )

    async def read_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self.read_source_execution(execution_id, tenant_id=tenant_id)

    async def read_source_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        repository = self._repositories.executions
        if type(repository) is ExecutionRepositoryImpl:
            return await repository.get_header_fresh(execution_id, tenant_id=tenant_id)
        return await repository.state_store.read(
            lambda _transaction: repository.get_header(execution_id, tenant_id=tenant_id)
        )

    async def read_source_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        repository = self._repositories.executions
        if type(repository) is ExecutionRepositoryImpl:
            return await repository.get_fresh(execution_id, tenant_id=tenant_id)
        return await repository.state_store.read(
            lambda _transaction: repository.get(execution_id, tenant_id=tenant_id)
        )

    async def get_operation(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None:
        return await self._repositories.operations.get(operation_id, tenant_id=tenant_id)

    async def append_operation(self, operation: OperationLedgerInput) -> OperationLedgerRecord:
        return await self._repositories.operations.append(operation)

    async def compare_and_swap_operation(self, operation_id: str, *, tenant_id: str,
        expected_status: OperationStatus, next_record: OperationLedgerRecord) -> OperationLedgerRecord:
        return await self._repositories.operations.compare_and_swap(operation_id, tenant_id=tenant_id,
            expected_status=expected_status, next_record=next_record)

    async def list_idempotency(self, resource_kind: ResourceKind, resource_id: str,
                              *, tenant_id: str) -> tuple[IdempotencyRecord, ...]:
        return await self._repositories.idempotency.list_by_resource(resource_kind, resource_id, tenant_id=tenant_id)

    async def get_idempotency(self, scope: str, key_digest: str, *, tenant_id: str) -> IdempotencyRecord | None:
        return await self._repositories.idempotency.get(scope, key_digest, tenant_id=tenant_id)

    async def get_result(self, execution_id: str, *, tenant_id: str) -> ResultRecord | None:
        return await self._repositories.executions.get_result(execution_id, tenant_id=tenant_id)

    async def result_record(self, execution: ExecutionRecord, *, tenant_id: str) -> ResultRecord | None:
        repository = self._repositories.executions
        return (execution.result if type(repository) is ExecutionRepositoryImpl else
                await repository.get_result(execution.execution_id, tenant_id=tenant_id))

    async def read_output(self, reference: ObjectRef) -> bytes:
        if self._object_reader is None:
            return await read_runtime_object(self._object_store, reference)
        if reference.store_id != RUNTIME_OBJECT_STORE_ID:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return await self._object_reader(PersistenceDomain.EXECUTION, reference)


__all__ = ["StoredExecutionManagementAdapter"]
