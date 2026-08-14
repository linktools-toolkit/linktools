#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private SQL Runtime transaction and lifecycle composition primitives."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.core import environ

from ...errors import AIError, ErrorCode
from .._domain import RuntimeDomain
from .._persistence import (
    ArtifactStore,
    ConversationStore,
    EvaluationStore,
    ExecutionStore,
    MemoryStore,
    RecoveryStore,
    RuntimeRepository,
    RuntimeDomainStates,
    TaskStore,
)
from ._contracts import RuntimeRetentionMode
from ._plan import RuntimeStatePlan
from ...storage import SqlStorageContext
from ._transaction import TransactionHub

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ...storage import ObjectStore


_logger = environ.get_logger("ai.runtime.state.sql")


class _SqlRuntimeTransaction:
    def __init__(self, context: SqlStorageContext, hub: TransactionHub, owner_domain: RuntimeDomain) -> None:
        self._context = context
        self._hub = hub
        self._owner_domain = owner_domain
        self._session: "AsyncSession | None" = None

    @asynccontextmanager
    async def mutation(self) -> AsyncIterator[None]:
        await self._hub.enter(self._owner_domain)
        outer = self._hub.depth == 1
        session = self._session
        try:
            if outer:
                session = self._context.sessions()
                self._session = session
                try:
                    await session.begin()
                except BaseException as error:
                    await self._close_failed(session)
                    self._session = None
                    await self._exit_hub(type(error))
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            elif session is None:
                raise RuntimeError("SQL session requested outside owner transaction")
            try:
                yield
            except BaseException as error:
                if not outer:
                    await self._exit_hub(type(error))
                    raise
                await self._rollback_close_exit(error, session)
                if isinstance(error, AIError):
                    raise
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            if not outer:
                await self._exit_hub(None)
                return
            try:
                await session.commit()
            except BaseException as error:
                await self._rollback_best_effort(session)
                await self._close_failed(session)
                self._session = None
                await self._exit_hub(type(error))
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            try:
                await session.close()
            except BaseException as error:
                self._session = None
                await self._exit_hub(None)
                _logger.error("SQL session close failed after commit: domain=%s committed=True close_failed=True", self._owner_domain.value, exc_info=environ.debug)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            self._session = None
            await self._exit_hub(None)
        except AIError:
            raise

    def mark_changed(self) -> None:
        self._hub.mark_changed(self._owner_domain)

    def current_session(self) -> "AsyncSession":
        if asyncio.current_task() is not self._hub.active_task or self._hub.active_domain is not self._owner_domain or self._hub.depth <= 0 or self._session is None:
            raise RuntimeError("SQL session requested outside owner transaction")
        return self._session

    async def _rollback_close_exit(self, error: BaseException, session: "AsyncSession") -> None:
        failed = False
        try:
            await session.rollback()
        except BaseException:
            failed = True
        try:
            await session.close()
        except BaseException:
            failed = True
        self._session = None
        try:
            await self._exit_hub(type(error))
        except BaseException:
            failed = True
        if failed:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _exit_hub(self, error_type: object) -> None:
        await self._hub.exit(self._owner_domain, error_type)

    async def _rollback_best_effort(self, session: "AsyncSession") -> None:
        try:
            await session.rollback()
        except BaseException:
            _logger.error("SQL transaction rollback failed: domain=%s", self._owner_domain.value, exc_info=environ.debug)

    async def _close_failed(self, session: "AsyncSession") -> None:
        try:
            await session.close()
        except BaseException:
            _logger.error("SQL session close failed: domain=%s", self._owner_domain.value, exc_info=environ.debug)


@dataclass
class _SqlRuntime:
    persistence: RuntimeDomainStates
    components: tuple[RuntimeRepository, ...]
    context: SqlStorageContext

    async def initialize(self) -> None:
        initialized: list[RuntimeRepository] = []
        seen: set[int] = set()
        try:
            for component in self.components:
                if id(component) in seen:
                    continue
                await component.initialize()
                initialized.append(component)
                seen.add(id(component))
        except BaseException:
            for component in reversed(initialized):
                await component.close()
            raise

    async def close(self) -> None:
        first: BaseException | None = None
        seen: set[int] = set()
        for component in reversed(self.components):
            if id(component) in seen:
                continue
            seen.add(id(component))
            try:
                await component.close()
            except BaseException as error:
                if first is None:
                    first = error
        if first is not None:
            raise first


def build_sql_runtime(
    context: SqlStorageContext,
    *,
    namespace: str,
    tenant_id: str,
    plan: RuntimeStatePlan,
    object_store: "ObjectStore | None" = None,
) -> _SqlRuntime:
    from ...storage import SqlObjectStore
    from ._memory import (
        RuntimeObjectRouter,
        _build_in_memory_parts,
    )
    from ._repositories import (
        _SqlApprovalRepository,
        _SqlArtifactRepository,
        _SqlEventRepository,
        _SqlExecutionRepository,
        _SqlExternalCallRepository,
        _SqlEvaluationRepository,
        _SqlIdempotencyRepository,
        _SqlMemoryRepository,
        _SqlOperationRepository,
        _SqlRecoveryCheckpointRepository,
        _SqlSessionRepository,
        _SqlTaskRepository,
        _SqlToolRepository,
    )
    from ._schema import build_runtime_sql_metadata

    from ._memory import RuntimeTransactionBinding

    metadata = build_runtime_sql_metadata(
        plan,
        include_object_tables=object_store is None and any(
            plan.route(domain).retention is RuntimeRetentionMode.DURABLE
            and domain in {RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY}
            for domain in RuntimeDomain
        ),
    )
    hub = TransactionHub()
    transaction_binding = RuntimeTransactionBinding()
    durable_domains = frozenset(domain for domain in RuntimeDomain if plan.route(domain).retention is RuntimeRetentionMode.DURABLE)
    memory_domains = frozenset(RuntimeDomain) - durable_domains
    memory_parts = _build_in_memory_parts(
        namespace=namespace,
        domains=memory_domains,
        transaction_binding=transaction_binding,
        transaction_hub=hub,
    )
    memory_components = memory_parts.components
    sql_transactions = {
        domain: _SqlRuntimeTransaction(context, hub, domain)
        for domain in RuntimeDomain
        if plan.route(domain).retention is RuntimeRetentionMode.DURABLE
    }
    sql_components: list[RuntimeRepository] = []
    sql_sessions = None
    if RuntimeDomain.CONVERSATION in sql_transactions:
        sql_sessions = _SqlSessionRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.CONVERSATION, transaction=sql_transactions[RuntimeDomain.CONVERSATION])
        sql_components.append(sql_sessions)
    sql_operations = {
        domain: _SqlOperationRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=domain, transaction=sql_transactions[domain])
        for domain in sql_transactions
    }
    sql_components.extend(sql_operations.values())
    sql_idempotency = {
        domain: _SqlIdempotencyRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=domain, transaction=sql_transactions[domain], runtime_domain=domain)
        for domain in (RuntimeDomain.EXECUTION, RuntimeDomain.EVALUATION)
        if domain in sql_transactions
    }
    sql_components.extend(sql_idempotency.values())
    sql_events = None
    sql_executions = None
    if RuntimeDomain.EXECUTION in sql_transactions:
        sql_events = _SqlEventRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.EXECUTION, transaction=sql_transactions[RuntimeDomain.EXECUTION])
        sql_components.append(sql_events)
        sql_executions = _SqlExecutionRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.EXECUTION, transaction=sql_transactions[RuntimeDomain.EXECUTION], idempotency=sql_idempotency[RuntimeDomain.EXECUTION], events=sql_events, operations=sql_operations[RuntimeDomain.EXECUTION])
        sql_components.append(sql_executions)
    sql_memory = _SqlMemoryRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.MEMORY, transaction=sql_transactions[RuntimeDomain.MEMORY], operations=sql_operations[RuntimeDomain.MEMORY]) if RuntimeDomain.MEMORY in sql_transactions else None
    sql_artifact = _SqlArtifactRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.ARTIFACT, transaction=sql_transactions[RuntimeDomain.ARTIFACT]) if RuntimeDomain.ARTIFACT in sql_transactions else None
    sql_task = _SqlTaskRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.TASK, transaction=sql_transactions[RuntimeDomain.TASK]) if RuntimeDomain.TASK in sql_transactions else None
    sql_evaluation = _SqlEvaluationRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.EVALUATION, transaction=sql_transactions[RuntimeDomain.EVALUATION]) if RuntimeDomain.EVALUATION in sql_transactions else None
    sql_checkpoint = _SqlRecoveryCheckpointRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in sql_transactions else None
    sql_tool = _SqlToolRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in sql_transactions else None
    sql_approval = _SqlApprovalRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in sql_transactions else None
    sql_external = _SqlExternalCallRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in sql_transactions else None
    sql_components.extend(item for item in (sql_memory, sql_artifact, sql_task, sql_evaluation, sql_approval, sql_external, sql_checkpoint, sql_tool) if item is not None)
    durable = durable_domains
    memory = memory_parts
    operation_stores = {
        domain: sql_operations[domain] if domain in sql_operations else memory_parts.operation_repositories[domain]
        for domain in RuntimeDomain
        if domain in memory_parts.operation_repositories or domain in sql_operations
    }
    def operation(domain: RuntimeDomain) -> object:
        return operation_stores[domain]
    conversation_sessions = sql_sessions if sql_sessions is not None else memory.sessions
    execution_store = sql_executions if sql_executions is not None else memory.executions
    execution_idempotency = sql_idempotency.get(RuntimeDomain.EXECUTION, memory.execution_idempotency)
    execution_events = sql_events if sql_events is not None else memory.events
    memory_records = sql_memory if sql_memory is not None else memory.memories
    artifact_records = sql_artifact if sql_artifact is not None else memory.artifacts
    task_records = sql_task if sql_task is not None else memory.tasks
    evaluation_records = sql_evaluation if sql_evaluation is not None else memory.evaluations
    evaluation_idempotency = sql_idempotency.get(RuntimeDomain.EVALUATION, memory.evaluation_idempotency)
    recovery_approvals = sql_approval if sql_approval is not None else memory.approvals
    recovery_external = sql_external if sql_external is not None else memory.external_calls
    recovery_checkpoints = sql_checkpoint if sql_checkpoint is not None else memory.recovery_checkpoint
    recovery_tools = sql_tool if sql_tool is not None else memory.tools
    object_stores = {}
    sql_objects = object_store or SqlObjectStore._from_context(context)
    for domain in RuntimeDomain:
        object_stores[domain] = sql_objects if domain in durable else memory.object_stores[domain]
    persistence = RuntimeDomainStates(
        namespace=namespace,
        conversation=ConversationStore(conversation_sessions, operation(RuntimeDomain.CONVERSATION)),
        execution=ExecutionStore(execution_store, execution_idempotency, execution_events, operation(RuntimeDomain.EXECUTION)),
        memory=MemoryStore(memory_records, operation(RuntimeDomain.MEMORY)),
        artifact=ArtifactStore(artifact_records, operation(RuntimeDomain.ARTIFACT)),
        task=TaskStore(task_records, operation(RuntimeDomain.TASK)),
        evaluation=EvaluationStore(evaluation_records, evaluation_idempotency, operation(RuntimeDomain.EVALUATION)),
        recovery=RecoveryStore(recovery_approvals, recovery_external, recovery_checkpoints, operation(RuntimeDomain.RECOVERY), recovery_tools),
        object_router=RuntimeObjectRouter(object_stores),
    )
    components = tuple(dict.fromkeys((*sql_components, *memory_components)))
    _logger.info("SQL runtime composed: namespace=%s tenant=%s durable=%s", namespace, tenant_id, sorted(domain.value for domain in durable))
    return _SqlRuntime(persistence, components, context)


__all__ = []
