#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private SQL Runtime transaction and lifecycle composition primitives."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ...storage import SqlStorageContext
from ._contracts import RuntimeRepository
from ._plan import RuntimeDomain
from ._transaction import TransactionHub

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession



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


def _build_sql_domains(
    context: SqlStorageContext,
    *,
    namespace: str,
    tenant_id: str,
    domains: frozenset[RuntimeDomain],
    metadata: object,
    transaction_hub: TransactionHub | None = None,
) -> object:
    from ._repositories import (
        _SqlApprovalRepository,
        _SqlArtifactRepository,
        _SqlEvaluationRepository,
        _SqlEventRepository,
        _SqlExecutionRepository,
        _SqlExternalCallRepository,
        _SqlIdempotencyRepository,
        _SqlMemoryRepository,
        _SqlOperationRepository,
        _SqlRecoveryCheckpointRepository,
        _SqlSessionRepository,
        _SqlTaskRepository,
        _SqlToolRepository,
    )
    hub = transaction_hub or TransactionHub()
    sql_transactions = {
        domain: _SqlRuntimeTransaction(context, hub, domain)
        for domain in domains
    }
    sql_components: list[RuntimeRepository] = []
    sql_operations = {
        domain: _SqlOperationRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=domain, transaction=sql_transactions[domain])
        for domain in sql_transactions
    }
    sql_components.extend(sql_operations.values())
    sql_sessions = None
    if RuntimeDomain.CONVERSATION in domains:
        sql_sessions = _SqlSessionRepository(
            context,
            metadata,
            namespace=namespace,
            tenant_id=tenant_id,
            owner_domain=RuntimeDomain.CONVERSATION,
            transaction=sql_transactions[RuntimeDomain.CONVERSATION],
            operation=sql_operations[RuntimeDomain.CONVERSATION],
        )
        sql_components.append(sql_sessions)
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
    sql_memory = _SqlMemoryRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.MEMORY, transaction=sql_transactions[RuntimeDomain.MEMORY], operations=sql_operations[RuntimeDomain.MEMORY]) if RuntimeDomain.MEMORY in domains else None
    sql_artifact = _SqlArtifactRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.ARTIFACT, transaction=sql_transactions[RuntimeDomain.ARTIFACT]) if RuntimeDomain.ARTIFACT in domains else None
    sql_task = _SqlTaskRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.TASK, transaction=sql_transactions[RuntimeDomain.TASK]) if RuntimeDomain.TASK in domains else None
    sql_evaluation = _SqlEvaluationRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.EVALUATION, transaction=sql_transactions[RuntimeDomain.EVALUATION]) if RuntimeDomain.EVALUATION in domains else None
    sql_checkpoint = _SqlRecoveryCheckpointRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in domains else None
    sql_tool = _SqlToolRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in domains else None
    sql_approval = _SqlApprovalRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in domains else None
    sql_external = _SqlExternalCallRepository(context, metadata, namespace=namespace, tenant_id=tenant_id, owner_domain=RuntimeDomain.RECOVERY, transaction=sql_transactions[RuntimeDomain.RECOVERY]) if RuntimeDomain.RECOVERY in domains else None
    sql_components.extend(item for item in (sql_memory, sql_artifact, sql_task, sql_evaluation, sql_approval, sql_external, sql_checkpoint, sql_tool) if item is not None)
    from ._contracts import (
        ArtifactState,
        ConversationState,
        EvaluationState,
        ExecutionState,
        MemoryState,
        RecoveryState,
        TaskState,
    )
    from ._memory import _DomainRepositoryParts

    states: dict[RuntimeDomain, object] = {}
    if sql_sessions is not None:
        states[RuntimeDomain.CONVERSATION] = ConversationState(sql_sessions, sql_operations[RuntimeDomain.CONVERSATION])
    if sql_executions is not None and sql_events is not None:
        states[RuntimeDomain.EXECUTION] = ExecutionState(sql_executions, sql_events, sql_idempotency[RuntimeDomain.EXECUTION], sql_operations[RuntimeDomain.EXECUTION])
    if sql_memory is not None:
        states[RuntimeDomain.MEMORY] = MemoryState(sql_memory, sql_operations[RuntimeDomain.MEMORY])
    if sql_artifact is not None:
        states[RuntimeDomain.ARTIFACT] = ArtifactState(sql_artifact, sql_operations[RuntimeDomain.ARTIFACT])
    if sql_task is not None:
        states[RuntimeDomain.TASK] = TaskState(sql_task, sql_operations[RuntimeDomain.TASK])
    if sql_evaluation is not None:
        states[RuntimeDomain.EVALUATION] = EvaluationState(sql_evaluation, sql_idempotency[RuntimeDomain.EVALUATION], sql_operations[RuntimeDomain.EVALUATION])
    if all(value is not None for value in (sql_approval, sql_external, sql_checkpoint, sql_tool)):
        states[RuntimeDomain.RECOVERY] = RecoveryState(sql_approval, sql_external, sql_checkpoint, sql_operations[RuntimeDomain.RECOVERY], sql_tool)
    if frozenset(states) != domains:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _logger.info("SQL domains composed: namespace=%s tenant=%s domains=%s", namespace, tenant_id, sorted(domain.value for domain in domains))
    return _DomainRepositoryParts(states, tuple(dict.fromkeys(sql_components)))


__all__ = []
