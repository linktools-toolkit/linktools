#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter-owned, owner-scoped cleanup for transient Runtime state."""

from linktools.core import environ

from ..core import step_run_id
from ..runtime import ConversationCursor, RuntimeDomain, RuntimeRetention, RuntimeStoragePlan, RuntimeStores
from ._persistence import RuntimeObjectRouter, _RuntimeOwnerPruner
from ._memory import _memory_working_scope_key
from ._step import RuntimeStepPersistence


_logger = environ.get_logger("ai.adapter.retention")


class _RuntimeRetention:
    def __init__(self, stores: RuntimeStores, steps: RuntimeStepPersistence, plan: RuntimeStoragePlan, *, namespace: str) -> None:
        self._stores = stores
        self._steps = steps
        self._plan = plan
        self._namespace = namespace
        self._transient_domains = frozenset(domain for domain in RuntimeDomain if plan.route(domain).retention is RuntimeRetention.TRANSIENT)
        self._pruner = _RuntimeOwnerPruner(stores, self._transient_domains)
        self._closed = False

    async def release_execution_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        execution = await self._stores.execution.executions.get(execution_id, tenant_id=tenant_id)
        owner_scope = f"execution:{execution_id}"
        if execution is None:
            if RuntimeDomain.EXECUTION in self._transient_domains:
                await self._stores.release_object_scope(RuntimeDomain.EXECUTION, owner_scope=owner_scope)
            return
        candidates = frozenset(
            step_run_id(
                namespace=self._namespace,
                tenant_id=tenant_id,
                execution_id=execution_id,
                segment_sequence=sequence,
            )
            for sequence in range(1, execution.agent_run_sequence + 1)
        )
        memory_scope_key = None
        if execution.memory_scope is not None:
            memory_scope_key = _memory_working_scope_key(execution_id, execution.memory_scope)
        for domain in (RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY):
            if domain not in self._transient_domains:
                continue
            await self._pruner.prune_execution_working(
                execution_id,
                tenant_id,
                memory_scope_key,
                candidates,
                domains=frozenset({domain}),
            )
            await self._stores.release_object_scope(domain, owner_scope=owner_scope)
        await self._steps.release_staging_many(candidate_step_run_ids=tuple(sorted(candidates)))
        if RuntimeDomain.EXECUTION in self._transient_domains:
            await self._pruner.prune_execution_terminal(execution_id, tenant_id)
            await self._stores.release_object_scope(RuntimeDomain.EXECUTION, owner_scope=owner_scope)
        _logger.info("execution transient handoff released: tenant=%s execution=%s attempts=%s", tenant_id, execution_id, len(candidates))

    async def release_session(self, session_id: str, *, tenant_id: str, continuation: ConversationCursor | None) -> None:
        if RuntimeDomain.CONVERSATION not in self._transient_domains:
            return
        continuation_step_run_id = None if continuation is None else continuation.step_run_id
        release_archive = await self._pruner.prune_session(session_id, tenant_id, continuation_step_run_id)
        if release_archive and continuation_step_run_id is not None:
            await self._steps.release_archive(RuntimeDomain.CONVERSATION, continuation_step_run_id)
        _logger.debug("session transient handoff released: tenant=%s session=%s archive=%s", tenant_id, session_id, release_archive)

    async def release_task_graph(self, graph_id: str, *, tenant_id: str) -> None:
        if RuntimeDomain.TASK in self._transient_domains:
            await self._pruner.prune_task_graph(graph_id, tenant_id)

    async def release_evaluation(self, evaluation_id: str, *, tenant_id: str) -> None:
        if RuntimeDomain.EVALUATION in self._transient_domains:
            await self._pruner.prune_evaluation(evaluation_id, tenant_id)

    async def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            await self._pruner.clear_transient(self._transient_domains)
        except BaseException as error:
            failure = error
        try:
            if isinstance(self._stores.object_router, RuntimeObjectRouter):
                await self._stores.object_router._clear_transient(self._transient_domains)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            _logger.error("runtime transient retention close failed", exc_info=environ.debug)
            raise failure
        self._closed = True


__all__ = []
