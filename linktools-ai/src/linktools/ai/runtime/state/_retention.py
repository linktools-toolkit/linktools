#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned transient object and Step retention."""

from typing import Protocol

from linktools.core import environ

from ...storage import ObjectStore
from ._contracts import ConversationCursor, ConversationState, ExecutionState
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._steps import RuntimeStepStore

_logger = environ.get_logger("ai.runtime.state.retention")


class _RuntimeObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> ObjectStore: ...
    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None: ...
    async def clear_transient(self) -> None: ...


class RuntimeRetentionController:
    def __init__(
        self,
        *,
        conversation: ConversationState,
        execution: ExecutionState,
        memory: object,
        artifact: object,
        task: object,
        evaluation: object,
        recovery: object,
        objects: _RuntimeObjectRouter,
        steps: RuntimeStepStore,
        plan: RuntimeStatePlan,
        namespace: str,
    ) -> None:
        del memory, artifact, task, evaluation, recovery, namespace
        self._conversation = conversation
        self._execution = execution
        self._objects = objects
        self._steps = steps
        self._transient_domains = frozenset(
            domain for domain in RuntimeDomain if plan.route(domain).retention is RuntimeRetentionMode.TRANSIENT
        )
        self._closed = False

    async def release_execution_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        await self._steps.flush_dirty_execution_projections()
        execution = await self._execution.executions.get(execution_id, tenant_id=tenant_id)
        if execution is not None and execution.session_id is not None:
            await self._conversation.sessions.release_execution(
                execution.session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )
        if execution is not None:
            run_ids = tuple(f"{execution_id}:{sequence}" for sequence in range(1, execution.agent_run_sequence + 1))
            await self._steps.release_staging_many(candidate_step_run_ids=run_ids)
        for domain in self._transient_domains:
            await self._objects.release_object_scope(domain, owner_scope=f"execution:{execution_id}")
        _logger.info("execution transient handoff released: tenant=%s execution=%s", tenant_id, execution_id)

    async def release_session(
        self, session_id: str, *, tenant_id: str, continuation: ConversationCursor | None
    ) -> None:
        if RuntimeDomain.CONVERSATION not in self._transient_domains:
            return
        if continuation is not None:
            await self._steps.release_archive(RuntimeDomain.CONVERSATION, continuation.step_run_id)
        await self._objects.release_object_scope(RuntimeDomain.CONVERSATION, owner_scope=f"session:{session_id}")

    async def release_task_graph(self, graph_id: str, *, tenant_id: str) -> None:
        del graph_id, tenant_id

    async def release_evaluation(self, evaluation_id: str, *, tenant_id: str) -> None:
        del evaluation_id, tenant_id

    async def close(self) -> None:
        if self._closed:
            return
        await self._objects.clear_transient()
        self._closed = True
        _logger.debug("runtime transient retention closed")


__all__ = []
