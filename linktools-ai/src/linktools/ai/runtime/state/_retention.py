#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned retention and transient object cleanup."""

from typing import Protocol

from linktools.core import environ

from ...core import canonical_sha256, step_run_id
from ...storage import ObjectStore
from ._contracts import (
    ArtifactState,
    ConversationCursor,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    TaskState,
)
from ._memory import _RuntimeOwnerPruner
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._steps import RuntimeStepStore

class _RuntimeObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> "ObjectStore": ...
    def working_object_store(self, domain: RuntimeDomain, *, owner_scope: str) -> "ObjectStore": ...
    async def release_object_scope(self, domain: RuntimeDomain, *, owner_scope: str) -> None: ...
    async def clear_transient(self) -> None: ...


_logger = environ.get_logger("ai.runtime.state.retention")


class RuntimeRetentionController:
    def __init__(
        self,
        *,
        conversation: ConversationState,
        execution: ExecutionState,
        memory: MemoryState,
        artifact: ArtifactState,
        task: TaskState,
        evaluation: EvaluationState,
        recovery: RecoveryState,
        objects: "_RuntimeObjectRouter",
        steps: RuntimeStepStore,
        plan: RuntimeStatePlan,
        namespace: str,
    ) -> None:
        self._execution = execution
        self._objects = objects
        self._steps = steps
        self._namespace = namespace
        self._transient_domains = frozenset(
            domain for domain in RuntimeDomain if plan.route(domain).retention is RuntimeRetentionMode.TRANSIENT
        )
        self._pruner = _RuntimeOwnerPruner(
            conversation=conversation,
            execution=execution,
            memory=memory,
            artifact=artifact,
            task=task,
            evaluation=evaluation,
            recovery=recovery,
            transient_domains=self._transient_domains,
        )
        self._closed = False

    async def release_execution_handoff(self, execution_id: str, *, tenant_id: str) -> None:
        execution = await self._execution.executions.get(execution_id, tenant_id=tenant_id)
        candidates: tuple[str, ...] = ()
        if execution is not None:
            candidates = tuple(
                step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    segment_sequence=sequence,
                )
                for sequence in range(1, execution.agent_run_sequence + 1)
            )
            await self._steps.release_staging_many(candidate_step_run_ids=candidates)
        owner_scope = f"execution:{execution_id}"
        for domain in self._transient_domains:
            if domain is RuntimeDomain.EXECUTION:
                await self._pruner.prune_execution_terminal(execution_id, tenant_id)
            elif domain in {RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY}:
                memory_scope_digest = None
                if execution is not None and execution.memory_scope is not None:
                    memory_scope_digest = _memory_working_scope_digest(execution_id, execution.memory_scope)
                await self._pruner.prune_execution_working(
                    execution_id,
                    tenant_id,
                    memory_scope_digest,
                    frozenset(candidates),
                    domains=frozenset({domain}),
                )
            try:
                await self._objects.release_object_scope(domain, owner_scope=owner_scope)
            except BaseException:
                _logger.error("transient object scope release failed: domain=%s scope=%s", domain.value, owner_scope, exc_info=True)
                raise
        _logger.info("execution transient handoff released: tenant=%s execution=%s", tenant_id, execution_id)

    async def release_session(self, session_id: str, *, tenant_id: str, continuation: "ConversationCursor | None") -> None:
        if RuntimeDomain.CONVERSATION not in self._transient_domains:
            return
        continuation_step_run_id = None if continuation is None else continuation.step_run_id
        release_archive = await self._pruner.prune_session(session_id, tenant_id, continuation_step_run_id)
        if release_archive and continuation_step_run_id is not None:
            await self._steps.release_archive(RuntimeDomain.CONVERSATION, continuation_step_run_id)
        await self._objects.release_object_scope(RuntimeDomain.CONVERSATION, owner_scope=f"session:{session_id}")

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
        for cleanup in (self._pruner.clear_transient, self._objects.clear_transient):
            try:
                await cleanup()
            except BaseException as error:
                if failure is None:
                    failure = error
                _logger.error("transient retention cleanup failed", exc_info=True)
        if failure is not None:
            raise failure
        self._closed = True
        _logger.debug("runtime transient retention closed: domains=%s", sorted(domain.value for domain in self._transient_domains))


def _memory_working_scope_digest(execution_id: str, memory_scope: str) -> str:
    return canonical_sha256({"execution_id": execution_id, "memory_scope_digest": canonical_sha256(memory_scope)})


__all__ = []
