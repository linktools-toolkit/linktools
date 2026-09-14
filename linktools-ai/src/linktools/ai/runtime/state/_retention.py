#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned transient object and Step retention."""

from typing import Protocol

from linktools.core import environ

from ...core import SessionStatus, step_run_id
from ...errors import AIError, ErrorCode
from ...storage import ObjectStore
from ._contracts import ConversationCursor, ConversationState, ExecutionState
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._steps import RuntimeStepStore

_logger = environ.get_logger("ai.runtime.state.retention")


class _RuntimeObjectRouter(Protocol):
    def object_store(self, domain: RuntimeDomain) -> ObjectStore: ...
    async def release_object_scope(
        self, domain: RuntimeDomain, *, owner_scope: str
    ) -> None: ...
    async def clear_transient(self) -> None: ...


class RuntimeRetentionController:
    def __init__(
        self,
        *,
        conversation: ConversationState,
        execution: ExecutionState,
        memory: object,
        artifact: object,
        evaluation: object,
        recovery: object,
        objects: _RuntimeObjectRouter,
        steps: RuntimeStepStore,
        plan: RuntimeStatePlan,
        namespace: str,
    ) -> None:
        del memory, artifact, evaluation, recovery
        self._conversation = conversation
        self._execution = execution
        self._namespace = namespace
        self._objects = objects
        self._steps = steps
        self._transient_domains = frozenset(
            domain
            for domain in RuntimeDomain
            if plan.route(domain).retention is RuntimeRetentionMode.TRANSIENT
        )
        self._closed = False

    async def release_execution_handoff(
        self, execution_id: str, *, tenant_id: str
    ) -> None:
        execution = await self._execution.executions.get(
            execution_id, tenant_id=tenant_id
        )
        if execution is not None and execution.session_id is not None:
            await self._conversation.sessions.release_execution(
                execution.session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )
        if execution is not None:
            run_ids = tuple(
                step_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    segment_sequence=sequence,
                )
                for sequence in range(1, execution.agent_run_sequence + 1)
            )
            await self._steps.release_staging_many(
                candidate_step_run_ids=run_ids,
                execution_id=execution_id,
            )
        for domain in self._transient_domains:
            await self._objects.release_object_scope(
                domain, owner_scope=f"execution:{execution_id}"
            )
        _logger.info(
            "execution transient handoff released: tenant=%s execution=%s",
            tenant_id,
            execution_id,
        )

    async def release_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None:
        if RuntimeDomain.CONVERSATION not in self._transient_domains:
            return
        sessions = await self._conversation.sessions.list(tenant_id=tenant_id)
        by_id = {record.session_id: record for record in sessions}
        current = by_id.get(session_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        protected_sessions: set[str] = set()
        protected_runs: set[str] = set()
        for record in sessions:
            if record.status is SessionStatus.CLOSED:
                continue
            if record.continuation is not None:
                protected_runs.add(record.continuation.step_run_id)
            parent_id = record.timeline_parent_session_id
            visited = {record.session_id}
            while parent_id is not None:
                if parent_id in visited:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                visited.add(parent_id)
                parent = by_id.get(parent_id)
                if parent is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                protected_sessions.add(parent_id)
                if parent.continuation is not None:
                    protected_runs.add(parent.continuation.step_run_id)
                parent_id = parent.timeline_parent_session_id

        candidates = []
        candidate_id: str | None = session_id
        visited: set[str] = set()
        while candidate_id is not None:
            if candidate_id in visited:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            visited.add(candidate_id)
            record = by_id.get(candidate_id)
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            candidates.append(record)
            candidate_id = record.timeline_parent_session_id

        released_runs: set[str] = set()
        for record in candidates:
            if record.status is not SessionStatus.CLOSED:
                continue
            candidate_continuation = (
                continuation if record.session_id == session_id else record.continuation
            )
            if candidate_continuation is not None:
                run_id = candidate_continuation.step_run_id
                if run_id not in protected_runs and run_id not in released_runs:
                    await self._steps.release_archive(RuntimeDomain.CONVERSATION, run_id)
                    released_runs.add(run_id)
            if record.session_id not in protected_sessions:
                await self._objects.release_object_scope(
                    RuntimeDomain.CONVERSATION,
                    owner_scope=f"session:{record.session_id}",
                )

    async def release_evaluation(self, evaluation_id: str, *, tenant_id: str) -> None:
        del evaluation_id, tenant_id

    async def close(self) -> None:
        if self._closed:
            return
        await self._objects.clear_transient()
        self._closed = True
        _logger.debug("runtime transient retention closed")


__all__ = []
