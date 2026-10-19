#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned transient object and AgentRun retention."""

from ...storage import StorageRetentionMode

from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ...core import ExecutionStatus, agent_run_id
from ...storage import ObjectStore
from ._contracts import ConversationCursor, ConversationRepositories, ExecutionRepositories
from ._execution_repositories import ExecutionRepositoryImpl
from ._plan import RuntimeDomain, RuntimeStoragePlan
from ._steps import RuntimeAgentRunStore

if TYPE_CHECKING:
    from ._contracts import ExecutionRecord
    from ._store import StateTransaction

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
        conversation: ConversationRepositories,
        execution: ExecutionRepositories,
        objects: _RuntimeObjectRouter,
        run_store: RuntimeAgentRunStore,
        plan: RuntimeStoragePlan,
        namespace: str,
    ) -> None:
        self._conversation = conversation
        self._execution = execution
        self._namespace = namespace
        self._objects = objects
        self._run_store = run_store
        self._transient_domains = frozenset(
            domain
            for domain in RuntimeDomain
            if plan.route(domain).retention is StorageRetentionMode.TRANSIENT
        )
        self._closed = False

    async def release_execution_handoff(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> bool:
        repository = self._execution.executions

        async def close_and_read(
            _transaction: "StateTransaction | None" = None,
        ) -> tuple[bool, "ExecutionRecord | None"]:
            released = await repository.close_retention(execution_id, tenant_id=tenant_id)
            execution = await repository.get(execution_id, tenant_id=tenant_id) if released else None
            return released, execution

        if type(repository) is ExecutionRepositoryImpl:
            async def read_closed(_transaction: "StateTransaction") -> "ExecutionRecord | None":
                current = await repository.get(execution_id, tenant_id=tenant_id)
                if (current is not None and current.conversation and current.retention_closed
                        and current.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED,
                                               ExecutionStatus.CANCELLED}):
                    return current
                return None

            execution = await repository.state_store.read(read_closed) if terminal_commit else None
            # Keep the closing decision and its metadata in the same transaction;
            # cleanup belongs outside this retryable mutation callback.
            if execution is None:
                released, execution = await repository.state_store.mutate(close_and_read)
            else:
                released = True
        else:
            released, execution = await close_and_read()
        if not released:
            return False
        if execution is not None and execution.session_id is not None:
            sessions = self._conversation.sessions

            async def read_session(_transaction: "StateTransaction"):
                return await sessions.get(execution.session_id, tenant_id=tenant_id)

            session = await sessions.state_store.read(read_session) if execution.conversation else None
            if session is None or session.active_execution_id == execution_id or not execution.conversation:
                await sessions.release_execution(
                    execution.session_id,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    history_unavailable=execution.invalidates_session_history,
                )
        if execution is not None:
            agent_run_ids = tuple(
                agent_run_id(
                    namespace=self._namespace,
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                    agent_run_seq=sequence,
                )
                for sequence in range(1, execution.agent_run_seq + 1)
            )
            await self._run_store.release_staging_many(
                candidate_agent_run_ids=agent_run_ids,
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
        return True

    async def release_session(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation: ConversationCursor | None,
    ) -> None:
        # Forked sessions may still reference transient conversation state.
        del session_id, tenant_id, continuation

    async def close(self) -> None:
        if self._closed:
            return
        await self._objects.clear_transient()
        self._closed = True
        _logger.debug("runtime transient retention closed")


__all__ = []
