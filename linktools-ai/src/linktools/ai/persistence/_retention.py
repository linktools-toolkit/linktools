#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned transient object and AgentRun retention."""

from ..storage import StorageRetentionMode

from typing import TYPE_CHECKING, Protocol

from linktools.core import environ

from ..core import ExecutionStatus
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore
from .contracts import ConversationCursor, ConversationRepositories, ExecutionRepositories, ExecutionTerminalPublication
from ._execution_repositories import ExecutionRepositoryImpl
from ._plan import PersistenceDomain, PersistencePlan

if TYPE_CHECKING:
    from .contracts import ExecutionRecord
    from .state import StateTransaction

_logger = environ.get_logger("ai.runtime.state.retention")


class _RuntimeObjectRouter(Protocol):
    def object_store(self, domain: PersistenceDomain) -> ObjectStore: ...
    async def release_object_scope(
        self, domain: PersistenceDomain, *, owner_scope: str
    ) -> None: ...
    async def clear_transient(self) -> None: ...


class RuntimeRetentionController:
    def __init__(
        self,
        *,
        conversation: ConversationRepositories,
        execution: ExecutionRepositories,
        objects: _RuntimeObjectRouter,
        plan: PersistencePlan,
        namespace: str,
    ) -> None:
        self._conversation = conversation
        self._execution = execution
        self._namespace = namespace
        self._objects = objects
        self._transient_domains = frozenset(
            domain
            for domain in PersistenceDomain
            if plan.route(domain).retention is StorageRetentionMode.TRANSIENT
        )
        self._closed = False
        self._terminal_publications: dict[str, ExecutionRecord | ExecutionTerminalPublication] = {}

    def confirm_terminal_publication(self, publication: "ExecutionRecord | ExecutionTerminalPublication") -> None:
        """Retain a publication receipt only after its complete commit is confirmed."""
        execution = publication.execution if isinstance(publication, ExecutionTerminalPublication) else publication
        if execution.status not in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            raise ValueError("publication confirmation requires a terminal execution")
        self._terminal_publications[execution.execution_id] = publication

    def terminal_publication_confirmed(self, execution: "ExecutionRecord") -> bool:
        """A source snapshot can reuse this process's exact confirmed publication."""
        publication = self._terminal_publications.get(execution.execution_id)
        return (publication.execution if isinstance(publication, ExecutionTerminalPublication) else publication) == execution

    def terminal_publication(
        self, execution_id: str, *, tenant_id: str,
    ) -> ExecutionTerminalPublication | None:
        """Return this owner's confirmed full publication, never a cached source guess."""
        publication = self._terminal_publications.get(execution_id)
        if isinstance(publication, ExecutionTerminalPublication) and self._execution.executions.tenant_id == tenant_id:
            return publication
        return None

    async def release_execution_handoff(
        self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False,
    ) -> bool:
        repository = self._execution.executions
        async def read_source(transaction: "StateTransaction") -> "ExecutionRecord | None":
            return await repository.get_in_transaction(transaction, execution_id, tenant_id=tenant_id)

        # A complete terminal publication already owns this immutable identity
        # and admission provenance. Closing still reads current dependency holds.
        retained = (self._terminal_publications.get(execution_id)
                    if type(repository) is ExecutionRepositoryImpl and repository.tenant_id == tenant_id else None)
        publication = retained if isinstance(retained, ExecutionTerminalPublication) else None
        if publication is not None:
            retained = publication.execution
        if retained is None:
            retained = await repository.state_store.read(read_source)
        if retained is not None and retained.status not in {
            ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
        }:
            return False
        if (retained is not None and not retained.retention_closed and retained.session_id is not None
                and retained.conversation and not self.terminal_publication_confirmed(retained)):
            sessions = self._conversation.sessions
            await sessions.state_store.read(
                lambda transaction: sessions.pending_turn_commit_in_transaction(
                    transaction, session_id=retained.session_id, execution_id=execution_id,
                    allow_unadmitted=retained.started_at is None,
                )
            )

        async def close_and_read() -> tuple[bool, "ExecutionRecord | None"]:
            released = await repository.close_retention(execution_id, tenant_id=tenant_id)
            execution = await repository.get(execution_id, tenant_id=tenant_id) if released else None
            return released, execution

        if retained is None:
            # No source record remains to close; orphaned transient scopes still
            # need cleanup, without a second lookup that could observe new work.
            released, execution = True, None
        elif type(repository) is ExecutionRepositoryImpl:
            execution = (retained if terminal_commit and retained.conversation and retained.retention_closed else None)
            # Keep the closing decision and its metadata in the same transaction;
            # cleanup belongs outside this retryable mutation callback.
            if execution is None:
                try:
                    released, execution = await repository.state_store.mutate(
                        lambda transaction: repository.close_retention_in_transaction(
                            transaction, execution_id, tenant_id=tenant_id,
                            observed=(None if publication is None or publication.execution.dependency_hold_ids
                                      else publication.observed),
                        )
                    )
                except AIError as error:
                    if publication is None or error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    released, execution = await repository.state_store.mutate(
                        lambda transaction: repository.close_retention_in_transaction(
                            transaction, execution_id, tenant_id=tenant_id,
                        )
                    )
            else:
                released = True
        else:
            released, execution = await close_and_read()
        if not released:
            return False
        if (execution is not None and execution.session_id is not None
                and not (publication is not None and publication.session_released)):
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
                    require_turn_commit=execution.conversation,
                )
        for domain in self._transient_domains:
            await self._objects.release_object_scope(
                domain, owner_scope=f"execution:{execution_id}"
            )
        self._terminal_publications.pop(execution_id, None)
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
        self._terminal_publications.clear()
        self._closed = True
        _logger.debug("runtime transient retention closed")


__all__ = []
