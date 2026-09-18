#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation state checkpoint commands."""

from ...errors import AIError, ErrorCode
from ._contracts import (
    ConversationCursor,
    ConversationHistoryRepository,
    HistoryQuality,
    SessionRecord,
    SessionRepository,
)
from ._step_contracts import ContinuableSnapshot, RunRecord
from ._step_archive import StateStepArchive
from ._store import StateStore, StateTransaction


class ConversationStateCommands:
    """Commit the durable conversation snapshot and continuation together."""

    def __init__(
        self,
        state_store: StateStore,
        sessions: SessionRepository,
        steps: StateStepArchive | None,
        histories: ConversationHistoryRepository | None = None,
    ) -> None:
        self._state_store = state_store
        self._sessions = sessions
        self._steps = steps
        self._histories = histories

    async def commit_snapshot_and_advance(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        step_run: RunRecord,
        snapshot: ContinuableSnapshot,
    ) -> SessionRecord:
        if self._steps is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = await self._steps.prepare_snapshots(
            step_run,
            (snapshot,),
        )

        async def mutate(transaction: StateTransaction) -> SessionRecord:
            if self._steps is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = await self._sessions.get_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
            )
            if current.continuation == next_cursor:
                return current
            await self._steps.sync_projection_in_transaction(
                transaction,
                step_run,
                events=(),
                snapshots=prepared.snapshots,
            )
            if self._histories is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            history_id = current.history_id or prepared[0].owner_id
            history = await self._histories.get_in_transaction(
                transaction,
                history_id,
                tenant_id=tenant_id,
            )
            if history is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            quality = (
                "conservative"
                if prepared[0].history_quality is HistoryQuality.CONSERVATIVE
                else "complete"
            )
            return await self._sessions.advance_continuation_in_transaction(
                transaction,
                session_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                expected=expected,
                next_cursor=next_cursor,
                history_quality=quality,
            )

        result = await self._state_store.mutate(mutate)
        return result


__all__ = ["ConversationStateCommands"]
