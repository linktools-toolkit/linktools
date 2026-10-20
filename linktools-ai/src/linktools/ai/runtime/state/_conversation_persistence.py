#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation-owned archive access and committed model-context selection."""

from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelRequest, ModelResponse

from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from ._contracts import LoadedContextMessage, LoadedModelContext
from ._step_archive import InMemoryStepArchive, StateStepArchive
from ._step_contracts import AgentRunStore


class ConversationPersistence:
    """Own conversation data access independently from execution coordination."""

    def __init__(self, archive: AgentRunStore) -> None:
        self.archive = archive

    async def read_conversation_payload(self, payload: StoredPayload) -> bytes:
        archive = self.archive
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.transcript_repository.read_payload(payload)

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        archive = self.archive
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for message in archive.transcript_repository.iter_session_messages(
            history_id,
            tenant_id=tenant_id,
        ):
            yield message


    def iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_messages(
            history_id=history_id,
            agent_run_id=agent_run_id,
            tenant_id=tenant_id,
        )


    async def _iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        archive = self.archive
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            async for message in archive.iter_session_messages(
                history_id,
                tenant_id=tenant_id,
            ):
                yield message
            return
        if isinstance(archive, InMemoryStepArchive):
            async for message in archive.iter_messages(agent_run_id=agent_run_id):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


    async def committed_conversation_message_count(
        self, *, history_id: str | None, agent_run_id: str, tenant_id: str,
    ) -> int:
        archive = self.archive
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await archive.committed_session_message_count(history_id,
                agent_run_id=agent_run_id, tenant_id=tenant_id)
        if isinstance(archive, InMemoryStepArchive):
            return await archive.transcript_message_count(agent_run_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def conversation_message_count(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
    ) -> int:
        archive = self.archive
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await archive.session_message_count(
                history_id,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            return await archive.transcript_message_count(agent_run_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_message_range(
            history_id=history_id,
            agent_run_id=agent_run_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )


    async def _iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        archive = self.archive
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            async for message in archive.iter_session_message_range(
                history_id,
                tenant_id=tenant_id,
                start=start,
                end=end,
            ):
                yield message
            return
        if isinstance(archive, InMemoryStepArchive):
            async for message in archive.iter_message_range(
                agent_run_id=agent_run_id,
                start=start,
                end=end,
            ):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


    async def load_committed_conversation_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        message_count: int | None,
        tenant_id: str,
    ) -> LoadedModelContext:
        archive = self.archive
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if message_count is None:
                return await archive.load_loaded_model_context(
                    owner_id=history_id,
                )
            return await archive.load_committed_session_model_context(
                history_id,
                agent_run_id=agent_run_id,
                message_count=message_count,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            values = await archive.load_model_context(agent_run_id=agent_run_id)
            return LoadedModelContext(
                tuple(
                    LoadedContextMessage(value, None)
                    for value in values
                    if isinstance(value, (ModelRequest, ModelResponse))
                )
            )
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


    async def load_conversation_model_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        message_count: int | None = None,
    ) -> tuple[object, ...]:
        context = await self.load_committed_conversation_context(
            history_id=history_id,
            agent_run_id=agent_run_id,
            message_count=message_count,
            tenant_id=tenant_id,
        )
        return context.model_messages()
