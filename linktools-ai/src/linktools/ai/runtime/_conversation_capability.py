#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation-owned context selection and ordinary-turn admission requests."""

from collections.abc import Awaitable, Callable
from copy import copy
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext

from ..capability import AgentContext
from ..core import ExecutionLineageKind, agent_run_id as make_agent_run_id
from ..errors import AIError, ErrorCode
from ..storage import TurnCoordination
from .service_api import ExecutionRequest
from .state import RuntimeDomain, RuntimeStorageRoute
from .state._contracts import ExecutionRecord, ExecutionStartReservation, LoadedModelContext, SessionRecord
from .state._step_contracts import AgentRunStore
from .state._steps import RuntimeAgentRunStore, StateStepArchive

if TYPE_CHECKING:
    from ._run_hooks import AgentRunCapture

AppT = TypeVar("AppT")


class ConversationCapability(AbstractCapability[AgentContext[AppT]]):
    """Retain actual context for continuation and execution forks.

    Runtime confirms the candidate projection with the terminal execution.
    A complete configured cache may support a pending turn; the core owns
    admission eligibility, effects, fencing and durable reconciliation.
    """

    id: str | None = "runtime.conversation"
    _capture: "AgentRunCapture | None" = None

    def __init__(
        self, *, storage: RuntimeStorageRoute | None = None, cache: object | None = None,
    ) -> None:
        if storage is not None and not isinstance(storage, RuntimeStorageRoute):
            raise TypeError("storage must be RuntimeStorageRoute")
        self._storage = storage
        self._cache = cache

    @property
    def storage(self) -> RuntimeStorageRoute | None:
        return self._storage

    @property
    def cache(self) -> object | None:
        return self._cache

    def bind(self, capture: "AgentRunCapture") -> "ConversationCapability[AppT]":
        """Bind isolated run resources without changing the native capability type."""
        bound = copy(self)
        bound._capture = capture
        return bound

    async def after_run(
        self, ctx: RunContext[AgentContext[AppT]], *, result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        if self._capture is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._capture.after_run(ctx, result=result)
        await self._capture.context.save_checkpoint(
            ctx, messages=result.all_messages(),
            state="interrupted" if isinstance(result.output, DeferredToolRequests) else "complete",
        )
        return result

    async def on_run_error(
        self, ctx: RunContext[AgentContext[AppT]], *, error: BaseException,
    ) -> AgentRunResult[Any]:
        if self._capture is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
        self._capture.recorder.finish_transcript(interrupted=True)
        await self._capture.context.save_checkpoint(
            ctx, messages=self._capture.live_messages or ctx.messages, state="interrupted",
        )
        raise error

    @classmethod
    async def prepare_turn(
        cls, request: ExecutionRequest, reservation: ExecutionStartReservation,
        session: SessionRecord | None, coordination: TurnCoordination | None, *,
        start_pending: Callable[[ExecutionRequest, ExecutionStartReservation, SessionRecord | None,
                                 TurnCoordination | None], Awaitable[str | None]],
        dependency_hold_id: str | None,
    ) -> str | None:
        if not reservation.execution.conversation or dependency_hold_id is not None or coordination is None:
            return None
        return await start_pending(request, reservation, session, coordination)

    @classmethod
    async def _messages(
        cls, store: AgentRunStore, agent_run_id: str, *, include_interrupted: bool = False,
    ) -> list[ModelMessage]:
        checkpoint = await store.latest_checkpoint(
            agent_run_id=agent_run_id, include_interrupted=include_interrupted,
        )
        if checkpoint is None:
            raise LookupError(agent_run_id)
        return list(checkpoint.messages if checkpoint.context_messages is None else checkpoint.context_messages)

    @classmethod
    async def load_session_context(
        cls, *, execution: ExecutionRecord, session: SessionRecord | None,
        history_id: str, tenant_id: str, run_store: RuntimeAgentRunStore,
        conversation_archive: AgentRunStore,
    ) -> LoadedModelContext:
        if not execution.conversation:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        if isinstance(conversation_archive, StateStepArchive):
            owner = history_id
        elif session is not None and session.continuation is not None:
            owner = session.continuation.agent_run_id
        else:
            return LoadedModelContext(())
        if isinstance(run_store, RuntimeAgentRunStore) and session is not None and session.continuation is not None:
            return await run_store.conversation_persistence.load_committed_conversation_context(
                history_id=history_id, agent_run_id=session.continuation.agent_run_id,
                message_count=session.continuation.message_count, tenant_id=tenant_id,
            )
        return await run_store.load_loaded_model_context(RuntimeDomain.CONVERSATION, owner)

    @classmethod
    async def load_history(
        cls, execution: ExecutionRecord, *, session: SessionRecord | None,
        namespace: str, tenant_id: str, run_store: RuntimeAgentRunStore,
        execution_run_store: AgentRunStore, conversation_run_store: AgentRunStore,
        recovery_run_store: AgentRunStore,
        load_execution: Callable[[str], Awaitable[ExecutionRecord | None]],
    ) -> list[object]:
        if not execution.conversation:
            return []
        if (
            execution.lineage_kind is ExecutionLineageKind.SESSION_RESUME
            and session is not None
            and (
                session.history_id is not None
                or (
                    session.continuation is not None
                    and session.continuation.history_id is not None
                )
            )
        ):
            history_id = session.history_id or session.continuation.history_id
            try:
                return [
                    message
                    async for message in run_store.conversation_persistence.iter_session_messages(
                        history_id,
                        tenant_id=tenant_id,
                    )
                ]
            except AIError as error:
                if error.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
                raise
        if execution.conversation_agent_run_id is not None:
            try:
                return list(
                    await cls._messages(
                        conversation_run_store,
                        execution.conversation_agent_run_id,
                        include_interrupted=True,
                    )
                )
            except LookupError as error:
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
        if execution.fork_base_execution_id is None:
            return []
        base = await load_execution(execution.fork_base_execution_id)
        if base is not None and not base.conversation:
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        if base is None or base.agent_run_seq < 1:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        agent_run_id = make_agent_run_id(
            namespace=namespace,
            tenant_id=tenant_id,
            execution_id=base.execution_id,
            agent_run_seq=base.agent_run_seq,
        )
        if not base.durable_trace and base.session_id is None:
            execution_run_store = recovery_run_store
            if isinstance(execution_run_store, StateStepArchive):
                checkpoint = await execution_run_store.latest_checkpoint(agent_run_id=agent_run_id)
                if checkpoint is None:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
                run = await execution_run_store.get_agent_run(agent_run_id=agent_run_id)
                if (run is None or checkpoint.agent_run_id != agent_run_id
                        or not await execution_run_store.verify_checkpoint_projection(
                            agent_run_id=agent_run_id, checkpoint=checkpoint,
                        )
                        or await execution_run_store.transcript_message_count_for_run(run) != len(checkpoint.messages)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return list(checkpoint.messages if checkpoint.context_messages is None else checkpoint.context_messages)
        elif not base.durable_trace and isinstance(conversation_run_store, StateStepArchive):
            cursor = None if session is None else session.continuation
            if (cursor is None or cursor.history_id is None or cursor.message_count is None
                    or cursor.agent_run_id != agent_run_id):
                raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
            try:
                context = await conversation_run_store.load_committed_session_model_context(
                    cursor.history_id, agent_run_id=agent_run_id,
                    message_count=cursor.message_count, tenant_id=tenant_id,
                )
                return list(context.model_messages())
            except AIError as error:
                if error.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE:
                    raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error
                raise
        try:
            return list(await cls._messages(execution_run_store, agent_run_id))
        except LookupError as error:
            raise AIError(ErrorCode.EXECUTION_HISTORY_UNAVAILABLE) from error


__all__ = ["ConversationCapability"]
