#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task observations route Agent metadata to the producing execution reader."""

from __future__ import annotations
from collections.abc import Mapping
from typing import TYPE_CHECKING
from ..core import Principal
from ..errors import AIError, ErrorCode
from ..execution import ExecutionHistoryService, ModelInteractionItem, ModelInteractionReadBoundary, ModelInteractionSubscription, UsageReadCutoff
if TYPE_CHECKING:
    from ..execution import ExecutionService
    from ..runtime.agent_execution import AgentExecutionPort


class TaskHistory:
    def __init__(self, execution: ExecutionService, ports: Mapping[str, AgentExecutionPort[object]]) -> None:
        self._execution = execution
        self._ports = ports

    async def _reader(self, execution_id: str, principal: Principal) -> ExecutionHistoryService | None:
        view = await self._execution.inspect(execution_id, principal=principal)
        if view.binding_kind == "task":
            return None
        port = self._ports.get(execution_id) or self._ports.get(view.root_execution_id)
        if port is None or port.operations.history is None:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED)
        return port.operations.history

    async def capture_model_interaction_cutoffs(self, execution_id: str, *, principal: Principal) -> ModelInteractionReadBoundary:
        reader = await self._reader(execution_id, principal)
        if reader is None:
            return ModelInteractionReadBoundary((), (), False, False)
        return await reader.capture_model_interaction_cutoffs(execution_id, principal=principal)

    async def read_model_interaction_updates(
        self, execution_id: str, *, principal: Principal, after_cutoffs: tuple[UsageReadCutoff, ...] = (), limit: int = 200,
    ) -> tuple[ModelInteractionReadBoundary, tuple[ModelInteractionItem, ...]]:
        reader = await self._reader(execution_id, principal)
        if reader is None:
            return ModelInteractionReadBoundary((), (), False, False), ()
        return await reader.read_model_interaction_updates(execution_id, principal=principal, after_cutoffs=after_cutoffs, limit=limit)

    async def read_model_interaction_metadata(
        self, execution_id: str, *, principal: Principal, agent_run_seq: int,
        after_model_request_seq: int, through_model_request_seq: int, limit: int = 200,
    ) -> tuple[ModelInteractionItem, ...]:
        reader = await self._reader(execution_id, principal)
        if reader is None:
            return ()
        return await reader.read_model_interaction_metadata(
            execution_id, principal=principal, agent_run_seq=agent_run_seq,
            after_model_request_seq=after_model_request_seq, through_model_request_seq=through_model_request_seq, limit=limit,
        )

    async def subscribe_model_interactions(self, execution_id: str, *, principal: Principal) -> ModelInteractionSubscription | None:
        reader = await self._reader(execution_id, principal)
        return None if reader is None else await reader.subscribe_model_interactions(execution_id, principal=principal)
