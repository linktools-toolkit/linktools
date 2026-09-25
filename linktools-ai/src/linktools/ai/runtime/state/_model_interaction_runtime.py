#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime coordination for model-interaction step persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ._contracts import ExecutionRunSealHead
from ._model_interaction_store import (
    ModelInteractionInMemoryStepArchive,
    ModelInteractionStateStepArchive,
)
from ._plan import RuntimeDomain
from ._step_archive import (
    CapturedExecutionProjection,
    ExecutionTerminalSealPlan,
    _ProjectionOffset,
)
from ._steps import RuntimeAgentRunStore

if TYPE_CHECKING:
    from ._step_archive import _AgentRunProjectionFlight


class ModelInteractionRuntimeAgentRunStore(RuntimeAgentRunStore):
    """Align staged interaction offsets with the durable request high-water."""

    async def capture_execution_projection(
        self,
        agent_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _AgentRunProjectionFlight] | None":
        await self._align_execution_interaction_offset(agent_run_id)
        return await super().capture_execution_projection(agent_run_id)

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _AgentRunProjectionFlight,
        *,
        execution_id: str,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if (
            isinstance(archive, ModelInteractionInMemoryStepArchive)
            and captured.interactions
        ):
            local_count = (
                len(captured.snapshots[-1].messages)
                if captured.snapshots
                else 0
            )
            prepared = await archive.prepare_interactions(
                captured.run,
                captured.interactions,
                lambda digest: self._staging.staged_payload(
                    captured.run.agent_run_id,
                    digest,
                ),
                local_message_count=local_count,
            )
            captured = replace(captured, interactions=prepared)  # type: ignore[arg-type]
        await super().commit_captured_execution_projection(
            captured,
            flight,
            execution_id=execution_id,
        )

    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        agent_run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        for agent_run_id in dict.fromkeys(agent_run_ids):
            await self._align_execution_interaction_offset(agent_run_id)
        return await super().prepare_execution_terminal_seal(
            execution_id=execution_id,
            agent_run_ids=agent_run_ids,
            binding_digest=binding_digest,
        )

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        agent_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        await super().materialize_from_recovery(
            target=target,
            agent_run_id=agent_run_id,
            execution_id=execution_id,
        )
        if target is RuntimeDomain.EXECUTION:
            await self._align_execution_interaction_offset(agent_run_id)

    async def _align_execution_interaction_offset(self, agent_run_id: str) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, ModelInteractionStateStepArchive):
            return
        head: ExecutionRunSealHead = await archive.execution_history_head_record(agent_run_id)
        async with self._history_lock.hold(agent_run_id):
            offset = self._projection_offsets.setdefault(agent_run_id, _ProjectionOffset())
            offset.interactions = max(offset.interactions, head.interaction_count)


__all__ = ["ModelInteractionRuntimeAgentRunStore"]
