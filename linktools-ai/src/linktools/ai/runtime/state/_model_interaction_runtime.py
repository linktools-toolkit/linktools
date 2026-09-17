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
from ._steps import (
    CapturedExecutionProjection,
    ExecutionTerminalSealPlan,
    RuntimeStepStore,
    _ProjectionOffset,
)

if TYPE_CHECKING:
    from ._steps import _RunProjectionFlight


class ModelInteractionRuntimeStepStore(RuntimeStepStore):
    """Align staged interaction offsets with the durable request high-water."""

    async def capture_execution_projection(
        self,
        step_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _RunProjectionFlight] | None":
        await self._align_execution_interaction_offset(step_run_id)
        return await super().capture_execution_projection(step_run_id)

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _RunProjectionFlight,
        *,
        execution_id: str,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if (
            isinstance(archive, ModelInteractionInMemoryStepArchive)
            and captured.interactions
        ):
            prepared = await archive.prepare_interactions(
                captured.run,
                captured.interactions,
                lambda digest: self._staging.staged_payload(
                    captured.run.run_id,
                    digest,
                ),
                source_messages=(
                    captured.snapshots[-1].messages if captured.snapshots else None
                ),
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
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        for run_id in dict.fromkeys(run_ids):
            await self._align_execution_interaction_offset(run_id)
        return await super().prepare_execution_terminal_seal(
            execution_id=execution_id,
            run_ids=run_ids,
            binding_digest=binding_digest,
        )

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        await super().materialize_from_recovery(
            target=target,
            step_run_id=step_run_id,
            execution_id=execution_id,
        )
        if target is RuntimeDomain.EXECUTION:
            await self._align_execution_interaction_offset(step_run_id)

    async def _align_execution_interaction_offset(self, run_id: str) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, ModelInteractionStateStepArchive):
            return
        head: ExecutionRunSealHead = await archive.execution_history_head_record(run_id)
        async with self._history_lock.hold(run_id):
            offset = self._projection_offsets.setdefault(run_id, _ProjectionOffset())
            offset.interactions = max(offset.interactions, head.interaction_count)


__all__ = ["ModelInteractionRuntimeStepStore"]
