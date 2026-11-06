#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify recovery snapshots already published by upstream StepPersistence."""

from collections.abc import Callable

from pydantic_ai_harness.step_persistence import StepStore

from ..errors import AIError, ErrorCode
from ._official_step_reader import OfficialStepReader
from ._step_archive import StagingAgentRunStore
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord


class CheckpointPersistence:
    """Check upstream durable truth without rebuilding or publishing snapshots."""

    def __init__(self, staging: StagingAgentRunStore, *, native_checkpoints: Callable[[str], bool]) -> None:
        self._staging = staging
        self._native_checkpoints = native_checkpoints
        self._reader = OfficialStepReader()

    def bind_step_store(self, store: StepStore) -> None:
        self._reader.bind_step_store(store)

    @property
    def reader(self) -> OfficialStepReader:
        return self._reader

    @property
    def step_store(self) -> StepStore:
        return self.reader.step_store

    async def materialize_recovery_checkpoint(
        self, *, agent_run_id: str, require_complete: bool,
        captured: tuple[AgentRunRecord, AgentRunCheckpoint | None] | None = None,
    ) -> None:
        if not self._native_checkpoints(agent_run_id):
            run, checkpoint = captured if captured is not None else (
                await self._staging.get_agent_run(agent_run_id=agent_run_id),
                await self._staging.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True),
            )
            if (run is not None and run.agent_run_id != agent_run_id
                    or checkpoint is not None and checkpoint.agent_run_id != agent_run_id
                    or require_complete and (run is None or checkpoint is None or checkpoint.state != "complete")):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        snapshot = await self.reader.latest_snapshot(run_id=agent_run_id, include_interrupted=True)
        run = await self.reader.get_run(run_id=agent_run_id)
        if snapshot is None:
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if run is None or snapshot.run_id != agent_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if require_complete and snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if captured is not None:
            captured_run, checkpoint = captured
            if (captured_run.agent_run_id != agent_run_id
                    or checkpoint is not None and (
                        checkpoint.agent_run_id != agent_run_id
                        or checkpoint.idempotency_key != snapshot.idempotency_key
                        or checkpoint.state != snapshot.state
                    )):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
