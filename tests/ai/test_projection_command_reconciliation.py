#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone projection commands own their transaction and reconcile lost acknowledgments."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._step_archive import PreparedExecutionProjection
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("operation", ("checkpoint", "projection"))
@pytest.mark.parametrize("lost_ack", (False, True))
async def test_standalone_projection_commands_enlist_authority_and_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, operation: str, lost_ack: bool,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        messages = [ModelResponse(parts=[TextPart("standalone durable content")])]
        checkpoint = AgentRunCheckpoint("run", 1, messages, transcript_message_count_before=0)
        batch = await archive.prepare_checkpoints(run, (checkpoint,))
        projection = PreparedExecutionProjection(
            run=run, events=(), checkpoints=batch.checkpoints,
            base_event_offset=0, base_checkpoint_offset=0,
            target_event_offset=0, target_checkpoint_offset=1,
            target_transcript_message_count=1, producer_generation=1,
        )
        commands = StateCommands(
            state.execution.executions, namespace="archive-boundaries", events=state.execution.events,
            execution_run_store=archive, background_tasks=set(),
        )
        group = archive.state_store.storage_group
        original = group.mutate

        async def mutate(stores, callback):
            assert set(archive.transaction_stores).issubset(stores)
            result = await original(stores, callback)
            if lost_ack:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", mutate)
            if operation == "checkpoint":
                await commands._materialize_prepared_checkpoint_with_reconciliation(
                    archive, run, batch, execution_id="execution",
                )
            else:
                await commands._sync_prepared_projection_with_reconciliation(
                    archive, projection, execution_id="execution",
                )
        restored = await archive.latest_checkpoint(agent_run_id="run")
        assert restored is not None and restored.messages == messages
        assert await archive.get_agent_run(agent_run_id="run") == run
        assert await archive.checkpoint_count(agent_run_id="run") == 1
