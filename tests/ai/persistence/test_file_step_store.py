#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Crash-safe FILE StepStore conformance checks."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent, ToolEffectRecord

from linktools.ai.core.errors import ErrorCode, AIError
from linktools.ai.adapter import DurableFileStepStore


def _run(run_id: str) -> RunRecord:
    return RunRecord(run_id=run_id, conversation_id="c-conversation", parent_run_id=None, agent_name="agent", metadata={"segment_sequence": "1", "agent_name": "agent"}, started_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_file_step_store_round_trip_and_restart(tmp_path: Path) -> None:
    first = DurableFileStepStore(tmp_path, "namespace")
    await first.initialize()
    run = _run("r-run")
    await first.register_run(run)
    await first.append_event(StepEvent(run_id=run.run_id, kind="model_request_started", step_index=1, timestamp=datetime.now(timezone.utc), conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", tool_call_id=None, tool_name=None, error=None, metadata={}))
    await first.save_snapshot(ContinuableSnapshot(run_id=run.run_id, step_index=1, messages=[ModelRequest(parts=[UserPromptPart(content="hello")], conversation_id=run.conversation_id)], conversation_id=run.conversation_id, parent_run_id=None, agent_name="agent", timestamp=datetime.now(timezone.utc)))
    await first.record_tool_effect(ToolEffectRecord(tool_call_id="call", tool_name="tool", run_id=run.run_id, status="started", started_at=datetime.now(timezone.utc), ended_at=None, idempotency_key="key", effect_summary=None))
    with pytest.raises(AIError) as error:
        await first.register_run(run)
    assert error.value.code is ErrorCode.STORAGE_CONFLICT
    await first.close()

    second = DurableFileStepStore(tmp_path, "namespace")
    await second.initialize()
    assert await second.get_run(run_id=run.run_id) == run
    assert len(await second.list_events(run_id=run.run_id)) == 1
    assert await second.latest_snapshot(run_id=run.run_id) is not None
    unresolved = await second.list_unresolved_tool_effects(run_id=run.run_id)
    assert len(unresolved) == 1
    await second.close()


@pytest.mark.asyncio
async def test_file_step_store_rejects_path_identifiers(tmp_path: Path) -> None:
    store = DurableFileStepStore(tmp_path, "namespace")
    await store.initialize()
    with pytest.raises(AIError) as error:
        await store.get_run(run_id="../escape")
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    await store.close()


def test_file_step_store_writes_hashed_run_directory(tmp_path: Path) -> None:
    async def run() -> Path:
        store = DurableFileStepStore(tmp_path, "namespace")
        await store.initialize()
        await store.register_run(_run("r-run"))
        path = next((tmp_path / "steps").rglob("run.json"))
        await store.close()
        return path

    path = asyncio.run(run())
    assert path.parent.name != "r-run"
    assert (path.parent / "run.json").is_file()
