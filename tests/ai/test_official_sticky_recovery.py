#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A sticky history processor cannot disable a valid official continuation."""

import asyncio

import pytest
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence._official_step_reader import OfficialStepReader

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, seed_native_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", (False, True))
async def test_cold_runtime_continues_sticky_snapshot_with_source_owned_history(tmp_path, conversation) -> None:
    blocked = asyncio.Event()
    calls = []

    async def stopped(messages, _info):
        blocked.set()
        await asyncio.Event().wait()

    async def resumed(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("continued")])

    def features(model):
        result = RuntimeFeatures("sticky-native")
        result.agent("default", allow_tools=())
        result.capability(ExecutionCapability(retain_history=conversation), scope="runtime")
        result.capability(ProcessHistory(lambda _messages: [ModelRequest(parts=[UserPromptPart("sticky context")])]),
                          id="sticky", scope="runtime")
        result.capability(StepPersistence(store=SqliteStepStore(database=tmp_path / "steps.db"), capture_frontier=True),
                          scope="runtime")
        return model_features(_Models(_UsageFunctionModel(model)), result)

    async with Persistence.open("sticky-native", storage=PersistenceStorage.from_root(tmp_path / "runtime", exclusive_writer=True)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features(stopped), auto_recover=False) as runtime:
            agent = runtime.agents.get()
            runner = await agent.create_session("session") if conversation else agent
            execution = await runner.start("question", idempotency_key="start")
            await asyncio.wait_for(blocked.wait(), 5)
            await runtime.handoff()
            checkpoint = await persistence.execution.executions.get(execution.execution_id, tenant_id="default")
            source = checkpoint.recovery.agent_run_id
            reader = runtime._run_store.read_store(PersistenceDomain.CHECKPOINT)
            snapshot = await reader.latest_snapshot(run_id=source, include_interrupted=True)
            native_run = await reader.get_run(run_id=source)
            observed = await reader.latest_checkpoint(agent_run_id=source, include_interrupted=True)
            assert snapshot is not None and native_run is not None and observed is not None
            transcript_run = native_run.metadata["linktools.transcript_run_id"]
            first = next(index for index, message in enumerate(snapshot.messages) if message.run_id == transcript_run)
            assert observed.messages == snapshot.messages[first:]
            assert observed.context_messages == snapshot.messages
            assert [part.content for message in observed.messages for part in message.parts
                    if isinstance(part, UserPromptPart)] == ["sticky context"]
    async with Persistence.open("sticky-native", storage=PersistenceStorage.from_root(tmp_path / "runtime", exclusive_writer=True)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features(resumed), auto_recover=False) as runtime:
            reader = runtime._run_store.read_store(PersistenceDomain.CHECKPOINT)
            assert await reader.latest_snapshot(run_id=source, include_interrupted=True) == snapshot
            await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal,
                                                         idempotency_key="recover")
            outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=5)
            assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
            assert calls and "sticky context" in str(calls[-1])
            if conversation:
                page = await runtime.history.messages("session", principal=runtime.default_principal, limit=100)
                parts = [part for message in page.items for part in message.parts]
                assert [part.content for part in parts if isinstance(part, UserPromptPart)] == ["sticky context"]
                assert [part.content for part in parts if isinstance(part, TextPart)] == ["continued"]


@pytest.mark.asyncio
async def test_cold_native_context_remains_available_without_raw_transcript_lineage(tmp_path) -> None:
    database = tmp_path / "trimmed-native.db"
    messages = [ModelRequest(parts=[UserPromptPart("retained sticky context")], run_id="compacted-prefix")]
    await seed_native_snapshot(SqliteStepStore(database=database), run_id="run", messages=messages)
    reader = OfficialStepReader(SqliteStepStore(database=database))
    with pytest.raises(AIError) as unavailable:
        await reader.latest_checkpoint(agent_run_id="run", include_interrupted=True)
    assert unavailable.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
    assert (await reader.load_loaded_model_context(owner_id="run")).model_messages() == tuple(messages)
    checkpoint = await reader.latest_checkpoint(agent_run_id="run", include_transcript=False)
    assert checkpoint is not None and checkpoint.messages == []
    assert checkpoint.context_messages == messages
    assert (await reader.latest_snapshot(run_id="run")).messages == messages
