#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation retains its compacted context independently of native recovery."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_terminal_archive_batch import _split_sqlite_storage


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_native_checkpoint_does_not_replace_conversation_compaction(tmp_path, split) -> None:
    steps = InMemoryStepStore()
    summaries = []
    provider_views = []
    work_calls = []
    completed = False

    async def work(_context: AgentContext[None], index: int) -> str:
        work_calls.append(index)
        return "old result " + str(index)

    async def respond(messages, _info):
        nonlocal completed
        if "context summarization assistant" in str(messages):
            summaries.append(messages)
            return ModelResponse(parts=[TextPart("trusted-runtime-summary")])
        provider_views.append(messages)
        if not completed and len(work_calls) < 14:
            index = len(work_calls)
            return ModelResponse(parts=[TextPart("old reasoning " * 1500 if index < 3 else "recent"),
                ToolCallPart("work", {"index": index}, tool_call_id=f"call-{index}")])
        completed = True
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("native-conversation-compaction")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistence(store=steps), scope="runtime")
    features.tool(work, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("work",))
    model = _UsageFunctionModel(respond, profile={"context_window": 3000})
    model_features(_Models(model), features)
    storage = _split_sqlite_storage(tmp_path / "split.db") if split else PersistenceStorage.in_memory()
    async with Persistence.open("native-conversation-compaction", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            session = await runtime.agents.get().create_session("session")
            first = await session.start("work", idempotency_key="first")
            assert (await first.wait()).result.status is ExecutionStatus.SUCCEEDED
            assert summaries
            count = len(summaries)
            first_run = (await steps.list_runs())[0]
            native = await steps.latest_snapshot(run_id=first_run.run_id)
            assert "trusted-runtime-summary" not in str(native.messages)
            assert "old result 0" in str(native.messages)
            second = await session.start("continue", idempotency_key="second")
            assert (await second.wait()).result.status is ExecutionStatus.SUCCEEDED
            assert "trusted-runtime-summary" in str(provider_views[-1])
            assert len(summaries) == count
