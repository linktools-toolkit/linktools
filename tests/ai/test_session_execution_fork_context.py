#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Session-associated execution forks keep the existing continuation CAS boundary."""
from ._runtime_test_helpers import model_features
from copy import deepcopy

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.model.messages import encode_model_messages

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("ordinary", "pending", "trace"))
@pytest.mark.parametrize("later", (False, True))
async def test_session_execution_fork_preserves_committed_cursor_and_rejects_changed_head(tmp_path, mode, later):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("session-execution-fork")
    group.agent("default", model="default", allow_tools=())
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if mode == "trace":
        group.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('session-execution-fork', storage=PersistenceStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if mode == 'pending' else None)) as persistence_35, Runtime.open(persistence=persistence_35, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        session = await runtime.agents.get().create_session("session")
        first = await session.start("first")
        await first.wait(timeout_seconds=5)
        if later:
            second = await session.start("future")
            await second.wait(timeout_seconds=5)
            call_count = len(calls)
            with pytest.raises(AIError) as raised:
                await first.fork("fork")
            assert raised.value.code is ErrorCode.SESSION_BUSY
            assert len(calls) == call_count
        else:
            fork = await first.fork("fork")
            assert (await fork.wait(timeout_seconds=5)).result.status.value == "SUCCEEDED"
            final = await session.start("final")
            assert (await final.wait(timeout_seconds=5)).result.status.value == "SUCCEEDED"
            assert encode_model_messages(tuple(calls[-1][:len(calls[-2])])) == encode_model_messages(tuple(calls[-2]))
            assert len(calls[-1]) == len(calls[-2]) + 2
