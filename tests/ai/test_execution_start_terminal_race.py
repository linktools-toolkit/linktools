#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A terminal transaction between replay reads must not look like corruption."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, IdempotencyStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,corruption", (
    ("success", None), ("failed", None), ("cancelled", None),
    ("success", "active"), ("success", "digest"), ("success", "request"),
))
async def test_start_replay_refreshes_only_a_mixed_terminal_view(
    monkeypatch: pytest.MonkeyPatch, outcome: str, corruption: str | None,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        del messages, info
        entered.set()
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("offline model failure")
        return ModelResponse(parts=[TextPart("done")])

    group = CapabilityGroup("terminal-race")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "terminal-race", models=_Models(_UsageFunctionModel(model)),
        storage=storage, capabilities=(group,),
    ) as runtime:
        agent = runtime.agents.get()
        execution = await agent.start("work", idempotency_key="same")
        await asyncio.wait_for(entered.wait(), 2)
        repository = storage.execution.idempotency
        original_get = type(repository).get
        reads = 0
        stale = None

        async def interleaved_get(self, *args, **kwargs):
            nonlocal reads, stale
            record = await original_get(self, *args, **kwargs)
            if self is not repository or record is None or record.resource_id != execution.execution_id:
                return record
            reads += 1
            if reads == 1:
                stale = record
                assert record.status is IdempotencyStatus.STARTED
                if outcome == "cancelled":
                    await execution.cancel()
                else:
                    release.set()
                await execution.wait(timeout_seconds=3)
                return record
            if corruption == "active":
                return stale
            if corruption == "digest":
                return replace(record, result_digest="0" * 64)
            if corruption == "request":
                return replace(record, request_digest="0" * 64)
            return record

        monkeypatch.setattr(type(repository), "get", interleaved_get)
        if outcome == "success" and corruption is None:
            replay = await agent.start("work", idempotency_key="same")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
        else:
            with pytest.raises(AIError) as caught:
                await agent.start("work", idempotency_key="same")
            if corruption is not None:
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            elif outcome == "cancelled":
                assert caught.value.code is ErrorCode.EXECUTION_CANCELLED
            else:
                assert caught.value.code.value == (await execution.wait()).result.error_code
        assert reads == 2
