#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""A terminal transaction between replay reads must not look like corruption."""
from ._runtime_test_helpers import model_features

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, IdempotencyStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint,outcome,corruption", (
    ("start", "success", None), ("start", "failed", None), ("start", "cancelled", None),
    ("start", "success", "active"), ("start", "success", "digest"), ("start", "success", "request"),
    ("start", "success", "identity_version"),
    ("resolve", "success", None), ("resolve", "success", "active"),
    ("resolve", "success", "digest"), ("resolve", "success", "request"),
    ("resolve", "success", "identity_version"),
    ("session", "success", None), ("session", "success", "identity_version"),
))
async def test_start_replay_refreshes_only_a_mixed_terminal_view(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, outcome: str, corruption: str | None,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        del messages, info
        entered.set()
        await release.wait()
        if outcome == "failed":
            raise RuntimeError("offline model failure")
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("terminal-race")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.in_memory()
    async with Persistence.open('terminal-race', storage=storage) as persistence_41, Runtime.attach(persistence=persistence_41, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        if entrypoint == "session":
            agent = await agent.create_session("session")
        execution = await agent.start("work", idempotency_key="same")
        await asyncio.wait_for(entered.wait(), 2)
        repository = runtime._state.execution.idempotency
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
            if corruption == "identity_version":
                return replace(record, request_identity_version=1)
            return record

        monkeypatch.setattr(type(repository), "get", interleaved_get)
        if entrypoint == "resolve":
            async def resolve(binding_digest, request, **kwargs):
                return await runtime._execution_service.resolve_existing(
                    binding_digest, request, binding_contract=kwargs.get("binding_contract"),
                )
            monkeypatch.setattr(runtime._execution_service, "start", resolve)

        async def retry():
            if entrypoint == "session":
                admitted = await runtime._execution_service.lookup_session_admission(
                    "session", principal=runtime.default_principal, idempotency_key="same",
                )
                assert admitted.execution_id == execution.execution_id
                assert admitted.user_input == "work"
                return execution
            return await agent.start("work", idempotency_key="same")

        if outcome == "success" and corruption is None:
            replay = await retry()
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
        else:
            with pytest.raises(AIError) as caught:
                await retry()
            if corruption is not None:
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            elif outcome == "cancelled":
                assert caught.value.code is ErrorCode.EXECUTION_CANCELLED
            else:
                assert caught.value.code.value == (await execution.wait()).result.error_code
        assert reads == 2
