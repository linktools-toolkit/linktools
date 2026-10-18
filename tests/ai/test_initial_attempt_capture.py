#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preparation uses captured control values without claiming producer authority."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, SessionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._execution_context import ExecutionInputContext
from linktools.ai.runtime._local import LocalExecutionBackend
from pydantic_ai.messages import ModelResponse, TextPart

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("cancel", "close", "metadata"))
async def test_changes_after_launch_are_observed_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    launched, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("launch-change")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = RuntimeStorage.sqlite(tmp_path / "state.db")
    second_state = RuntimeStorage.sqlite(tmp_path / "state.db")
    original = LocalExecutionBackend._run

    async def delayed(worker, *args, **kwargs):
        launched.set()
        await release.wait()
        return await original(worker, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_run", delayed)
    async with Runtime.open("launch-change", storage=first_state, models=models, runtime_features=group) as first, Runtime.open(
        "launch-change", storage=second_state, models=models, runtime_features=group,
    ) as second:
        session = await first.agents.get().create_session("session", metadata={"marker": "before"})
        execution = await session.start("question", idempotency_key="same")
        await asyncio.wait_for(launched.wait(), 5)
        try:
            remote = second.agents.get().session("session")
            if change == "cancel":
                await (await second.executions.get(execution.execution_id)).cancel(idempotency_key="remote-cancel")
            elif change == "close":
                await remote.close(force=True, idempotency_key="remote-close", wait_timeout_seconds=1)
            else:
                observed = await second_state.conversation.sessions.get("session", tenant_id=second.tenant_id)
                await remote.update(expected_revision=observed.revision, metadata={"marker": "after"},
                                    idempotency_key="remote-update")
        finally:
            release.set()
        result = (await asyncio.wait_for(execution.wait(), 5)).result
        if change == "metadata":
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(calls) == 1
            current = await first_state.execution.executions.get(execution.execution_id, tenant_id=first.tenant_id)
            context = ExecutionInputContext.from_payload(current.input_context.payload.decode())
            assert context.session_metadata == {"marker": "after"}
        else:
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == []
        session_record = await second_state.conversation.sessions.get("session", tenant_id=second.tenant_id)
        assert session_record.active_execution_id is None
        if change == "close":
            assert session_record.status is SessionStatus.CLOSED


@pytest.mark.asyncio
async def test_other_runtime_claims_before_delayed_worker_without_duplicate_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched, release_first, provider_entered, release_provider = (asyncio.Event() for _ in range(4))
    calls, workers = [], []

    async def model(messages, info):
        calls.append(messages)
        provider_entered.set()
        await release_provider.wait()
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("launch-winner")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = RuntimeStorage.sqlite(tmp_path / "state.db")
    second_state = RuntimeStorage.sqlite(tmp_path / "state.db")
    original = LocalExecutionBackend._run

    async def delayed(worker, *args, **kwargs):
        workers.append(worker)
        if len(workers) == 1:
            launched.set()
            await release_first.wait()
        return await original(worker, *args, **kwargs)

    monkeypatch.setattr(LocalExecutionBackend, "_run", delayed)
    async with Runtime.open("launch-winner", storage=first_state, models=models, runtime_features=group) as first, Runtime.open(
        "launch-winner", storage=second_state, models=models, runtime_features=group,
    ) as second:
        await first.agents.get().create_session("session")
        one = await first.agents.get().session("session").start("question", idempotency_key="same")
        await asyncio.wait_for(launched.wait(), 5)
        try:
            two = await second.agents.get().session("session").start("question", idempotency_key="same")
            assert two.execution_id == one.execution_id
            await asyncio.wait_for(provider_entered.wait(), 5)
        finally:
            release_first.set()
            release_provider.set()
        results = await asyncio.wait_for(asyncio.gather(one.wait(), two.wait()), 5)
        assert all(value.result.status is ExecutionStatus.SUCCEEDED for value in results)
        assert len(calls) == 1
        current = await first_state.execution.executions.get(one.execution_id, tenant_id=first.tenant_id)
        assert current.agent_run_seq == 1
        assert (await first_state.conversation.sessions.get("session", tenant_id=first.tenant_id)).active_execution_id is None
