#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native attempt replacement settles interrupted Trace without inventing observations."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime
from tests.ai._runtime_process_test_helpers import _NAMESPACE, _RuntimeProcess, _capabilities
from tests.ai._runtime_test_helpers import _Models, model_features, native_checkpoint


@pytest.mark.asyncio
async def test_hard_killed_native_attempt_marks_its_original_request_interrupted(tmp_path: Path) -> None:
    database = tmp_path / "trace-attempt.db"
    owner = await asyncio.to_thread(_RuntimeProcess, database, auto_recover=False)
    successor = None
    try:
        execution_id = owner.initial["execution_id"]
        assert owner.initial["model_calls"] == 1
        await asyncio.to_thread(owner.request, "flush_trace")
        owner.process.kill()
        await asyncio.to_thread(owner.process.join, 10)
        assert not owner.process.is_alive()

        async with Persistence.open(
            _NAMESPACE, storage=PersistenceStorage.sqlite(database), read_only=True,
        ) as state:
            checkpoint = await state.execution.executions.get(execution_id, tenant_id="default")
            original_run_id = checkpoint.recovery.agent_run_id
            trace = state.run_store.read_store(PersistenceDomain.TRACE)
            observed = await trace.list_model_interactions(agent_run_id=original_run_id)
            assert len(observed) == 1 and observed[0].status == "RUNNING"
            original = observed[0]
            assert original.model_request_seq == 1 and original.response_context is None
            assert (original.request_context is None) is (original.request_envelope is None)
            messages = [message async for message in trace.iter_messages(agent_run_id=original_run_id)]
            assert any(isinstance(part, UserPromptPart) and part.content == "Wait"
                       for message in messages for part in message.parts)

        async def unused_model(messages, info):
            raise AssertionError("the intermediate admission must not call the provider")
            yield "unreachable"

        async with Persistence.open(
            _NAMESPACE, storage=PersistenceStorage.sqlite(database),
        ) as state, Runtime.attach(
            persistence=state,
            features=model_features(_Models(FunctionModel(stream_function=unused_model)), _capabilities(conversation=True)),
            auto_recover=False,
        ) as admitted:
            current = await state.execution.executions.get(execution_id, tenant_id="default")
            pending = await admitted._execution_service.runtime_backend()._commit_recovery_resume(current)
            assert pending.recovery.agent_run_id != original_run_id
            assert pending.recovery.source_agent_run_id == original_run_id
            native = native_checkpoint().store
            assert await native.get_run(run_id=pending.recovery.agent_run_id) is None
            assert await native.latest_snapshot(run_id=pending.recovery.agent_run_id, include_interrupted=True) is None

        successor = await asyncio.to_thread(
            _RuntimeProcess, database, execution_id=execution_id, auto_recover=False,
        )
        await asyncio.to_thread(successor.request, "recover_keyed")
        assert (await asyncio.to_thread(successor.request, "await_model"))["model_calls"] == 1
        result = await asyncio.to_thread(successor.request, "finish")
        assert result["status"] == "SUCCEEDED" and result["model_calls"] == 1
        await asyncio.to_thread(successor.close, check=True)

        async with Persistence.open(
            _NAMESPACE, storage=PersistenceStorage.sqlite(database), read_only=True,
        ) as state:
            checkpoint = await state.execution.executions.get(execution_id, tenant_id="default")
            assert checkpoint.recovery.agent_run_id not in {original_run_id, pending.recovery.agent_run_id}
            execution = await state.execution.executions.get(execution_id, tenant_id="default")
            assert execution.agent_run_seq == 3
            trace = state.run_store.read_store(PersistenceDomain.TRACE)
            previous = await trace.list_model_interactions(agent_run_id=original_run_id)
            current = await trace.list_model_interactions(agent_run_id=checkpoint.recovery.agent_run_id)
            assert len(current) == 1 and current[0].status == "SUCCEEDED"
            assert current[0].response_context is not None
            assert len(previous) == 1 and previous[0].status == "INTERRUPTED"
            assert replace(previous[0], status="RUNNING") == original
            assert previous[0].stream_attempts == original.stream_attempts
            assert [message async for message in trace.iter_messages(agent_run_id=original_run_id)] == messages
    finally:
        await asyncio.to_thread(owner.close)
        if successor is not None:
            await asyncio.to_thread(successor.close)
