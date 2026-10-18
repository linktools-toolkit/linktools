#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A terminal standalone run retains exact model context without a durable trace."""

from ._runtime_test_helpers import workspace_features

import asyncio
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from sqlalchemy import event

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionInputContext, Runtime, RuntimeStorage, ConversationCapability
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_archive import _decode_step, _encode_step
from linktools.ai.workspace import Workspace
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.runtime._message import encode_model_messages

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_pending_turn_commit import _turn_state
from .test_pending_turn_sessionless import _commit, _standalone


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("cached", (False, True), ids=("no-cache", "cache"))
async def test_standalone_fork_preserves_actual_tools_and_imported_context_after_reopen(
    tmp_path: Path, backend: str, cached: bool,
) -> None:
    calls = []
    tool_calls = []

    async def lookup(context: AgentContext[None], value: str) -> str:
        tool_calls.append(value)
        return "returned:" + value

    async def model(messages, info):
        calls.append(deepcopy(messages))
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if prompts[-1] == "source" and not returns:
            return ModelResponse(parts=[
                ToolCallPart("lookup", {"value": "first"}, tool_call_id="first-call"),
                ToolCallPart("lookup", {"value": "second"}, tool_call_id="second-call"),
            ])
        return ModelResponse(parts=[TextPart("answer:" + prompts[-1])])

    group = RuntimeFeatures("standalone-fork")
    group.tool(lookup, effect_policy="none")
    group.agent("default", model="default", system_prompt="Keep the original system message", allow_tools=("lookup",))
    group.capability(ConversationCapability(), scope="runtime")
    imported = ExecutionInputContext.from_messages((
        ModelRequest(parts=[UserPromptPart("imported question")]),
        ModelResponse(parts=[TextPart("imported answer")]),
    ))

    def storage():
        cache = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if cached else None
        return (RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache) if backend == "sqlite"
                else RuntimeStorage.filesystem(tmp_path, cache=cache))

    state = storage()
    async with Runtime.open("standalone-fork", storage=state, models=_Models(_UsageFunctionModel(model)), runtime_features=group) as runtime:
        with capture_run_messages() as actual:
            source = await runtime.agents.get().start("source", input_context=imported)
            assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        expected = encode_model_messages(tuple(actual))
        source_id = source.execution_id
        record = await state.execution.executions.get(source_id, tenant_id=runtime.tenant_id)
        assert record is not None and not record.durable_trace and not record.restart_checkpoints
        run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                              execution_id=source_id, agent_run_seq=record.agent_run_seq)
        archive = state.run_store.read_store(RuntimeDomain.RECOVERY)
        checkpoint = await archive.latest_checkpoint(agent_run_id=run_id)
        assert checkpoint is not None
        assert encode_model_messages(tuple(checkpoint.context_messages or checkpoint.messages)) == expected
        assert await archive.checkpoint_count(agent_run_id=run_id) == 1
        assert await archive.list_events(agent_run_id=run_id) == []
        assert await state.run_store.read_store(RuntimeDomain.EXECUTION).get_agent_run(agent_run_id=run_id) is None

    async with Runtime.open("standalone-fork", storage=storage(), models=_Models(_UsageFunctionModel(model)), runtime_features=group) as runtime:
        original = await runtime.executions.get(source_id)
        fork = await original.fork("fork")
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert encode_model_messages(tuple(calls[-1][:-1])) == expected
        retry = await fork.retry("retry")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert encode_model_messages(tuple(calls[-1][:-1])) == expected
        assert tool_calls == ["first", "second"]


@pytest.mark.asyncio
async def test_terminal_standalone_tool_context_joins_one_physical_write_transaction(tmp_path: Path) -> None:
    async def lookup(context: AgentContext[None]) -> str:
        return "tool result"

    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="call")])
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("one-fork-boundary")
    group.tool(lookup, effect_policy="none")
    group.agent("default", model="default", allow_tools=("lookup",))
    group.capability(ConversationCapability(), scope="runtime")
    state = RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open("one-fork-boundary", storage=state, models=_Models(_UsageFunctionModel(model)), runtime_features=group) as runtime:
        transactions = set()

        def capture(connection, _cursor, statement, _parameters, _context, _many):
            if statement.split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                transactions.add(connection.get_transaction())

        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            source = await runtime.agents.get().start("question", idempotency_key="question")
            assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
            assert len(transactions) == 1 and None not in transactions
            replay = await runtime.agents.get().start("question", idempotency_key="question")
            assert replay.execution_id == source.execution_id
            assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
            assert len(transactions) == 1
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        fork = await source.fork("next")
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED


async def _damage_fork_context(store, damage: str) -> None:
    async def damage_context(transaction):
        if damage == "absent":
            records = await transaction.scan_records()
            run = next(record for record in records if record.kind == "agent_run")
            await transaction.delete_records((run.key_digest,))
        else:
            facts = await transaction.scan_facts()
            fact = next(fact for fact in facts if fact.kind == "step_checkpoint")
            data = {"invalid": True}
            if damage != "corrupt":
                checkpoint = _decode_step(fact.data)
                values = {"agent_run_id": "wrong-run", "projection_digest": "0" * 64,
                          "transcript_message_count": 1}
                data = _encode_step(replace(checkpoint, **{damage: values[damage]}))
            owner = await transaction.get_record(fact.owner_key_digest)
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.delete_fact_streams(fact.owner_key_digest)
            await transaction.insert_fact(replace(fact, data=data))

    await store.mutate(damage_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("absent", "corrupt", "agent_run_id", "projection_digest", "transcript_message_count"))
async def test_pending_confirmation_rejects_missing_or_corrupt_fork_context(tmp_path: Path, damage: str) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        turn = _standalone(state)
        await _commit(state, turn)
        await _damage_fork_context(state.storage.recovery.checkpoints.state_store, damage)
        with pytest.raises(AIError) as raised:
            await state.commands.read_pending_turn(turn.reservation, expected_session=None)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", (False, True), ids=("no-cache", "cache"))
@pytest.mark.parametrize("damage", ("agent_run_id", "projection_digest", "transcript_message_count"))
async def test_reopened_standalone_fork_rejects_mismatched_checkpoint_binding(
    tmp_path: Path, cached: bool, damage: str,
) -> None:
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart("answer")])

    group = RuntimeFeatures("corrupt-fork")
    group.agent("default", model="default", allow_tools=())
    group.capability(ConversationCapability(), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if cached else None)
    async with Runtime.open("corrupt-fork", storage=storage, models=models, runtime_features=group) as runtime:
        source = await runtime.agents.get().start("source")
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        source_id = source.execution_id
        await _damage_fork_context(storage.recovery.checkpoints.state_store, damage)
    with pytest.raises(AIError) as raised:
        async with Runtime.open("corrupt-fork", storage=RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if cached else None), models=models, runtime_features=group) as runtime:
            source = await runtime.executions.get(source_id)
            fork = await source.fork("fork")
            await fork.wait(timeout_seconds=5)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", (False, True), ids=("ordinary", "pending"))
async def test_standalone_fork_keeps_actual_file_result_after_file_changes(tmp_path: Path, cached: bool) -> None:
    path = tmp_path / "source.txt"
    path.write_text("original file bytes", encoding="utf-8")
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "source.txt"}, tool_call_id="file-call")])
        return ModelResponse(parts=[TextPart("file read")])

    workspace = Workspace.load(tmp_path)
    group = workspace_features("file-fork", workspace=workspace)
    group.agent("default", model="default", allow_tools=("read_file",))
    group.capability(ConversationCapability(), scope="runtime")
    async with Runtime.open("file-fork", storage=RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if cached else None), models=_Models(_UsageFunctionModel(model)), runtime_features=group) as runtime:
        with capture_run_messages() as messages:
            source = await runtime.agents.get().start("read")
            assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        expected = encode_model_messages(tuple(messages))
        assert b"original file bytes" in expected
        path.write_text("changed file bytes", encoding="utf-8")
        fork = await source.fork("continue")
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert encode_model_messages(tuple(calls[-1][:-1])) == expected
        assert "changed file bytes" not in str(calls[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("running", "failed", "cancelled"))
async def test_unfinished_standalone_run_never_publishes_a_complete_fork_seed(tmp_path: Path, state: str) -> None:
    entered = asyncio.Event()

    async def model(messages, info):
        entered.set()
        if state == "failed":
            raise ValueError("provider failed")
        await asyncio.Event().wait()

    group = RuntimeFeatures("incomplete-fork")
    group.agent("default", model="default", allow_tools=())
    group.capability(ConversationCapability(), scope="runtime")
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    with pytest.raises(AIError) as raised:
        async with Runtime.open("incomplete-fork", storage=storage, models=_Models(_UsageFunctionModel(model)), runtime_features=group) as runtime:
            source = await runtime.agents.get().start("source")
            await asyncio.wait_for(entered.wait(), 5)
            if state == "failed":
                assert (await source.wait()).result.status is ExecutionStatus.FAILED
            elif state == "cancelled":
                await source.cancel()
                assert (await source.wait()).result.status is ExecutionStatus.CANCELLED
            try:
                fork = await source.fork("fork")
                await fork.wait(timeout_seconds=5)
            finally:
                run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                                      execution_id=source.execution_id, agent_run_seq=1)
                archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
                assert await archive.latest_checkpoint(agent_run_id=run_id) is None
                if state == "running":
                    await source.cancel()
    assert raised.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
