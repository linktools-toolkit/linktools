#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing checkpoint recovery can retry default idempotent tool calls."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("legacy_stage", ("fresh", "unstarted", "completed", "budget"))
def test_idempotent_tool_retries_original_call_after_worker_process_exit(tmp_path: Path, legacy_stage: str) -> None:
    script = textwrap.dedent('''
        import asyncio, json, os, sys
        from pathlib import Path
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from linktools.ai.capability import ExecutionCapability, AgentContext
        from pydantic_ai_harness.step_persistence import StepPersistence
        from tests.ai._runtime_test_helpers import native_checkpoint
        from linktools.ai.core import CommitPolicy, ExecutionStatus, ToolOperationStatus, RunBudget
        from linktools.ai.persistence import Persistence, PersistenceStorage
        from linktools.ai.runtime import Runtime, RuntimeFeatures
        from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

        root, mode = Path(sys.argv[1]), sys.argv[2]
        legacy_stage = sys.argv[3]
        legacy_strict = legacy_stage in {"unstarted", "completed"}
        entered = asyncio.Event()
        async def model(messages, info):
            returned = any(isinstance(part, ToolReturnPart) and part.tool_call_id == "original-call"
                for message in messages for part in message.parts)
            with (root / "model-calls").open("a") as stream:
                stream.write(mode + (":result" if returned else ":request") + "\\n")
            if returned:
                if mode == "create" and legacy_stage == "completed":
                    entered.set()
                    await asyncio.Event().wait()
                assert any(isinstance(part, ToolReturnPart) and part.content == "stored 7"
                    for message in messages for part in message.parts)
                return ModelResponse([TextPart("done")])
            if legacy_stage == "unstarted" and mode == "create":
                entered.set()
                await asyncio.Event().wait()
            assert mode == "create" or legacy_strict, "recovery must not regenerate the tool call"
            return ModelResponse([ToolCallPart("write", {"value": 7}, "original-call")])
        async def write(ctx: AgentContext[None], value: int) -> str:
            with (root / "tool-calls").open("a") as stream:
                stream.write(str(value) + "\\n")
            (root / "business-effect").write_text(str(value))
            if mode == "create" and not legacy_strict:
                entered.set()
                await asyncio.Event().wait()
            return "stored " + str(value)
        async def main():
            group = RuntimeFeatures("checkpoint-replacement")
            group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
            group.capability(native_checkpoint())
            group.tool(write, **({"effect_policy": "non_replay_safe"}
                if legacy_strict and mode == "create" else {}))
            group.agent("default", allow_tools=("write",))
            storage = PersistenceStorage.sqlite(root / "source.db")
            async with Persistence.open("checkpoint-replacement", storage=storage) as state, Runtime.attach(
                persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), group),
                auto_recover=False, metrics=None,
            ) as runtime:
                if mode == "create":
                    session = await runtime.agents.get().create_session("session")
                    handle = await session.start("original", idempotency_key="one",
                        budget=RunBudget(tool_calls=2) if legacy_stage == "budget" else None)
                    await asyncio.wait_for(entered.wait(), 5)
                    record = await state.execution.executions.get(handle.execution_id, tenant_id=runtime.tenant_id)
                    assert record is not None and record.restart_checkpoints
                    operations = await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                    if legacy_stage == "completed":
                        assert len(operations) == 1 and operations[0].replay_safe is False
                        assert operations[0].status is ToolOperationStatus.COMPLETED
                    elif legacy_stage == "unstarted":
                        assert not operations
                    else:
                        assert len(operations) == 1 and operations[0].replay_safe is True
                    (root / "execution").write_text(handle.execution_id)
                    print("CRASH_ORIGINAL_EXECUTION", handle.execution_id, flush=True)
                    os._exit(73)
                execution_id = (root / "execution").read_text()
                await runtime.executions.recover(execution_id, principal=runtime.default_principal,
                    idempotency_key="replace-worker")
                handle = await runtime.executions.get(execution_id)
                result = await handle.wait(timeout_seconds=10)
                assert result.result.status is ExecutionStatus.SUCCEEDED, result
                if legacy_stage == "budget":
                    assert (await handle.budget_usage()).tool_calls == 2
                operations = await state.checkpoint.tools.list_by_execution(execution_id, tenant_id=runtime.tenant_id)
                if legacy_strict:
                    assert len(operations) == 1 and operations[0].replay_safe is False
                else:
                    assert len(operations) == 1 and operations[0].replay_safe is True
                assert (root / "model-calls").read_text().splitlines() == {
                    "fresh": ["create:request", "resume:result"],
                    "budget": ["create:request", "resume:result"],
                    "unstarted": ["create:request", "resume:request", "resume:result"],
                    "completed": ["create:request", "create:result", "resume:result"],
                }[legacy_stage]
                assert (root / "tool-calls").read_text().splitlines() == (["7"] if legacy_strict else ["7", "7"])
                assert (root / "business-effect").read_text() == "7"
                print("SAME_EXECUTION_CHECKPOINT_RECEIPT_RECOVERY", execution_id, flush=True)
        asyncio.run(main())
    ''')
    env = dict(os.environ)
    created = subprocess.run([sys.executable, "-c", script, str(tmp_path), "create", legacy_stage],
        env=env, capture_output=True, text=True, timeout=20)
    print(created.stdout, created.stderr, flush=True)
    assert created.returncode == 73
    resumed = subprocess.run([sys.executable, "-c", script, str(tmp_path), "resume", legacy_stage],
        env=env, capture_output=True, text=True, timeout=20)
    print(resumed.stdout, resumed.stderr, flush=True)
    assert resumed.returncode == 0


from linktools.ai.capability import ToolCallFailed
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy
from tests.ai.test_tool_effect_semantics import _call
from tests.ai._runtime_test_helpers import tool_run_context
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("completed", "failed", "unknown"))
async def test_recovered_idempotent_call_preserves_existing_receipt(tmp_path: Path, outcome: str) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "receipts.db"))
    await state.initialize(namespace="receipt-compat", tenant_id="tenant")
    objects = InMemoryObjectStore()
    try:
        def bridge(run, owner, recovery=None):
            return RuntimeToolOperationBridge(
                state.checkpoint.tools, objects, namespace="receipt-compat", tenant_id="tenant",
                execution_id="execution", agent_run_id=run, binding_digest="b" * 64,
                owner=owner, background_tasks=set(), payload_policy=PayloadPolicy(),
                recovery_agent_run_id=recovery,
            )
        original = bridge("old-run", "old-worker")
        decision = await original.begin(
            tool_run_context(), ToolCallPart("effect", {}, "call"),
            ToolDefinition(name="effect"), {}, True,
        )
        assert decision is not None
        operation_id, fence = decision.operation_id, decision.fence
        if outcome == "completed":
            await original.complete(decision, {"value": "original"})
        elif outcome == "failed":
            await original.fail(decision, ToolCallFailed("original failure"))
        else:
            with pytest.raises(AIError) as raised:
                await original.unknown(decision, RuntimeError("ambiguous"))
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN

        async def effect():
            raise AssertionError("persisted receipt must not execute the leaf again")
        recovered = bridge("new-run", "new-worker", "old-run")
        descriptor = ManagedToolDescriptor("tool_operation", "replay_safe", "business")
        if outcome == "completed":
            result, _ = await _call(effect, descriptor, bridge=recovered)
            assert result == {"value": "original"}
        elif outcome == "failed":
            with pytest.raises(ToolCallFailed, match="original failure"):
                await _call(effect, descriptor, bridge=recovered)
        else:
            with pytest.raises(AIError) as raised:
                await _call(effect, descriptor, bridge=recovered)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        records = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
        assert len(records) == 1
        assert records[0].tool_operation_id == operation_id
        assert records[0].fence == fence
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_default_idempotent_dispatch_obeys_budget_without_entering_leaf() -> None:
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.core import RunBudget
    from linktools.ai.runtime._budget import RunBudgetContext
    from linktools.ai.runtime._tool_boundary import BoundaryToolset
    from tests.ai._runtime_test_helpers import tool_run_context, tool_with_metadata

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="idempotent-budget", tenant_id="tenant")
    try:
        await state.execution.budgets.ensure("scope", RunBudget(tool_calls=0))
        bridge = RuntimeToolOperationBridge(
            state.checkpoint.tools, InMemoryObjectStore(), namespace="idempotent-budget",
            tenant_id="tenant", execution_id="execution", agent_run_id="run",
            binding_digest="b" * 64, owner="worker", background_tasks=set(),
            payload_policy=PayloadPolicy(),
        )
        effects = []
        async def effect():
            effects.append("effect")
            return "done"
        descriptor = ManagedToolDescriptor("tool_operation", "replay_safe", "business")
        boundary = BoundaryToolset(
            (FunctionToolset([tool_with_metadata(effect, descriptor)]),),
            {"effect": descriptor}, id="idempotent-budget", tool_operations=bridge,
            budget=RunBudgetContext(state.execution.budgets, "scope", "execution", "run"),
        )
        context = tool_run_context()
        tools = await boundary.get_tools(context)
        with pytest.raises(AIError) as raised:
            await boundary.call_tool("effect", {}, context, tools["effect"])
        assert raised.value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
        assert effects == []
        assert (await state.execution.budgets.read("scope")).tool_calls == 0
        operation, = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
        assert operation.status.value == "PENDING" and operation.owner is None
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", (False, True))
async def test_parallel_idempotent_rich_results_commit_and_reopen(tmp_path: Path, terminal: bool) -> None:
    import asyncio
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart
    from linktools.ai.capability import AgentContext, ExecutionCapability
    from pydantic_ai_harness.step_persistence import StepPersistence
    from tests.ai._runtime_test_helpers import native_checkpoint
    from linktools.ai.core import CommitPolicy, ExecutionStatus, ToolOperationStatus
    from linktools.ai.model.messages import decode_model_messages
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.storage import InMemoryCacheBackend
    from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    entered_a, entered_b, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    results_ready, finish_model = asyncio.Event(), asyncio.Event()
    effects, requests = [], []

    def assert_results(operations):
        by_call = {operation.tool_call_id: operation for operation in operations}
        assert set(by_call) == {"first-call", "second-call"}
        for call, operation in by_call.items():
            assert operation.tool_name == call.removesuffix("-call")
            assert operation.replay_safe and operation.status is ToolOperationStatus.COMPLETED
            assert operation.owner is not None and operation.fence > 0
            assert operation.result_payload is not None
            stored = decode_model_messages(operation.result_payload.decode())
            assert len(stored) == 1 and isinstance(stored[0], ModelRequest)
            assert len(stored[0].parts) == 1
            returned = stored[0].parts[0]
            assert isinstance(returned, ToolReturnPart)
            assert returned.tool_call_id == operation.tool_operation_id
            if call == "first-call":
                assert isinstance(returned.content, ToolReturn)
                assert returned.content.return_value == {"saved": 7}
                assert returned.content.content == "independent evidence"
                assert returned.content.metadata == {"origin": "first"}
            else:
                assert returned.content == "second result"
        return by_call

    async def first(ctx: AgentContext[None]) -> ToolReturn:
        entered_a.set()
        await release.wait()
        effects.append("first")
        return ToolReturn(return_value={"saved": 7}, content="independent evidence",
                          metadata={"origin": "first"})
    async def second(ctx: AgentContext[None]) -> str:
        entered_b.set()
        await release.wait()
        effects.append("second")
        return "second result"
    async def model(messages, info):
        requests.append(messages)
        returns = {p.tool_call_id: p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)}
        if not returns:
            return ModelResponse([ToolCallPart("first", {}, "first-call"),
                                  ToolCallPart("second", {}, "second-call")])
        assert returns["first-call"].content == {"saved": 7}
        assert returns["first-call"].metadata == {"origin": "first"}
        assert returns["second-call"].content == "second result"
        assert "independent evidence" in str(messages)
        results_ready.set()
        await finish_model.wait()
        return ModelResponse([TextPart("complete")])

    group = RuntimeFeatures("idempotent-parallel")
    group.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()))
    if not terminal:
        group.capability(native_checkpoint())
    group.tool(first)
    group.tool(second)
    group.agent("default", allow_tools=("first", "second"))
    features = model_features(_Models(_UsageFunctionModel(model)), group)
    database = tmp_path / "parallel.db"
    async with Persistence.open("idempotent-parallel", storage=PersistenceStorage.sqlite(
        database, cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )) as state, Runtime.attach(
        persistence=state, features=features, auto_recover=False, metrics=None,
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        handle = await session.start("save", idempotency_key="one")

        async def read_source_execution():
            repository = state.execution.executions
            return await repository.state_store.read(lambda _transaction: repository.get(
                handle.execution_id, tenant_id=runtime.tenant_id))

        try:
            await asyncio.wait_for(asyncio.gather(entered_a.wait(), entered_b.wait()), 5)
            admitted = await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
            source = await read_source_execution()
            assert effects == []
            if terminal:
                assert admitted == () and source is None
            else:
                assert source is not None and source.status is ExecutionStatus.STARTED
                assert source.restart_checkpoints
                assert {operation.tool_call_id for operation in admitted} == {"first-call", "second-call"}
                assert len(admitted) == 2
                for operation in admitted:
                    assert operation.status is ToolOperationStatus.CLAIMED and operation.replay_safe
                    assert operation.owner == source.producer_claim_id
                    assert operation.producer_generation == source.producer_generation
                    assert operation.result_payload is None
        finally:
            release.set()
        try:
            await asyncio.wait_for(results_ready.wait(), 5)
            assert sorted(effects) == ["first", "second"]
            if terminal:
                assert await read_source_execution() is None
                assert await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id) == ()
        finally:
            finish_model.set()
        result = (await handle.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert sorted(effects) == ["first", "second"]
        assert len(requests) == 2
        completed = assert_results(await state.checkpoint.tools.list_by_execution(
            handle.execution_id, tenant_id=runtime.tenant_id))
        source = await read_source_execution()
        assert source is not None and source.status is ExecutionStatus.SUCCEEDED
        assert all(operation.owner == source.producer_claim_id
                   and operation.producer_generation == source.producer_generation
                   for operation in completed.values())
        for operation in admitted:
            retained = completed[operation.tool_call_id]
            assert (retained.tool_operation_id, retained.owner, retained.fence) == (
                operation.tool_operation_id, operation.owner, operation.fence)
        original_id = handle.execution_id
    async with Persistence.open("idempotent-parallel", storage=PersistenceStorage.sqlite(
        database, cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )) as state, Runtime.attach(
        persistence=state, features=features, auto_recover=False, metrics=None,
    ) as runtime:
        reopened = assert_results(await state.checkpoint.tools.list_by_execution(original_id, tenant_id=runtime.tenant_id))
        assert reopened == completed
        replay = await runtime.agents.get().session("session").start("save", idempotency_key="one")
        assert replay.execution_id == original_id
        assert (await replay.wait()).result.output == {"text": "complete"}
        assert sorted(effects) == ["first", "second"]
        assert len(requests) == 2
        assert assert_results(await state.checkpoint.tools.list_by_execution(
            original_id, tenant_id=runtime.tenant_id)) == completed


@pytest.mark.asyncio
async def test_mcp_defaults_safe_and_preserves_local_strict_policy() -> None:
    from pydantic_ai import Tool
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.capability import tool_metadata
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.runtime._mcp import _MCPModelToolset, _model_tool_name
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, managed_tool_descriptor_from_metadata
    from tests.ai._runtime_test_helpers import tool_run_context

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="mcp-policy", tenant_id="tenant")
    calls = []
    try:
        async def default_tool() -> str:
            calls.append("default")
            rows = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
            assert len(rows) == 1 and rows[0].replay_safe is True
            assert rows[0].status is ToolOperationStatus.CLAIMED
            return "default result"
        async def strict_tool() -> str:
            calls.append("strict")
            rows = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
            assert len(rows) == 2
            strict, = (row for row in rows if row.tool_name == _model_tool_name("same-server", "strict_tool"))
            assert strict.replay_safe is False and strict.status is ToolOperationStatus.CLAIMED
            return "strict result"
        source = FunctionToolset([
            Tool(default_tool, metadata={"meta": tool_metadata(effect_policy="non_replay_safe"),
                                         "annotations": {"idempotentHint": False}}),
            Tool(strict_tool, metadata=tool_metadata(effect_policy="non_replay_safe")),
        ])
        bridge = RuntimeToolOperationBridge(
            state.checkpoint.tools, InMemoryObjectStore(), namespace="mcp-policy", tenant_id="tenant",
            execution_id="execution", agent_run_id="run", binding_digest="b" * 64,
            owner="worker", background_tasks=set(), payload_policy=PayloadPolicy(),
        )
        boundary = BoundaryToolset(
            (_MCPModelToolset(source, "same-server", None),), {}, id="mcp-policy",
            descriptor=managed_tool_descriptor_from_metadata, tool_operations=bridge,
        )
        context = tool_run_context()
        tools = await boundary.get_tools(context)
        for name in ("default_tool", "strict_tool"):
            model_name = _model_tool_name("same-server", name)
            context.tool_call_id = name
            result = await boundary.call_tool(model_name, {}, context, tools[model_name])
            assert result == ("default result" if name == "default_tool" else "strict result")
        assert calls == ["default", "strict"]
        rows = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
        assert len(rows) == 2 and all(row.status is ToolOperationStatus.COMPLETED for row in rows)
    finally:
        await state.close()
