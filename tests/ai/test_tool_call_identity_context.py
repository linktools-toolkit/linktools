#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invocation identity follows native calls without mutating admitted dependencies."""

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import AgentContext, ToolCallIdentity, TraceCapability
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe.journal import ModelRequestJournal
from linktools.ai.persistence.runs import AgentRunRecord, StagingAgentRunStore
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder


def _deps() -> AgentContext[None]:
    return AgentContext(
        app=None, principal=Principal("actor", "tenant"), namespace="identity",
        session_id="session", execution_id="child", session_metadata={"label": "kept"},
        root_execution_id="root", parent_execution_id="parent", agent_id="worker", agent_run_seq=3,
    )


def _native(deps: AgentContext[None]) -> RunContext[AgentContext[None]]:
    return RunContext(
        deps=deps, model=TestModel(), usage=RunUsage(), run_id="run",
        tool_call_id="call", tool_name="work", run_step=99, retry=2,
    )


def _two_calls(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart("done")])
    return ModelResponse(parts=[
        ToolCallPart("work", {"value": value}, tool_call_id=f"call-{value}")
        for value in (1, 2)
    ])


def test_identity_without_trace_keeps_admitted_fields_and_no_invented_sequence() -> None:
    deps = _deps()
    ctx = _native(deps)
    identity = ToolCallIdentity.from_context(ctx)
    assert (identity.namespace, identity.principal, identity.session_id) == (
        deps.namespace, deps.principal, deps.session_id,
    )
    assert (identity.execution_id, identity.root_execution_id, identity.parent_execution_id) == (
        "child", "root", "parent",
    )
    assert (identity.agent_id, identity.agent_run_id, identity.agent_run_seq) == ("worker", "run", 3)
    assert (identity.tool_call_id, identity.tool_name, identity.retry) == ("call", "work", 2)
    assert identity.model_request_seq is None
    with pytest.raises(AIError) as failure:
        TraceCapability.for_context(ctx)
    assert failure.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
    with pytest.raises(FrozenInstanceError):
        identity.tool_call_id = "changed"
    assert deps.tool_call is None and deps.trace is None


@pytest.mark.parametrize("field", ["run_id", "tool_call_id", "tool_name"])
def test_identity_requires_native_call_boundary(field: str) -> None:
    with pytest.raises(AIError) as failure:
        ToolCallIdentity.from_context(replace(_native(_deps()), **{field: None}))
    assert failure.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_parallel_native_calls_keep_their_own_identity() -> None:
    deps = _deps()
    entered = set()
    both_entered = asyncio.Event()
    identities = {}

    async def work(ctx: RunContext[AgentContext[None]], value: int) -> str:
        before = ToolCallIdentity.from_context(ctx)
        entered.add(value)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        after = ToolCallIdentity.from_context(ctx)
        assert before == after
        assert ctx.deps is deps
        identities[value] = after
        return str(value)

    agent = Agent(FunctionModel(_two_calls), tools=[work], deps_type=AgentContext)
    result = await agent.run("run both", deps=deps, run_id="parallel-run")
    assert result.output == "done"
    assert {value: identity.tool_call_id for value, identity in identities.items()} == {
        1: "call-1", 2: "call-2",
    }
    assert {identity.agent_run_id for identity in identities.values()} == {"parallel-run"}
    assert deps.tool_call is None and deps.trace is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ordinary_tool_adapter_copies_dependencies_per_call(asynchronous: bool) -> None:
    deps = _deps()
    received = []
    trace = TraceCapability().bind(AbstractCapability())

    def capture(ctx: AgentContext[None], value: int) -> str:
        assert ctx is not deps
        assert ctx.trace is trace
        assert ctx.tool_call is not None
        assert ctx.tool_call.tool_call_id == f"call-{value}"
        assert ctx.tool_call.agent_run_id == "adapter-run"
        assert ctx.session_metadata == deps.session_metadata
        assert ctx.tool_call.model_request_seq is None
        with pytest.raises(FrozenInstanceError):
            ctx.tool_call = None
        received.append(ctx)
        return str(value)

    async def capture_async(ctx: AgentContext[None], value: int) -> str:
        await asyncio.sleep(0)
        return capture(ctx, value)

    tool = RuntimeFeatures().tool(capture_async if asynchronous else capture, name="work")
    agent = Agent(FunctionModel(_two_calls), tools=[tool], capabilities=[trace], deps_type=AgentContext)
    result = await agent.run("run both", deps=deps, run_id="adapter-run")
    assert result.output == "done"
    assert len(received) == 2 and received[0] is not received[1]
    assert deps.tool_call is None and deps.trace is None


@pytest.mark.asyncio
async def test_restored_deferred_call_keeps_originating_request_sequence() -> None:
    # The recorder's persisted event/transcript pair, not a second test map,
    # supplies provenance after restoring the outstanding call.
    store = StagingAgentRunStore()
    await store.initialize()
    try:
        run = AgentRunRecord("run")
        recorder = AgentRunRecorder(store, execution_id="child", agent_run_id="run")
        await recorder.register_agent_run(run)
        journal = ModelRequestJournal(
            source_namespace="identity", tenant_id="tenant", execution_id="child",
            agent_run_id="run", next_model_request_seq=7,
        )
        fact = journal.begin(2)
        call = ToolCallPart("work", {"value": 1}, tool_call_id="deferred")
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"), phase="completed",
            response=ModelResponse(parts=[call]), include_observation=False,
        )
        restored = AgentRunRecorder(store, execution_id="child", agent_run_id="run")
        await restored.register_agent_run(run)
        trace = TraceCapability().bind(
            AbstractCapability(), model_request_seq=restored.model_request_seq_for_tool_call,
        )
        ctx = replace(
            _native(_deps()), tool_call_id="deferred", tool_call_approved=True,
            tool_call_metadata={"model_request_seq": 1000}, capabilities={trace.id: trace},
        )
        assert TraceCapability.for_context(ctx) is trace
        assert ToolCallIdentity.from_context(ctx).model_request_seq == 7
        assert ToolCallIdentity.from_context(replace(ctx, run_step=100, retry=3)).model_request_seq == 7
        assert ToolCallIdentity.from_context(replace(ctx, tool_call_id="uncaptured")).model_request_seq is None
    finally:
        await store.close()
