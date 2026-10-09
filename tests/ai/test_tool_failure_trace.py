#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool execution errors remain observable without inventing SDK returns."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, CapabilityGroup, ToolCallFailed, ToolCallRetry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage

from ._runtime_test_helpers import _UsageFunctionModel
from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("fatal", "retry", "failed"))
async def test_tool_failure_trace_is_durable_and_does_not_fabricate_a_return(failure: str) -> None:
    requests = 0

    async def broken(_context: AgentContext[None]) -> str:
        if failure == "fatal":
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        if failure == "retry":
            raise ToolCallRetry("Correct the request")
        raise ToolCallFailed("The request failed")

    async def model(messages, info) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("broken", {}, tool_call_id="broken-call")])
        return ModelResponse(parts=[TextPart("Recovered")])

    group = CapabilityGroup("tool-failure-trace")
    group.tool(broken, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("broken",))
    async with Runtime.open(
        "tool-failure-trace", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("Call the tool")
        if failure == "fatal":
            with pytest.raises(AIError) as caught:
                await execution.wait(timeout_seconds=10)
            assert caught.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        else:
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.output == {"text": "Recovered"}
        trace = (await execution.trace(tool_call_id="broken-call")).items
        errors = [item for item in trace if item.payload["kind"] == "TOOL_ERROR"]
        assert len(errors) == 1
        assert errors[0].payload["status"] == "FAILED"
        assert errors[0].payload["duration_ns"] >= 0
        assert errors[0].payload["model_request_seq"] == 1
        history = (await execution.history(tool_call_id="broken-call", include_content=True)).items
        if failure == "fatal":
            assert [item.item_kind for item in history] == ["tool_call"]
            assert requests == 1
        else:
            assert len(history) == 2
            assert history[0].item_kind == "tool_call"
            assert history[1].content is not None
            assert requests == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel"))
async def test_terminal_seal_and_retention_release_outlast_pending_observation_flush(outcome: str) -> None:
    import asyncio

    from linktools.ai.core import ExecutionStatus

    entered = asyncio.Event()
    release = asyncio.Event()

    async def model(messages, info) -> ModelResponse:
        del messages, info
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise ValueError("provider failed")
        return ModelResponse(parts=[TextPart("finished")])

    group = CapabilityGroup("observation-terminal-drain")
    group.agent("default", model="default", allow_tools=())
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "observation-terminal-drain", models=_Models(_UsageFunctionModel(model)),
        storage=storage, capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("finish")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if outcome == "cancel":
                await execution.cancel()
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is {
            "success": ExecutionStatus.SUCCEEDED,
            "failure": ExecutionStatus.FAILED,
            "cancel": ExecutionStatus.CANCELLED,
        }[outcome]
        head = await storage.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        history = await execution.history(include_content=True)
        trace = await execution.trace()
        interactions = await execution.model_interactions(include_content=True)
        assert len(interactions.items) == 1
        model_status = interactions.items[0].status
        assert model_status == result.status.value
        assert [(item.payload["kind"], item.payload["status"]) for item in trace.items] == [
            ("MODEL_REQUEST", "STARTED"), ("MODEL_RESPONSE", model_status),
        ]
        assert [item.content for item in history.items if item.item_kind == "user"] == ["finish"]
        # The deferred observation deadline must not resurrect released staging
        # or mutate a sealed execution after its terminal result was handed off.
        await asyncio.sleep(1.1)
        assert await storage.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.default_principal.tenant_id,
        ) == head
        assert (await execution.history(include_content=True)).items == history.items
        assert (await execution.trace()).items == trace.items


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", ("deferred", "recovery_required"))
async def test_pause_handoff_drains_only_its_execution_while_another_run_is_live(tmp_path, handoff: str) -> None:
    import asyncio

    from pydantic_ai.messages import UserPromptPart

    from linktools.ai.core import ExecutionStatus
    from linktools.ai.workspace import Workspace, WorkspacePolicy, ToolPermissionPolicy, ToolPermissionRule

    entered = asyncio.Event()
    release = asyncio.Event()

    async def broken(_context: AgentContext[None]) -> str:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def model(messages, info) -> ModelResponse:
        del info
        if any(isinstance(part, UserPromptPart) and part.content == "hold this run"
               for message in messages for part in message.parts):
            entered.set()
            await release.wait()
            return ModelResponse(parts=[TextPart("held run finished")])
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "input.txt"}, tool_call_id="read-call")
            if handoff == "deferred" else ToolCallPart("broken", {}, tool_call_id="broken-call"),
        ])

    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(
        (ToolPermissionRule("ask", tool_name="read_file"),), default="allow",
    )))
    group = CapabilityGroup("parallel-handoff")
    group.tool(broken, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("read_file", "broken"))
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "parallel-handoff", models=_Models(_UsageFunctionModel(model)), storage=storage,
        capabilities=(CapabilityGroup("workspace", workspace=workspace), group),
    ) as runtime:
        held = await runtime.agents.get("default").start("hold this run")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            paused = await runtime.agents.get("default").start("pause this run")
            expected = ExecutionStatus.WAITING_DEFERRED if handoff == "deferred" else ExecutionStatus.RECOVERY_REQUIRED
            async def wait_paused():
                while True:
                    record = await storage.execution.executions.get(paused.execution_id, tenant_id=runtime.tenant_id)
                    if record is not None and record.status is expected:
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_paused(), 10)
            paused_history = await paused.history(include_content=True)
            assert [item.tool_call_id for item in paused_history.items if item.item_kind == "tool_call"] == [
                "read-call" if handoff == "deferred" else "broken-call",
            ]
            assert [item.content for item in paused_history.items if item.item_kind == "user"] == ["pause this run"]
            await asyncio.sleep(1.1)
            assert (await paused.history(include_content=True)).items == paused_history.items
            held_record = await storage.execution.executions.get(held.execution_id, tenant_id=runtime.tenant_id)
            assert held_record is not None and held_record.status is ExecutionStatus.STARTED
            held_history = await held.history(include_content=True)
            assert [item.content for item in held_history.items if item.item_kind == "user"] == ["hold this run"]
            assert not any(item.item_kind == "tool_call" for item in held_history.items)
        finally:
            release.set()
            assert (await held.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
