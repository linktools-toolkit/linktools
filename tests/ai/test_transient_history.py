#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient routes expose their live facts without retaining an archive."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute

from ._runtime_test_helpers import _UsageFunctionModel
from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel"))
async def test_transient_execution_reads_live_owner_and_releases_after_handoff(outcome: str) -> None:
    entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    release_second = asyncio.Event()

    async def echo(_ctx: AgentContext[None], value: str) -> str:
        return value

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            second_entered.set()
            await release_second.wait()
            if outcome == "failure":
                raise ValueError("provider failed")
            return ModelResponse(parts=[TextPart("finished")])
        entered.set()
        await release_first.wait()
        return ModelResponse(parts=[TextPart("calling echo"), ToolCallPart(
            "echo", {"value": "tool body"}, tool_call_id="echo-call",
        )])

    capabilities = CapabilityGroup("transient-history")
    capabilities.tool(echo, effect_policy="replay_safe")
    capabilities.agent("default", model="default", allow_tools=("echo",))
    storage = RuntimeStorage(RuntimeStoragePlan(execution=RuntimeStorageRoute.transient()))
    async with Runtime.open(
        "transient-history", models=_Models(_UsageFunctionModel(model)),
        storage=storage, capabilities=(capabilities,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("question")
        principal = runtime.default_principal
        subscription = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            subscription = await runtime.history.subscribe_model_interactions(
                execution.execution_id, principal=principal,
            )
            assert subscription is not None
            generation = subscription.generation
            boundary = await runtime.history.capture_model_interaction_cutoffs(
                execution.execution_id, principal=principal,
            )
            assert boundary.local_staging_available and not boundary.durable_history_available
            assert [value.model_request_seq for value in boundary.cutoffs] == [1]
            assert [value.model_request_seq for value in boundary.durable_cutoffs] == [0]
            first = await execution.history(include_content=True, limit=1)
            initial = await execution.history(include_content=True)
            assert [item.content for item in initial.items if item.item_kind == "user"] == ["question"]
            running = await execution.model_interactions(include_content=True)
            assert [item.status for item in running.items] == ["RUNNING"]
            assert running.items[0].request["messages"]
            assert running.items[0].response is None
            assert [item.text for item in (await execution.transcript(include_content=True)).items] == ["question"]
            assert len((await execution.trace()).items) == 1

            release_first.set()
            await asyncio.wait_for(second_entered.wait(), 5)
            assert subscription.generation > generation
            history = await execution.history(include_content=True)
            assert [(item.item_kind, item.content) for item in history.items
                    if item.item_kind not in {"instructions", "system"}] == [
                ("user", "question"), ("assistant", "calling echo"),
                ("tool_call", {"value": "tool body"}), ("tool_result", "tool body"),
            ]
            assert [item.text for item in (await execution.transcript(include_content=True)).items] == [
                "question", "calling echo",
            ]
            interactions = await execution.model_interactions(include_content=True)
            assert [item.status for item in interactions.items] == ["SUCCEEDED", "RUNNING"]
            assert interactions.items[0].response is not None
            assert interactions.items[1].request["messages"]
            metadata = await runtime.history.read_model_interaction_metadata(
                execution.execution_id, principal=principal, agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=2,
            )
            assert [item.status for item in metadata] == ["SUCCEEDED", "RUNNING"]
            assert all(not item.content_included and item.request == {} for item in metadata)
            assert len((await execution.trace()).items) == 5
            frozen = list(first.items)
            cursor = first.next_cursor
            while cursor is not None:
                page = await execution.history(include_content=True, cursor=cursor, limit=1)
                frozen.extend(page.items)
                cursor = page.next_cursor
            assert tuple(frozen) == initial.items
            if outcome == "cancel":
                await execution.cancel()
        finally:
            release_first.set()
            release_second.set()
            if subscription is not None:
                await subscription.close()
        result = await execution.wait(timeout_seconds=5)
        assert result.result.status is {
            "success": ExecutionStatus.SUCCEEDED,
            "failure": ExecutionStatus.FAILED,
            "cancel": ExecutionStatus.CANCELLED,
        }[outcome]
        boundary = await runtime.history.capture_model_interaction_cutoffs(
            execution.execution_id, principal=principal,
        )
        assert not boundary.local_staging_available and not boundary.durable_history_available
        assert boundary.cutoffs == ()
        if outcome == "success":
            with pytest.raises(AIError) as unavailable:
                await execution.history(include_content=True)
            assert unavailable.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
        else:
            assert (await execution.history(include_content=True)).items == ()
