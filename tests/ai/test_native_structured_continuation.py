#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A native turn can continue history that ends in a structured-output receipt."""

from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import SessionCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _persistence_capabilities, model_features


class _SessionPolicy(TenantAuthorizationPolicy):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", (False, True))
async def test_structured_turns_continue_after_reopen(tmp_path, checkpoints):
    requests = []
    storage = PersistenceStorage.sqlite(tmp_path / "continuation.db")
    features = _persistence_capabilities()
    features.capability(SessionCapability(checkpoints=checkpoints), scope="runtime")
    features.agent(
        "default",
        output_schema={"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
        allow_tools=(), allow_skills=(), allow_subagents=(),
    )

    def respond(messages, info):
        prompts = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart)]
        receipts = [part.tool_call_id for message in messages for part in message.parts
                    if isinstance(part, ToolReturnPart)]
        requests.append((prompts, receipts))
        index = len(requests)
        output_tool, = info.output_tools
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {"count": index}, tool_call_id=f"out-{index}")])

    model_features(_Models(_UsageFunctionModel(respond)), features)

    @asynccontextmanager
    async def open_runtime():
        async with (
            Persistence.open("structured-continuation", storage=storage) as state,
            Runtime.open(persistence=state, features=features, authorization=_SessionPolicy()) as runtime,
        ):
            yield runtime

    async def send(runtime, index):
        turn = await runtime.sessions.start(
            "session", f"question-{index}", agent="default",
            principal=runtime.default_principal, idempotency_key=f"send-{index}",
        )
        result = await turn.wait_final(timeout_seconds=5)
        assert result.view["status"] == "SUCCEEDED", result.view
        assert result.view["output"] == {"count": index}

    async with open_runtime() as runtime:
        await runtime.sessions.create_turn_session("session", principal=runtime.default_principal)
        await send(runtime, 1)
        await send(runtime, 2)
    async with open_runtime() as runtime:
        await send(runtime, 3)
        await send(runtime, 4)
        history = await runtime.history.turn_session("session", principal=runtime.default_principal)
        assert [turn["output"] for turn in history["timeline"]["items"]] == [
            {"count": index} for index in range(1, 5)
        ]

    assert requests == [
        ([f"question-{previous}" for previous in range(1, index + 1)],
         [f"out-{previous}" for previous in range(1, index)])
        for index in range(1, 5)
    ]


def test_joined_requests_do_not_make_changed_history_acceptable():
    from dataclasses import replace

    from pydantic_ai.messages import ModelRequest, TextPart, repair_messages

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime._regenerable_transcript import prepare_native_turn_transcript

    history = [
        ModelRequest(parts=[UserPromptPart("original-a")]),
        ModelRequest(parts=[UserPromptPart("original-b")]),
        ModelResponse(parts=[TextPart("previous reply")]),
    ]
    merged = repair_messages(history, repair_last_response=False)
    assert len(merged) < len(history)
    suffix = [ModelRequest(parts=[UserPromptPart("current")]), ModelResponse(parts=[TextPart("result")])]
    changed = replace(merged[0], parts=[UserPromptPart("original-a"), UserPromptPart("changed")])
    with pytest.raises(AIError) as failure:
        prepare_native_turn_transcript(
            "current", history, [changed, *merged[1:], *suffix], {"text": "result"},
            instructions=(), system_prompt="", merge_context_requests=True,
        )
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
