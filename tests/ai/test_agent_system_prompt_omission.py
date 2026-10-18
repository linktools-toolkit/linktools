#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Absent system prompts never become empty provider or history messages."""

from linktools.ai.capability import TraceCapability
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.runtime import ExecutionInputContext, Runtime, RuntimeStorage
from linktools.ai.spec import AgentSpec

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("case,expected", (
    ("omitted", []), ("empty", []), ("text", ["Keep this system prompt"]),
    ("whitespace", [" \n\t "]), ("derive-none", ["Keep this system prompt"]),
    ("derive-empty", []),
))
async def test_system_prompt_presence_matches_provider_observation_and_history(
    tmp_path: Path, case: str, expected: list[str],
) -> None:
    sent_system: list[list[str]] = []
    sent_instructions: list[str | None] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent_system.append([part.content for message in messages if isinstance(message, ModelRequest)
                            for part in message.parts if isinstance(part, SystemPromptPart)])
        sent_instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("system-prompts")
    options = {"instructions": ("Keep independent instructions",), "allow_tools": (),
               "allow_skills": (), "allow_subagents": ()}
    if case == "empty":
        options["system_prompt"] = ""
    elif case == "whitespace":
        options["system_prompt"] = " \n\t "
    elif case in {"text", "derive-none", "derive-empty"}:
        options["system_prompt"] = "Keep this system prompt"
    group.agent("default", **options)
    group.capability(TraceCapability(), scope="runtime")
    async with Runtime.open("system-prompts", models=_Models(_UsageFunctionModel(model)),
                            storage=RuntimeStorage.filesystem(tmp_path), features=group) as runtime:
        agent = runtime.agents.get()
        if case == "derive-none":
            agent = agent.derive(system_prompt=None)
        elif case == "derive-empty":
            agent = agent.derive(system_prompt="")
        execution = await agent.start("question")
        result = await execution.wait(timeout_seconds=10)
        assert result.result.status == "SUCCEEDED"
        assert sent_system == [expected]
        assert all("Keep independent instructions" in text for text in sent_instructions)
        interactions = await execution.model_interactions(include_content=True)
        request = interactions.items[0].request
        observed = [part["content"] for message in request["messages"] for part in message["parts"]
                    if part["part_kind"] == "system-prompt"]
        assert observed == expected
        history = await execution.history(include_content=True)
        assert [item.content for item in history.items if item.item_kind == "system"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", ("", "Historical system prompt", " \n "))
@pytest.mark.parametrize("replace_system", (False, True))
async def test_imported_system_history_respects_explicit_replacement(
    tmp_path: Path, existing: str, replace_system: bool,
) -> None:
    sent_system: list[list[str]] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent_system.append([part.content for message in messages if isinstance(message, ModelRequest)
                            for part in message.parts if isinstance(part, SystemPromptPart)])
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("imported-system")
    group.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    imported = ExecutionInputContext.from_messages((
        ModelRequest(parts=[SystemPromptPart(existing), UserPromptPart("earlier")]),
        ModelResponse(parts=[TextPart("previous")]),
    ), replace_history_system_prompt=replace_system)
    group.capability(TraceCapability(), scope="runtime")
    async with Runtime.open("imported-system", models=_Models(_UsageFunctionModel(model)),
                            storage=RuntimeStorage.filesystem(tmp_path), features=group) as runtime:
        execution = await runtime.agents.get().start("question", input_context=imported)
        result = await execution.wait(timeout_seconds=10)
        assert result.result.status == "SUCCEEDED"
        expected = [] if replace_system else [existing]
        assert sent_system == [expected]
        interactions = await execution.model_interactions(include_content=True)
        recorded = interactions.items[0].request["messages"]
        assert [part["content"] for message in recorded for part in message["parts"]
                if part["part_kind"] == "system-prompt"] == expected


@pytest.mark.parametrize("value", (None, 0, [], {"type": "text", "text": "system"}))
def test_agent_spec_keeps_non_string_system_prompt_rejection(value: object) -> None:
    with pytest.raises(TypeError, match="system_prompt must be a string"):
        AgentSpec("default", system_prompt=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("case,system_prompt", (
    ("omitted", None), ("empty", ""), ("text", "Keep system content"), ("whitespace", " \n\t "),
))
async def test_openai_http_capture_preserves_only_configured_system_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, system_prompt: str | None,
) -> None:
    import json

    import httpx2

    from .test_model_client_request_capture import _registry, _response

    dispatched: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        dispatched.append(request.content)
        return _response(request)

    registry, _clients = _registry(monkeypatch, handler)
    group = RuntimeFeatures("http-system")
    options = {"instructions": ("Preserve independent HTTP instructions",), "allow_tools": (),
               "allow_skills": (), "allow_subagents": ()}
    if case != "omitted":
        options["system_prompt"] = system_prompt
    group.agent("default", **options)
    group.capability(TraceCapability(), scope="runtime")
    async with Runtime.open("http-system", models=registry, storage=RuntimeStorage.filesystem(tmp_path),
                            features=group) as runtime:
        execution = await runtime.agents.get().start("question")
        result = await execution.wait(timeout_seconds=10)
        assert result.result.status == "SUCCEEDED"
        interactions = await execution.model_interactions(include_content=True, include_client_requests=True)
        interaction, = interactions.items
        assert [attempt.body for attempt in interaction.client_requests] == dispatched
        assert len(dispatched) == 1
        messages = json.loads(dispatched[0])["messages"]
        system = [message["content"] for message in messages if message["role"] == "system"]
        assert "" not in system
        assert any("Preserve independent HTTP instructions" in content for content in system)
        expected = [] if system_prompt in (None, "") else [system_prompt]
        observed = [part["content"] for message in interaction.request["messages"] for part in message["parts"]
                    if part["part_kind"] == "system-prompt"]
        assert observed == expected
        for content in expected:
            assert content in system
        history = await execution.history(include_content=True)
        assert [item.content for item in history.items if item.item_kind == "system"] == expected
