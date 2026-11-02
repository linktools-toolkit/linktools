#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observed direct responses retain their own durable interaction semantics."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, JsonValue, Principal
from linktools.ai.execution import ModelInteractionItem
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

_RESPONSE_FIELDS = frozenset(field.name for field in fields(ModelResponse))


class _Binding:
    provider = "offline"
    model_identity = "offline-decision"
    contract: dict[str, JsonValue] = {"provider": "offline", "model": "offline-decision"}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


class _InputBinding(_Binding):
    contract = {**_Binding.contract, "input_mime_types": ["text/plain"]}


class _DirectResponse(AbstractCapability):
    def __init__(
        self, *, entered: asyncio.Event | None = None, release: asyncio.Event | None = None,
        model_id: str | None = None,
    ) -> None:
        self.entered = entered
        self.release = release
        self.model_id = model_id

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        if request_context.model.model_name != "direct":
            return await handler(request_context)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return ModelResponse(
            [TextPart("direct answer")], model_name="direct",
            usage=RequestUsage(input_tokens=9, output_tokens=5),
            **({"model_id": self.model_id} if "model_id" in _RESPONSE_FIELDS else {}),
        )


def _features(
    provider_calls: list[str], *, direct: _DirectResponse | None = None,
    provider_entered: asyncio.Event | None = None,
    provider_release: asyncio.Event | None = None,
) -> RuntimeFeatures:
    features = RuntimeFeatures("direct-results")
    for name in ("direct", "provider"):
        def model(selected: str) -> FunctionModel:
            async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
                provider_calls.append(selected)
                assert selected == "provider", "a direct response must not dispatch its managed model"
                if provider_entered is not None:
                    provider_entered.set()
                if provider_release is not None:
                    await provider_release.wait()
                yield "provider answer"
            return FunctionModel(stream_function=stream, model_name=selected)

        features.agent(name, model=name, allow_tools=(), allow_skills=(), allow_subagents=())
        features.model(name, _InputBinding(model(name)))
    features.capability(direct or _DirectResponse(), id="direct-response")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability())
    return features


async def _assert_interaction(
    runtime: Runtime, execution_id: str, principal: Principal, *, direct: bool,
    model_id: str | None = None, with_attachment: bool = False,
) -> ModelInteractionItem:
    metadata = (await runtime.executions.model_interactions(
        execution_id, principal=principal,
    )).items
    assert len(metadata) == 1
    assert not hasattr(metadata[0], "is_direct_result")
    assert metadata[0].request == {} and metadata[0].response is None
    assert not metadata[0].content_included
    content = (await runtime.executions.model_interactions(
        execution_id, principal=principal, include_content=True, include_client_requests=True,
    )).items
    assert len(content) == 1
    item = content[0]
    assert not hasattr(item, "is_direct_result")
    assert item.status == "SUCCEEDED"
    assert item.response is not None
    assert item.response["parts"][0]["content"] == ("direct answer" if direct else "provider answer")
    if direct:
        if "model_id" in _RESPONSE_FIELDS:
            assert item.response.get("model_id") == model_id
        else:
            assert "model_id" not in item.response
        assert item.usage is not None
        assert (item.usage.input_tokens, item.usage.output_tokens) == (9, 5)
        assert item.request == {}
        assert item.client_requests in (None, ())
        assert item.stream_attempts in (None, ())
    else:
        assert item.request["messages"]
    execution = await runtime.executions.get(execution_id, principal=principal)
    events = (await execution.list_events(include_content=True, limit=100)).items
    finished = [event.payload for event in events if event.event_type == "MODEL_REQUEST_FINISHED"]
    assert len(finished) == 1
    assert all("is_direct_result" not in event.payload for event in events)
    trace = (await runtime.history.trace(execution_id, principal=principal)).items
    assert any(item.payload["kind"] == "MODEL_RESPONSE" for item in trace)
    assert all("is_direct_result" not in item.payload for item in trace)
    attachments = await runtime.history.attachment_facts(execution_id, principal=principal)
    if with_attachment:
        assert len(attachments.items) == 1
        fact = attachments.items[0]
        assert fact.fact == "accepted"
        assert fact.model_request_seq is None and fact.agent_run_seq is None
        assert fact.size == len(b"actual input bytes")
        assert fact.digest == hashlib.sha256(b"actual input bytes").hexdigest()
    else:
        assert attachments.items == ()
    return item


@pytest.mark.asyncio
@pytest.mark.parametrize(("direct", "model_id"), ((False, None), (True, None), (True, "returned:model")))
async def test_response_kind_survives_sqlite_reopen_without_invented_request(
    tmp_path: Path, direct: bool, model_id: str | None,
) -> None:
    calls: list[str] = []
    name = "direct" if direct else "provider"
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls, direct=_DirectResponse(model_id=model_id)),
    ) as runtime:
        session = await runtime.agents.get(name).create_session("session")
        execution = await session.start(
            ("hello", BinaryContent(b"actual input bytes", media_type="text/plain"))
            if direct else "hello",
        )
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        execution_id, principal = execution.execution_id, runtime.default_principal
        original = await _assert_interaction(runtime, execution_id, principal, direct=direct, model_id=model_id, with_attachment=direct)
        messages = (await runtime.history.messages("session", principal=principal)).items
        assert [part.content for message in messages if isinstance(message, ModelResponse)
                for part in message.parts if isinstance(part, TextPart)] == [f"{name} answer"]
    assert calls == ([] if direct else ["provider"])
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls),
    ) as runtime:
        restored = await _assert_interaction(runtime, execution_id, principal, direct=direct, model_id=model_id, with_attachment=direct)
        assert restored == original
        messages = (await runtime.history.messages("session", principal=principal)).items
        assert [part.content for message in messages if isinstance(message, ModelResponse)
                for part in message.parts if isinstance(part, TextPart)] == [f"{name} answer"]
    assert calls == ([] if direct else ["provider"])


@pytest.mark.asyncio
async def test_parallel_direct_and_provider_interactions_remain_distinct(tmp_path: Path) -> None:
    calls: list[str] = []
    direct_entered, provider_entered = asyncio.Event(), asyncio.Event()
    features = _features(
        calls, direct=_DirectResponse(entered=direct_entered, release=provider_entered),
        provider_entered=provider_entered, provider_release=direct_entered,
    )
    storage = PersistenceStorage.sqlite(tmp_path / "parallel.sqlite")
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features,
    ) as runtime:
        async def run(name: str) -> ModelInteractionItem:
            session = await runtime.agents.get(name).create_session(name)
            execution = await session.start("hello")
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            return await _assert_interaction(
                runtime, execution.execution_id, runtime.default_principal, direct=name == "direct",
            )
        first, second = await asyncio.gather(run("direct"), run("provider"))
        assert first.execution_id != second.execution_id
        assert first.request == {} and second.request["messages"]
    assert calls == ["provider"]


@pytest.mark.asyncio
async def test_cancelled_wrapper_without_response_does_not_claim_direct_result(tmp_path: Path) -> None:
    calls: list[str] = []
    entered, release = asyncio.Event(), asyncio.Event()
    features = _features(calls, direct=_DirectResponse(entered=entered, release=release))
    storage = PersistenceStorage.sqlite(tmp_path / "cancelled.sqlite")
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features,
    ) as runtime:
        session = await runtime.agents.get("direct").create_session("session")
        execution = await session.start("hello")
        await asyncio.wait_for(entered.wait(), timeout=5)
        await execution.cancel()
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.CANCELLED
        execution_id, principal = execution.execution_id, runtime.default_principal
        items = (await execution.model_interactions(include_content=True)).items
        assert len(items) == 1
        assert items[0].status == "CANCELLED"
        assert not hasattr(items[0], "is_direct_result")
        assert items[0].request == {} and items[0].response is None
    assert calls == []
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls),
    ) as runtime:
        restored = (await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True,
        )).items
        assert restored == items
    assert calls == []


@pytest.mark.asyncio
async def test_direct_tool_response_then_provider_result_keeps_both_interactions(tmp_path: Path) -> None:
    calls: list[str] = []

    class ToolResponse(_DirectResponse):
        async def wrap_model_request(
            self, ctx: RunContext[object], *, request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart)
                       for message in request_context.messages for part in message.parts):
                return ModelResponse([ToolCallPart("lookup", {}, tool_call_id="lookup-once")])
            return await handler(request_context)

    async def lookup(ctx: AgentContext[object]) -> str:
        calls.append("tool")
        return "tool result"

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append("provider")
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        assert len(returns) == 1 and returns[0].content == "tool result"
        yield "final answer"

    features = RuntimeFeatures("direct-tool")
    features.model("default", _InputBinding(FunctionModel(stream_function=stream, model_name="provider")))
    features.agent("default", allow_tools=("lookup",), allow_skills=(), allow_subagents=())
    features.tool(lookup, effect_policy="none")
    features.capability(ToolResponse(), id="direct-tool-response")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability())
    storage = PersistenceStorage.sqlite(tmp_path / "tool.sqlite")

    async def check(runtime: Runtime, execution_id: str, principal: Principal) -> None:
        first = await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True, limit=1,
        )
        assert len(first.items) == 1 and first.next_cursor is not None
        second = await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True, limit=1, cursor=first.next_cursor,
        )
        assert len(second.items) == 1 and second.next_cursor is None
        assert [first.items[0].model_request_seq, second.items[0].model_request_seq] == [1, 2]
        assert first.items[0].request == {} and second.items[0].request["messages"]
        execution = await runtime.executions.get(execution_id, principal=principal)
        events = (await execution.list_events(include_content=True, limit=100)).items
        assert all("is_direct_result" not in event.payload for event in events)
        messages = (await runtime.history.messages("session", principal=principal)).items
        parts = [part for message in messages for part in message.parts]
        assert sum(isinstance(part, ToolCallPart) for part in parts) == 1
        assert sum(isinstance(part, ToolReturnPart) for part in parts) == 1
        assert [part.content for part in parts if isinstance(part, TextPart)] == ["final answer"]

    async with Persistence.open("direct-tool", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features,
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("look up evidence")
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        execution_id, principal = execution.execution_id, runtime.default_principal
        await check(runtime, execution_id, principal)
    async with Persistence.open("direct-tool", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features,
    ) as runtime:
        await check(runtime, execution_id, principal)
    assert calls == ["tool", "provider"]


@pytest.mark.asyncio
async def test_equal_provider_text_in_distinct_turns_is_not_deduplicated(tmp_path: Path) -> None:
    calls: list[str] = []
    storage = PersistenceStorage.sqlite(tmp_path / "repeated.sqlite")
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls),
    ) as runtime:
        session = await runtime.agents.get("provider").create_session("session")
        identities = []
        for question in ("first", "second"):
            execution = await session.start(question)
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            item = await _assert_interaction(
                runtime, execution.execution_id, runtime.default_principal, direct=False,
            )
            identities.append(item.execution_id)
        assert len(set(identities)) == 2
        messages = (await runtime.history.messages("session", principal=runtime.default_principal)).items
        responses = [message for message in messages if isinstance(message, ModelResponse)]
        assert [[part.content for part in response.parts if isinstance(part, TextPart)]
                for response in responses] == [["provider answer"], ["provider answer"]]
    assert calls == ["provider", "provider"]


@pytest.mark.asyncio
async def test_cancellation_during_direct_synthetic_events_keeps_single_captured_response(tmp_path: Path) -> None:
    from pydantic_ai.messages import AgentStreamEvent, PartEndEvent

    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[str] = []

    class PauseDirectEnd(_DirectResponse):
        async def on_event(self, ctx: RunContext[object], *, event: AgentStreamEvent) -> None:
            if isinstance(event, PartEndEvent):
                entered.set()
                await release.wait()

    storage = PersistenceStorage.sqlite(tmp_path / "cancelled-after-response.sqlite")
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls, direct=PauseDirectEnd()),
    ) as runtime:
        session = await runtime.agents.get("direct").create_session("session")
        execution = await session.start("hello")
        await asyncio.wait_for(entered.wait(), timeout=5)
        execution_id, principal = execution.execution_id, runtime.default_principal
        captured_items = (await execution.model_interactions(include_content=True, include_client_requests=True)).items
        assert len(captured_items) == 1
        captured = captured_items[0]
        assert captured.status == "SUCCEEDED"
        assert captured.request == {} and captured.response is not None
        await execution.cancel()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
        after = await _assert_interaction(runtime, execution_id, principal, direct=True)
        assert after == captured
        transcript = await runtime.history.history(execution_id, principal=principal, include_content=True)
        assert [(item.item_kind, item.content, item.model_request_seq) for item in transcript.items] == [
            ("assistant", "direct answer", 1),
        ]
    async with Persistence.open("direct-results", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=_features(calls),
    ) as runtime:
        assert await _assert_interaction(runtime, execution_id, principal, direct=True) == captured
        restored = await runtime.history.history(execution_id, principal=principal, include_content=True)
        assert restored.items == transcript.items
    assert calls == []
