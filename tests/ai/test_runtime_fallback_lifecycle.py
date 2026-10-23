#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fallback dispatch respects cancellation, stream commitment and input ownership."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from types import TracebackType

import pytest
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.decision import DecisionHandOff, DecisionModelSettings, DecisionRequest, DecisionResponse
from pydantic_ai.models.function import FunctionModel, StreamFunctionDef
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, ReferencedInputsCapability
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.model import FallbackBinding, ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage

from .test_decision_model_binding import _Binding, _Decision, _OfflineDecisionModel
from .test_referenced_input_failure import _BODY, _REF, _SOURCE
from .test_runtime_fallback_binding import _features


class _LifecycleModel(FunctionModel):
    def __init__(self, stream_function: StreamFunctionDef, *, name: str, events: list[str]) -> None:
        super().__init__(stream_function=stream_function, model_name=name)
        self.events = events

    async def __aenter__(self) -> "_LifecycleModel":
        self.events.append(f"enter:{self.model_name}")
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.events.append(f"exit:{self.model_name}")


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ("primary", "secondary"))
@pytest.mark.parametrize("stop", ("cancel", "close"))
@pytest.mark.parametrize("boundary", ("entry", "iteration"))
async def test_stopping_active_candidate_closes_every_model_without_fallback(
    selected: str, stop: str, boundary: str,
) -> None:
    events: list[str] = []
    blocked = asyncio.Event()

    async def stream(name: str) -> AsyncIterator[str]:
        events.append(f"request:{name}")
        if name == "primary" and selected == "secondary":
            raise DecisionHandOff(name, "delegate", 0.5, "handoff")
        try:
            if boundary == "iteration":
                yield "started "
            blocked.set()
            await asyncio.Event().wait()
        finally:
            events.append(f"stream-closed:{name}")

    async def primary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        async for text in stream("primary"):
            yield text

    async def secondary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        async for text in stream("secondary"):
            yield text

    primary = _LifecycleModel(primary_stream, name="primary", events=events)
    secondary = _LifecycleModel(secondary_stream, name="secondary", events=events)
    features, _ = _features(primary, secondary)
    async with Runtime.open("fallback-stop", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("respond")
        await asyncio.wait_for(blocked.wait(), 5)
        if stop == "cancel":
            await execution.cancel()
            result = (await execution.wait(timeout_seconds=5)).result
            assert result.status is ExecutionStatus.CANCELLED
        else:
            await asyncio.wait_for(runtime.close(), 5)
    assert [event for event in events if event.startswith("request:")] == (
        ["request:primary"] if selected == "primary" else ["request:primary", "request:secondary"]
    )
    assert events.count(f"stream-closed:{selected}") == 1
    for name in ("primary", "secondary"):
        assert events.count(f"enter:{name}") == events.count(f"exit:{name}") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("iteration", "exit"))
async def test_handoff_after_first_stream_event_fails_without_replaying(boundary: str) -> None:
    requests: list[str] = []
    handoff = DecisionHandOff("primary", "delegate", 0.5, "too late to replay")

    async def primary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        requests.append("primary")
        yield "committed first text"
        if boundary == "iteration":
            raise handoff

    async def secondary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        requests.append("secondary")
        yield "must not replace committed text"

    class Primary(FunctionModel):
        @asynccontextmanager
        async def request_stream(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                yield response
            if boundary == "exit":
                raise handoff

    features, _ = _features(
        Primary(stream_function=primary_stream, model_name="primary"),
        FunctionModel(stream_function=secondary_stream, model_name="secondary"),
    )
    async with Runtime.open("fallback-stream", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("respond")
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_diagnostics.exception_type.endswith("DecisionHandOff")
        interactions = (await execution.model_interactions(include_content=True)).items
        assert len(interactions) == 1 and interactions[0].status == "FAILED"
    assert requests == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("provider", "deadline"))
async def test_non_handoff_failure_and_expired_provider_deadline_do_not_route(failure: str) -> None:
    class Primary(_OfflineDecisionModel):
        async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
            self.requests.append(request)
            if failure == "deadline":
                await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)
            raise ValueError("provider rejected request")

    primary, secondary = Primary(), _OfflineDecisionModel()
    features, _ = _features(primary, secondary)
    async with Runtime.open("fallback-failure", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("classify", output=_Decision)
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_diagnostics.exception_type.endswith("TimeoutError" if failure == "deadline" else "ValueError")
    assert len(primary.requests) == 1
    assert not secondary.requests


class _InputBinding(_Binding):
    def __init__(self, model: Model, *, allowed: bool) -> None:
        super().__init__(model)
        self.allowed = allowed

    @property
    def contract(self) -> Mapping[str, JsonValue]:
        return {"input_mime_types": ["application/pdf"] if self.allowed else []}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("unused-incompatible", "selected-incompatible", "both-supported", "primary-incompatible"))
async def test_references_validate_and_hydrate_only_the_selected_candidate(case: str) -> None:
    loads: list[ReferencedContent] = []
    requests: list[str] = []

    async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        loads.append(content)
        assert content.ref == _REF
        return _BODY

    async def stream(name: str, messages: list[ModelMessage]) -> AsyncIterator[str]:
        requests.append(name)
        content = [
            item for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content if isinstance(item, BinaryContent)
        ]
        assert len(content) == 1 and content[0].data == _BODY
        if name == "primary" and case in {"selected-incompatible", "both-supported"}:
            raise DecisionHandOff(name, "delegate", 0.5, "handoff")
        yield "inspected reference"

    async def primary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        async for text in stream("primary", messages):
            yield text

    async def secondary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        async for text in stream("secondary", messages):
            yield text

    features = RuntimeFeatures("fallback-references")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", FallbackBinding(
        _InputBinding(FunctionModel(stream_function=primary_stream, model_name="primary"), allowed=case != "primary-incompatible"),
        _InputBinding(FunctionModel(stream_function=secondary_stream, model_name="secondary"), allowed=case in {"both-supported", "primary-incompatible"}),
        fallback_on=DecisionHandOff,
    ))
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    content = ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE)
    async with Runtime.open("fallback-references", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start(["inspect", content])
        result = (await execution.wait(timeout_seconds=5)).result
        if case in {"selected-incompatible", "primary-incompatible"}:
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == "REQUEST_FIELD_INVALID"
        else:
            assert result.status is ExecutionStatus.SUCCEEDED
    expected = {"unused-incompatible": ["primary"], "selected-incompatible": ["primary"],
                "both-supported": ["primary", "secondary"], "primary-incompatible": []}[case]
    assert requests == expected
    assert len(loads) == len(expected)


@pytest.mark.asyncio
async def test_suspended_fallback_response_continues_on_its_selected_candidate() -> None:
    requests: list[str] = []

    async def primary_stream(messages: list[ModelMessage], info: object) -> AsyncIterator[str]:
        requests.append("primary")
        raise DecisionHandOff("primary", "delegate", 0.5, "handoff")
        yield "unreachable"

    class Response(CompletedStreamedResponse):
        def get(self) -> ModelResponse:
            return replace(super().get(), metadata=self.metadata)

    class Secondary(FunctionModel):
        def __init__(self) -> None:
            super().__init__(stream_function=primary_stream, model_name="secondary")
            self.calls = 0

        @asynccontextmanager
        async def request_stream(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            requests.append("secondary")
            self.calls += 1
            if self.calls == 2:
                assert isinstance(messages[-1], ModelResponse)
                assert messages[-1].state == "suspended"
            response = ModelResponse(
                [TextPart("working " if self.calls == 1 else "done")], model_name=self.model_name,
                provider_name="function", provider_response_id=f"segment-{self.calls}",
                state="suspended" if self.calls == 1 else "complete",
            )
            yield Response(response, model_request_parameters=model_request_parameters, replay_events=True)

    features, _ = _features(FunctionModel(stream_function=primary_stream, model_name="primary"), Secondary())
    async with Runtime.open("fallback-continuation", features=features, storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("respond")
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "working done"}
        assert result.usage.model_requests == 1
    assert requests == ["primary", "secondary", "secondary"]
