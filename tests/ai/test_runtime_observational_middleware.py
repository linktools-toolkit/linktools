#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observational middleware lifecycle regressions."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import MiddlewarePipeline, RunContext
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._capabilities import _ObservationalMiddlewareCapability
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if route_id not in {None, "default"} or dict(payload) != _TextModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


class _RecordingMiddleware:
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, RunContext]],
        *,
        mutating: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.mutating = mutating

    def _record(self, stage: str, context: RunContext) -> None:
        self.events.append((self.name, stage, context))

    async def before_run(self, context: RunContext) -> None:
        self._record("before_run", context)

    async def before_model(self, context: RunContext) -> None:
        self._record("before_model", context)

    async def after_model(self, context: RunContext) -> None:
        self._record("after_model", context)

    async def before_tool(self, context: RunContext) -> None:
        self._record("before_tool", context)

    async def after_tool(self, context: RunContext) -> None:
        self._record("after_tool", context)

    async def on_error(self, context: RunContext, error: BaseException) -> None:
        del error
        self._record("on_error", context)

    async def after_run(self, context: RunContext) -> None:
        self._record("after_run", context)


class _FailingObservationalMiddleware(_RecordingMiddleware):
    async def before_model(self, context: RunContext) -> None:
        self._record("before_model", context)
        raise RuntimeError("observational failure")


@pytest.fixture
def observed_context() -> RunContext:
    return RunContext("tenant", "principal", "execution", None, "run", "agent")


@pytest.mark.asyncio
async def test_middleware_pipeline_before_forward_after_reverse_order(
    observed_context: RunContext,
) -> None:
    events: list[tuple[str, str, RunContext]] = []
    pipeline = MiddlewarePipeline(
        (_RecordingMiddleware("a", events), _RecordingMiddleware("b", events))
    )

    await pipeline.before_run(observed_context)
    await pipeline.before_model(observed_context)
    await pipeline.after_model(observed_context)
    await pipeline.before_tool(observed_context)
    await pipeline.after_tool(observed_context)
    await pipeline.after_run(observed_context)

    assert [(name, stage) for name, stage, _ in events] == [
        ("a", "before_run"),
        ("b", "before_run"),
        ("a", "before_model"),
        ("b", "before_model"),
        ("b", "after_model"),
        ("a", "after_model"),
        ("a", "before_tool"),
        ("b", "before_tool"),
        ("b", "after_tool"),
        ("a", "after_tool"),
        ("b", "after_run"),
        ("a", "after_run"),
    ]


@pytest.mark.asyncio
async def test_middleware_on_error_runs_once_in_reverse_order(
    observed_context: RunContext,
) -> None:
    events: list[tuple[str, str, RunContext]] = []
    pipeline = MiddlewarePipeline(
        (_RecordingMiddleware("a", events), _RecordingMiddleware("b", events))
    )
    error = RuntimeError("boom")
    await pipeline.on_error(observed_context, error)
    assert [(name, stage) for name, stage, _ in events] == [
        ("b", "on_error"),
        ("a", "on_error"),
    ]


@pytest.mark.asyncio
async def test_observational_failure_is_ignored_and_pipeline_continues(
    observed_context: RunContext,
) -> None:
    events: list[tuple[str, str, RunContext]] = []
    pipeline = MiddlewarePipeline(
        (
            _FailingObservationalMiddleware("bad", events),
            _RecordingMiddleware("good", events),
        )
    )
    await pipeline.before_model(observed_context)
    assert [(name, stage) for name, stage, _ in events] == [
        ("bad", "before_model"),
        ("good", "before_model"),
    ]


@pytest.mark.asyncio
async def test_observational_capability_maps_public_hooks_exactly(
    observed_context: RunContext,
) -> None:
    events: list[tuple[str, str, RunContext]] = []
    capability = _ObservationalMiddlewareCapability(
        MiddlewarePipeline((_RecordingMiddleware("m", events),)),
        observed_context,
    )
    args = {"path": "file"}
    await capability.before_run(None)  # type: ignore[arg-type]
    request_context = object()
    assert await capability.before_model_request(None, request_context) is request_context  # type: ignore[arg-type]
    response = object()
    assert await capability.after_model_request(  # type: ignore[arg-type]
        None,
        request_context=request_context,
        response=response,
    ) is response
    assert await capability.before_tool_execute(  # type: ignore[arg-type]
        None,
        call=object(),
        tool_def=object(),
        args=args,
    ) is args
    result = object()
    assert await capability.after_tool_execute(  # type: ignore[arg-type]
        None,
        call=object(),
        tool_def=object(),
        args=args,
        result=result,
    ) is result
    run_result = object()
    assert await capability.after_run(None, result=run_result) is run_result  # type: ignore[arg-type]
    assert [stage for _, stage, _ in events] == [
        "before_run",
        "before_model",
        "after_model",
        "before_tool",
        "after_tool",
        "after_run",
    ]


@pytest.mark.asyncio
async def test_observational_capability_on_run_error_re_raises_original(
    observed_context: RunContext,
) -> None:
    events: list[tuple[str, str, RunContext]] = []
    capability = _ObservationalMiddlewareCapability(
        MiddlewarePipeline((_RecordingMiddleware("m", events),)),
        observed_context,
    )
    error = RuntimeError("boom")
    with pytest.raises(RuntimeError) as raised:
        await capability.on_run_error(None, error=error)  # type: ignore[arg-type]
    assert raised.value is error
    assert [(name, stage) for name, stage, _ in events] == [("m", "on_error")]


@pytest.mark.asyncio
async def test_runtime_open_rejects_mutating_middleware(tmp_path: Path) -> None:
    events: list[tuple[str, str, RunContext]] = []
    middleware = _RecordingMiddleware("mutating", events, mutating=True)
    with pytest.raises(AIError) as error:
        async with Runtime.open(
            Workspace.load(tmp_path),
            models=_TextModels(),  # type: ignore[arg-type]
            middleware=(middleware,),
        ):
            raise AssertionError("mutating middleware must fail before Runtime is yielded")
    assert error.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT
    assert events == []


@pytest.mark.asyncio
async def test_runtime_middleware_context_distinguishes_session_runs(tmp_path: Path) -> None:
    agent_path = tmp_path / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(AgentSpec("default", model="default", allow_tools=()))
    )
    events: list[tuple[str, str, RunContext]] = []
    middleware = _RecordingMiddleware("m", events)

    async with Runtime.open(
        Workspace.load(tmp_path),
        models=_TextModels(),  # type: ignore[arg-type]
        middleware=(middleware,),
    ) as runtime:
        first = await runtime.agent("default").run("hello", timeout_seconds=10)
        session = await runtime.agent("default").create_session("session")
        second = await runtime.agent("default").run(
            "hello",
            session_id=session.session_id,
            timeout_seconds=10,
        )

    contexts = [context for _, stage, context in events if stage == "before_run"]
    assert [context.execution_id for context in contexts] == [
        first.execution_id,
        second.execution_id,
    ]
    assert contexts[0].session_id is None
    assert contexts[1].session_id == session.session_id
    assert all(context.run_id for context in contexts)
