#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable host example proves logical RPM/TPM accounting without a service."""

from __future__ import annotations

from linktools.ai.persistence import Persistence

from linktools.ai.core import ExecutionContext
from ._runtime_test_helpers import model_features

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai.settings import ModelSettings

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel

pytestmark = pytest.mark.asyncio


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "linktools-ai/examples/runtime_rate_limits.py"
    spec = importlib.util.spec_from_file_location("runtime_rate_limits_example", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return module, clock


def _advance(limiter, clock, seconds):
    clock.now += seconds
    changed = limiter._changed
    limiter._changed = asyncio.Event()
    changed.set()


async def _request(limiter, *, tokens=10, handler=None):
    model = TestModel()
    ctx = RunContext(deps=None, model=model, usage=RunUsage())
    request = ModelRequestContext(
        model=model, messages=[], model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    async def respond(_request):
        return ModelResponse(parts=[TextPart("done")], usage=RequestUsage(input_tokens=tokens))

    return await limiter.wrap_model_request(ctx, request_context=request, handler=handler or respond)


@pytest.mark.parametrize("quota", ("rpm", "tpm"))
async def test_request_and_token_windows_queue_atomically_and_expire(example, quota):
    module, clock = example
    limiter = module.GlobalRateLimit(
        quota_key="project", rpm=1 if quota == "rpm" else 10,
        tpm=100 if quota == "rpm" else 40, token_reservation=40,
    )
    await _request(limiter)
    waiting = asyncio.create_task(_request(limiter))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert len(limiter._admissions) == 1
    assert sum(charge.tokens for charge in limiter._charges) == 40
    _advance(limiter, clock, 60)
    assert (await asyncio.wait_for(waiting, 1)).parts[0].content == "done"
    assert len(limiter._admissions) == 1


async def test_token_debt_and_unknown_stream_charges_survive_the_original_window(example):
    module, clock = example
    limiter = module.GlobalRateLimit(quota_key="project", rpm=10, tpm=100, token_reservation=40)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def long_request(_request):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("done")], usage=RequestUsage(input_tokens=110))

    active = asyncio.create_task(_request(limiter, handler=long_request))
    await entered.wait()
    _advance(limiter, clock, 120)
    # One reservation remains active even though its RPM admission expired.
    assert sum(charge.tokens for charge in limiter._charges) == 40
    finish.set()
    await active
    assert sum(charge.tokens for charge in limiter._charges) == 110
    waiting = asyncio.create_task(_request(limiter))
    await asyncio.sleep(0)
    assert not waiting.done()
    _advance(limiter, clock, 60)
    await asyncio.wait_for(waiting, 1)


@pytest.mark.parametrize("outcome", ("cancel", "error"))
async def test_dispatched_unknown_usage_keeps_reservation_but_queued_cancel_charges_nothing(example, outcome):
    module, clock = example
    limiter = module.GlobalRateLimit(quota_key="project", rpm=10, tpm=40, token_reservation=40)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def blocked(_request):
        entered.set()
        await finish.wait()
        raise RuntimeError("provider failed")

    active = asyncio.create_task(_request(limiter, handler=blocked))
    await entered.wait()
    _advance(limiter, clock, 120)
    queued = asyncio.create_task(_request(limiter))
    await asyncio.sleep(0)
    assert not queued.done()
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert len(limiter._charges) == 1 and limiter._charges[0].tokens == 40
    if outcome == "cancel":
        active.cancel()
    else:
        finish.set()
    with pytest.raises(asyncio.CancelledError if outcome == "cancel" else RuntimeError):
        await active
    waiting = asyncio.create_task(_request(limiter))
    await asyncio.sleep(0)
    assert not waiting.done()
    _advance(limiter, clock, 60)
    await asyncio.wait_for(waiting, 1)


@pytest.mark.parametrize("streaming", (False, True))
async def test_runtime_token_estimator_reads_native_metadata_and_observes_response_usage(example, streaming):
    module, _clock = example
    requests = []
    completed = []

    def reserve(ctx, request):
        requests.append((ctx.deps, request))
        assert request.model.system == "function"
        assert request.model.model_name
        assert request.model_id is None
        assert request.model_settings["max_tokens"] == 12
        assert any("Be brief" in part.content for part in request.model_request_parameters.instruction_parts)
        assert any(isinstance(part, UserPromptPart) for message in request.messages for part in message.parts)
        assert any(tool.name == "ping" for tool in request.model_request_parameters.function_tools)
        return 1

    class ObservedRateLimit(module.GlobalRateLimit):
        def get_model_settings(self) -> ModelSettings:
            return {"max_tokens": 12}

        def _finish(self, charge, observed_tokens):
            completed.append(observed_tokens)
            super()._finish(charge, observed_tokens)

    limiter = ObservedRateLimit(quota_key="project-a", rpm=10, tpm=100, token_reservation=reserve)

    async def answer(_messages, _info):
        return ModelResponse(parts=[TextPart("done")], usage=RequestUsage(input_tokens=7, output_tokens=3))

    async def chunks(_messages, _info):
        yield "complete"
        yield " response"

    async def ping(_context: AgentContext[object]) -> str:
        return "pong"

    group = RuntimeFeatures("application", filter=lambda _agent, item: item.kind != "capability")
    group.tool(ping, effect_policy="none")
    group.agent("default", instructions="Be brief", allow_tools=("ping",), allow_capabilities=())
    group.capability(limiter, id="host.rate-limit", scope="runtime")
    model = FunctionModel(stream_function=chunks) if streaming else _UsageFunctionModel(answer)
    group.model("default", _Models(model))
    app = {"quota": "project-a"}
    async with Persistence.open('quota-metadata', storage=PersistenceStorage.in_memory()) as persistence_189, Runtime.attach(persistence=persistence_189, context=ExecutionContext(app), features=await group.capture()) as runtime:
        execution = await runtime.agents.get().start("question", correlation={"request": "one"})
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        deps, request = requests[0]
        assert deps.app is app and deps.namespace == "quota-metadata"
        assert deps.execution_id == execution.execution_id and deps.correlation == {"request": "one"}
        assert len(completed) == 1 and completed[0] > 0
        assert limiter._charges[0].tokens == completed[0]
        if not streaming:
            assert completed == [10]


@pytest.mark.parametrize("reservation", [0, -1, True, 101, None])
async def test_invalid_token_estimate_fails_before_quota_or_provider_admission(example, reservation):
    module, _clock = example
    limiter = module.GlobalRateLimit(
        quota_key="project", rpm=10, tpm=100,
        token_reservation=lambda _ctx, _request: reservation,
    )
    with pytest.raises(ValueError, match="token_reservation"):
        await _request(limiter)
    assert not limiter._admissions and not limiter._charges
