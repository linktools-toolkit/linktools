#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Pinned continuations cannot reset candidate budgets or accept rejected output."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from functools import partial
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from linktools.ai.core import ExecutionStatus
from linktools.ai.model import FallbackBinding
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models
from .test_fallback_retry_ownership import _features
from .test_runtime_fallback_failures import _CompletedFallback
from .test_runtime_fallback_partial_usage import _Response


class _Candidate(FunctionModel):
    def __init__(
        self, name: str, calls: list[str], cancelled: list[tuple[str, str | None]],
        *, outcome: str = "failure", same_id: bool = False,
    ) -> None:
        self.calls, self.cancelled = calls, cancelled
        self.outcome, self.same_id = outcome, same_id
        self.count = 0
        self.error = ModelHTTPError(503, name, body={"error": "Bearer synthetic-continuation-secret"})
        super().__init__(self.respond, model_name=name)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        self.calls.append(self.model_name)
        self.count += 1
        initial = self.model_name == "primary" and self.count == 1
        if not initial and self.outcome == "failure":
            raise self.error
        return ModelResponse(
            [TextPart("partial" if initial else "finished")], model_name=self.model_name,
            provider_name="function", provider_response_id="old-job" if initial or self.same_id else "new-job",
            state="suspended" if initial or self.outcome == "suspended" else "complete",
            usage=RequestUsage(
                input_tokens=5 if initial else 8, output_tokens=2 if initial else 4,
                cost=Decimal(".005") if initial else Decimal(".008"), details={"known": 5 if initial else 8},
            ),
        )

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: object | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        response = await self.request(messages, model_settings, model_request_parameters)
        yield _Response(response, model_request_parameters=model_request_parameters, replay_events=True)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append((self.model_name, response.provider_response_id))


def _binding(
    calls: list[str], cancelled: list[tuple[str, str | None]], *, retries: int = 2,
    outcome: str = "failure", same_id: bool = False, fallback_on: object = (ModelAPIError,),
) -> tuple[FallbackBinding, _Candidate]:
    primary = _Candidate("primary", calls, cancelled, outcome=outcome, same_id=same_id)
    leaves = [_Models(_RetryingModel(candidate, retries, 0)) for candidate in (
        primary, _Candidate("secondary", calls, cancelled), _Candidate("third", calls, cancelled),
    )]
    return FallbackBinding(*leaves, fallback_on=fallback_on), primary


async def _native_checks_continuation_responses() -> bool:
    seen: list[str] = []
    def accept(response: ModelResponse) -> bool:
        seen.append(response.state)
        return False
    candidate = _Candidate("primary", [], [], outcome="success")
    native = FallbackModel(candidate, fallback_on=accept)
    messages = [ModelRequest([UserPromptPart("synthetic")])]
    initial = await native.request(messages, None, ModelRequestParameters())
    await native.request([*messages, initial], None, ModelRequestParameters())
    return len(seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("request", "stream", "runtime"))
@pytest.mark.parametrize("retries", (0, 2))
async def test_pinned_failure_uses_transport_budget_only_once(path: str, retries: int) -> None:
    calls: list[str] = []
    binding, primary = _binding(calls, [], retries=retries)
    if path == "runtime":
        async with Persistence.open('continuation', storage=PersistenceStorage.in_memory()) as persistence_105, Runtime.attach(persistence=persistence_105, features=_features(binding)) as app:
            execution = await app.agents.get().start("synthetic")
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == "MODEL_UNAVAILABLE"
            assert "synthetic-continuation-secret" not in repr(result.safe_error_details)
            assert "synthetic-continuation-secret" not in repr(result.error_diagnostics)
    else:
        async with binding.materialize() as model:
            messages = [ModelRequest([UserPromptPart("synthetic")])]
            if path == "request":
                response = await model.request(messages, None, ModelRequestParameters())
                with pytest.raises(ModelHTTPError) as failed:
                    await model.request([*messages, response], None, ModelRequestParameters())
            else:
                async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
                    async for _ in stream:
                        pass
                    response = stream.get()
                with pytest.raises(ModelHTTPError) as failed:
                    async with model.request_stream([*messages, response], None, ModelRequestParameters()):
                        pytest.fail("failed continuation cannot enter a stream")
            assert failed.value is primary.error
    # A suspended stream can have provider-side effects even before its next event.
    # Non-stream request retry ownership is unchanged.
    assert calls == ["primary"] * (retries + 2 if path == "request" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("request", "runtime"))
async def test_successful_pinned_continuation_is_not_disabled(path: str) -> None:
    calls: list[str] = []
    binding, _ = _binding(calls, [], outcome="success")
    if path == "runtime":
        async with Persistence.open('continuation', storage=PersistenceStorage.in_memory()) as persistence_139, Runtime.attach(persistence=persistence_139, features=_features(binding)) as app:
            result = (await (await app.agents.get().start("synthetic")).wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.SUCCEEDED
    else:
        async with binding.materialize() as model:
            messages = [ModelRequest([UserPromptPart("synthetic")])]
            response = await model.request(messages, None, ModelRequestParameters())
            assert (await model.request([*messages, response], None, ModelRequestParameters())).state == "complete"
    assert calls == ["primary", "primary"]


class _ResponsePolicy:
    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    async def __call__(self, response: ModelResponse) -> bool:
        self.seen.append(response.state)
        return response.parts[0].content == "finished"


def _response_policy(kind: str, seen: list[str]) -> object:
    def check(enabled: bool, response: ModelResponse) -> bool:
        seen.append(response.state)
        return enabled and response.parts[0].content == "finished"
    async def asynchronous(response: ModelResponse) -> bool:
        return check(True, response)
    return {"partial": partial(check, True), "async": asynchronous, "instance": _ResponsePolicy(seen)}[kind]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("partial", "async", "instance"))
@pytest.mark.parametrize("same_id", (False, True))
async def test_native_response_rejection_is_terminal_without_rebilling_partial(kind: str, same_id: bool) -> None:
    checks_continuations = await _native_checks_continuation_responses()
    calls: list[str] = []
    seen: list[str] = []
    binding, _ = _binding(calls, [], outcome="success", same_id=same_id, fallback_on=_response_policy(kind, seen))
    usage = RunUsage()
    async with Agent(binding.materialize()) as agent:
        if not checks_continuations:
            assert (await agent.run("synthetic", usage=usage)).output
            assert seen == ["suspended"]
        else:
            with pytest.raises(FallbackExceptionGroup) as failed:
                await agent.run("synthetic", usage=usage)
            attempt = failed.value.attempts[0]
            assert attempt.outcome == "rejected"
            assert attempt.timestamp is not None and attempt.duration.total_seconds() > 0
            assert attempt.usage.input_tokens == (3 if same_id else 8)
            assert attempt.usage.cost == (Decimal(".003") if same_id else Decimal(".008"))
            assert seen == ["suspended", "complete"]
    assert calls == ["primary", "primary"]
    assert usage.input_tokens == (8 if same_id else 13)
    assert usage.output_tokens == (4 if same_id else 6)
    assert usage.cost == (Decimal(".008") if same_id else Decimal(".013"))


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
@pytest.mark.parametrize("limit", ("tokens", "cost"))
async def test_rejected_continuation_preserves_native_usage_limits(same_id: bool, limit: str) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    binding, _ = _binding([], [], outcome="success", same_id=same_id, fallback_on=_ResponsePolicy([]))
    limits = UsageLimits(input_tokens_limit=10) if limit == "tokens" else UsageLimits(cost_limit=Decimal(".010"))
    usage = RunUsage()
    async with Agent(binding.materialize()) as agent:
        with pytest.raises(FallbackExceptionGroup if same_id else UsageLimitExceeded) as failed:
            await agent.run("synthetic", usage=usage, usage_limits=limits)
    group = failed.value if same_id else failed.value.__cause__
    assert isinstance(group, FallbackExceptionGroup) and len(group.attempts) == 1
    assert usage.input_tokens == (8 if same_id else 13)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
async def test_rejected_new_job_cancels_only_its_pinned_candidate(same_id: bool) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    calls: list[str] = []
    cancelled: list[tuple[str, str | None]] = []
    binding, _ = _binding(calls, cancelled, outcome="suspended", same_id=same_id, fallback_on=_ResponsePolicy([]))
    async with Agent(binding.materialize()) as agent:
        with pytest.raises(FallbackExceptionGroup):
            await agent.run("synthetic")
    assert calls == ["primary", "primary"]
    assert cancelled == ([("primary", "old-job")] if same_id else
                         [("primary", "new-job"), ("primary", "old-job")])

@pytest.mark.asyncio
async def test_concurrent_fresh_request_does_not_inherit_continuation_policy() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[str] = []
    failure = ModelHTTPError(503, "primary")
    async def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if isinstance(messages[-1], ModelResponse):
            calls.append("continuation")
            entered.set()
            await release.wait()
            raise failure
        prompt = messages[-1].parts[0].content
        calls.append(prompt)
        if prompt == "initial":
            return ModelResponse([TextPart("partial")], state="suspended", provider_response_id="job")
        raise failure
    def secondary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("secondary")
        return ModelResponse([TextPart("fresh")])
    binding = FallbackBinding(_Models(FunctionModel(primary, model_name="primary")),
                              _Models(FunctionModel(secondary, model_name="secondary")))
    async with binding.materialize() as model:
        initial = [ModelRequest([UserPromptPart("initial")])]
        response = await model.request(initial, None, ModelRequestParameters())
        continuation = asyncio.create_task(model.request([*initial, response], None, ModelRequestParameters()))
        await asyncio.wait_for(entered.wait(), 5)
        try:
            fresh = await model.request([ModelRequest([UserPromptPart("fresh")])], None, ModelRequestParameters())
            assert fresh.parts[0].content == "fresh"
        finally:
            release.set()
        with pytest.raises(ModelHTTPError) as failed:
            await continuation
        assert failed.value is failure
        assert (await model.request([ModelRequest([UserPromptPart("again")])], None,
                                    ModelRequestParameters())).parts[0].content == "fresh"
    assert calls == ["initial", "continuation", "fresh", "secondary", "again", "secondary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
async def test_runtime_rejected_continuation_retains_usage_and_diagnostics_after_reopen(
    tmp_path: Path, same_id: bool,
) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    calls: list[str] = []
    binding, _ = _binding(calls, [], outcome="success", same_id=same_id, fallback_on=_ResponsePolicy([]))
    # Runtime observes streams; this adapter exercises native nonstream policy
    # through the same completed-response boundary used by custom models.
    features = _features(_Models(_CompletedFallback(binding.materialize())))
    async with Persistence.open('continuation-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_281, Runtime.attach(persistence=persistence_281, features=features) as app:
        execution = await app.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == "INTERNAL_ERROR"
        assert result.usage.input_tokens == (8 if same_id else 13)
        assert result.safe_error_details["failed_attempts"][0]["usage"]["input_tokens"] == (3 if same_id else 8)
        assert result.error_diagnostics is not None
        assert "synthetic-continuation-secret" not in repr(result.error_diagnostics)
        execution_id, principal = execution.execution_id, app.default_principal
    async with Persistence.open('continuation-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_291, Runtime.attach(persistence=persistence_291, features=features) as app:
        restored = await app.executions.result(execution_id, principal=principal)
        assert restored.usage == result.usage
        assert restored.safe_error_details == result.safe_error_details
        assert restored.error_code == result.error_code
    assert calls == ["primary", "primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("success", (False, True))
async def test_nonfirst_candidate_continuation_does_not_rewind(success: bool) -> None:
    calls: list[str] = []
    count = 0
    failure = ModelHTTPError(503, "secondary")

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("primary")
        raise ModelHTTPError(503, "primary")

    def secondary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal count
        del messages, info
        calls.append("secondary")
        count += 1
        if count == 1:
            return ModelResponse([TextPart("partial")], state="suspended", provider_response_id="second-job",
                                 usage=RequestUsage(input_tokens=5))
        if not success:
            raise failure
        return ModelResponse([TextPart("finished")], usage=RequestUsage(input_tokens=8))

    def third(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        pytest.fail("the unused third candidate must not be requested")

    binding = FallbackBinding(*[
        _Models(_RetryingModel(FunctionModel(function, model_name=name), 2, 0))
        for name, function in (("primary", primary), ("secondary", secondary), ("third", third))
    ])
    async with Agent(binding.materialize()) as agent:
        if success:
            assert (await agent.run("synthetic")).output == "partialfinished"
        else:
            with pytest.raises(ModelHTTPError) as failed:
                await agent.run("synthetic")
            assert failed.value is failure
    assert calls == ["primary"] * 3 + ["secondary"] * (2 if success else 4)


@pytest.mark.asyncio
async def test_rejected_native_response_metadata_cannot_redirect_new_job_cleanup() -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    calls: list[str] = []
    cancelled: list[tuple[str, str | None]] = []
    returned: list[ModelResponse] = []
    original_metadata: list[dict[str, object]] = []

    class Candidate(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append((self.model_name, response.provider_response_id))

    def foreign(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse([TextPart("finished")], state="suspended", provider_response_id="new-job",
                             metadata={"business": "unchanged"})

    async def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        calls.append("primary")
        if len(calls) == 1:
            return ModelResponse([TextPart("partial")], state="suspended", provider_response_id="old-job")
        # A custom model may legitimately forward a public SDK composition's
        # response. Its opaque metadata must neither be merged nor interpreted.
        response = await FallbackModel(FunctionModel(foreign, model_name="foreign")).request(
            messages, None, ModelRequestParameters(),
        )
        returned.append(response)
        original_metadata.append(dict(response.metadata))
        return response

    def secondary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        pytest.fail("the backup candidate must not be requested")

    binding = FallbackBinding(
        _Models(Candidate(primary, model_name="primary")),
        _Models(Candidate(secondary, model_name="secondary")), fallback_on=_ResponsePolicy([]),
    )
    async with Agent(binding.materialize()) as agent:
        with pytest.raises(FallbackExceptionGroup):
            await agent.run("synthetic")
    assert calls == ["primary", "primary"]
    assert cancelled == [("primary", "new-job"), ("primary", "old-job")]
    assert returned[0].metadata == original_metadata[0]
    assert returned[0].parts == [TextPart("finished")]


@pytest.mark.asyncio
async def test_cancelled_continuation_backoff_does_not_replay_or_poison_next_request(monkeypatch) -> None:
    sleeping = asyncio.Event()
    calls: list[str] = []
    cancelled: list[tuple[str, str | None]] = []
    candidate = _Candidate("primary", calls, cancelled)
    primary = _RetryingModel(candidate, 2, 60)
    secondary = _Candidate("secondary", calls, cancelled, outcome="success")
    binding = FallbackBinding(_Models(primary), _Models(secondary))
    original_sleep = asyncio.sleep

    async def wait(delay: float) -> None:
        if delay:
            sleeping.set()
            await asyncio.Event().wait()
        else:
            await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", wait)
    async with Agent(binding.materialize()) as agent:
        task = asyncio.create_task(agent.run("synthetic"))
        await asyncio.wait_for(sleeping.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls == ["primary", "primary"]
    assert cancelled == [("primary", "old-job")]
    # The binding can be used again after cancellation without another retry.
    candidate.outcome = "success"
    async with binding.materialize() as model:
        response = await model.request([ModelRequest([UserPromptPart("fresh")])], None, ModelRequestParameters())
    assert response.state == "complete"
    assert calls == ["primary", "primary", "primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
@pytest.mark.parametrize("decrease", ("tokens", "cost"))
async def test_runtime_decreasing_cumulative_usage_is_invalid_but_new_job_usage_is_additive(
    tmp_path: Path, same_id: bool, decrease: str,
) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    calls: list[str] = []
    responses: list[ModelResponse] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("primary")
        initial = len(calls) == 1
        response = ModelResponse(
            [TextPart("partial" if initial else "finished")], model_name="primary", provider_name="function",
            provider_response_id="old-job" if initial or same_id else "new-job",
            state="suspended" if initial else "complete",
            usage=RequestUsage(input_tokens=5 if initial else (3 if decrease == "tokens" else 8),
                               cost=Decimal(".005") if initial else Decimal(".003" if decrease == "cost" else ".008")),
        )
        responses.append(response)
        return response

    binding = FallbackBinding(_Models(FunctionModel(respond, model_name="primary")), fallback_on=_ResponsePolicy([]))
    features = _features(_Models(_CompletedFallback(binding.materialize())))
    async with Persistence.open('invalid-usage', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_450, Runtime.attach(persistence=persistence_450, features=features) as app:
        execution = await app.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.FAILED
        if same_id:
            assert result.error_code == "MODEL_RESPONSE_INVALID"
            assert result.usage.input_tokens == 5
            assert result.safe_error_details["reason"] == "continuation_usage_inconsistent"
            assert result.safe_error_details["usage_omission_reason"] == "cumulative_usage_decreased"
            assert result.safe_error_details["previous_usage"]["input_tokens"] == 5
            assert Decimal(result.safe_error_details["previous_usage"]["cost"]) == Decimal(".005")
            assert result.safe_error_details["current_usage"]["input_tokens"] == responses[1].usage.input_tokens
            assert Decimal(result.safe_error_details["current_usage"]["cost"]) == responses[1].usage.cost
        else:
            assert result.error_code == "INTERNAL_ERROR"
            assert result.usage.input_tokens == 5 + responses[1].usage.input_tokens
            assert Decimal(result.safe_error_details["failed_attempts"][0]["usage"]["cost"]) == responses[1].usage.cost
        execution_id, principal = execution.execution_id, app.default_principal
    async with Persistence.open('invalid-usage', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_468, Runtime.attach(persistence=persistence_468, features=features) as app:
        restored = await app.executions.result(execution_id, principal=principal)
        assert restored.usage == result.usage
        assert restored.safe_error_details == result.safe_error_details
        assert restored.error_code == result.error_code
    assert calls == ["primary", "primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", ("neither", "previous", "current"))
async def test_cumulative_unknown_usage_and_decreasing_details_do_not_invent_invalid_counters(unknown: str) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    observed: list[ModelResponse] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        initial = not observed
        usage = RequestUsage(input_tokens=5 if initial else 8, cost=Decimal(".005"),
                             details={"diagnostic": 9 if initial else 4}, diagnostic=9 if initial else 4)
        if unknown == ("previous" if initial else "current"):
            usage = RequestUsage(unmeasured_requests=1)
        response = ModelResponse([TextPart("partial" if initial else "finished")], model_name="primary",
                                 provider_response_id="job", state="suspended" if initial else "complete", usage=usage)
        observed.append(response)
        return response

    binding = FallbackBinding(_Models(FunctionModel(respond, model_name="primary")), fallback_on=_ResponsePolicy([]))
    usage = RunUsage()
    async with Agent(binding.materialize()) as agent:
        with pytest.raises(FallbackExceptionGroup) as failed:
            await agent.run("synthetic", usage=usage)
    attempt_usage = failed.value.attempts[0].usage
    if unknown == "neither":
        assert attempt_usage.input_tokens == 3
        assert attempt_usage.cost == Decimal(0)
        assert attempt_usage.details == {"diagnostic": -5}
        assert attempt_usage.diagnostic == -5
        assert usage.input_tokens == 8 and usage.cost == Decimal(".005")
    else:
        assert attempt_usage is None
        assert usage.input_tokens == (0 if unknown == "previous" else 5)
    # Local accounting never mutates the provider's observed usage objects.
    if unknown != "current":
        assert observed[1].usage.input_tokens == 8
        assert observed[1].usage.cost == Decimal(".005")
        assert observed[1].usage.details == {"diagnostic": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", ("previous", "current"))
async def test_runtime_unknown_continuation_increment_remains_unknown_after_reopen(
    tmp_path: Path, unknown: str,
) -> None:
    if not await _native_checks_continuation_responses():
        pytest.skip("installed SDK does not apply response handlers to pinned continuations")
    calls: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        initial = not calls
        calls.append("primary")
        usage = RequestUsage(input_tokens=5 if initial else 8)
        if unknown == ("previous" if initial else "current"):
            usage = RequestUsage(unmeasured_requests=1)
        return ModelResponse(
            [TextPart("partial" if initial else "finished")], model_name="primary",
            provider_response_id="job", state="suspended" if initial else "complete", usage=usage,
        )

    binding = FallbackBinding(_Models(FunctionModel(respond, model_name="primary")), fallback_on=_ResponsePolicy([]))
    features = _features(_Models(_CompletedFallback(binding.materialize())))
    async with Persistence.open('unknown-increment', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_540, Runtime.attach(persistence=persistence_540, features=features) as app:
        execution = await app.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.safe_error_details["failed_attempts"][0]["usage"] is None
        # The aggregate preserves only known contributions, not a fabricated
        # measured zero for the rejected attempt.
        assert result.usage.input_tokens == (0 if unknown == "previous" else 5)
        execution_id, principal = execution.execution_id, app.default_principal
    async with Persistence.open('unknown-increment', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_549, Runtime.attach(persistence=persistence_549, features=features) as app:
        restored = await app.executions.result(execution_id, principal=principal)
        assert restored.usage == result.usage
        assert restored.safe_error_details == result.safe_error_details
        assert restored.safe_error_details["failed_attempts"][0]["usage"] is None
    assert calls == ["primary", "primary"]
