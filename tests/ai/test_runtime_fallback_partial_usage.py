#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""A failed continuation adds attempt usage without rebilling its partial response."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic_ai import Agent, messages as sdk_messages
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError
from linktools.ai.model import FallbackBinding
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.spec import AgentUsageLimits

from .test_runtime_fallback_failures import _Binding
from .test_model_request_lifecycle import _LifecycleRecorder


pytestmark = pytest.mark.skipif(
    not hasattr(sdk_messages, "ModelRequestAttempt"),
    reason="installed SDK does not expose fallback attempt observations",
)


class _Response(CompletedStreamedResponse):
    def get(self) -> ModelResponse:
        return replace(super().get(), metadata=self.metadata)


class _PartialThenFailure(FunctionModel):
    def __init__(
        self, requests: list[str], *, earlier_attempt: bool,
        partial_input_tokens: int = 5, partial_output_tokens: int = 2,
    ) -> None:
        super().__init__(model_name="partial", stream_function=self._unused_stream)
        self.requests = requests
        self.earlier_attempt = earlier_attempt
        self.partial_input_tokens = partial_input_tokens
        self.partial_output_tokens = partial_output_tokens
        self.calls = 0

    async def _unused_stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError("the completed response adapter owns streaming")
        yield ""  # pragma: no cover

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        self.calls += 1
        if self.calls == 1:
            response = ModelResponse(
                [TextPart("working ")], model_name="partial", provider_name="function",
                provider_response_id="first", state="suspended",
                usage=RequestUsage(
                    input_tokens=self.partial_input_tokens, output_tokens=self.partial_output_tokens,
                    cost=Decimal("0.005"),
                ),
            )
            if self.earlier_attempt:
                def early_rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                    del messages, info
                    self.requests.append("early-rejected")
                    return ModelResponse(
                        [TextPart("earlier rejected response")], model_name="early-rejected",
                        provider_name="function", usage=RequestUsage(input_tokens=2, cost=Decimal("0.002")),
                    )

                def partial(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                    del messages, info
                    return response

                def reject_early(value: ModelResponse) -> bool:
                    return value.model_name == "early-rejected"

                native = FallbackModel(
                    FunctionModel(early_rejected, model_name="early-rejected"),
                    FunctionModel(partial, model_name="partial"), fallback_on=reject_early,
                )
                response = await native.request(messages, model_settings, model_request_parameters)
            self.requests.append("partial")
            yield _Response(response, model_request_parameters=model_request_parameters, replay_events=True)
            return

        def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            self.requests.append("rejected")
            return ModelResponse(
                [TextPart("rejected synthetic response")], model_name="rejected", provider_name="function",
                usage=RequestUsage(input_tokens=4, output_tokens=1, cost=Decimal("0.004")),
            )

        def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            self.requests.append("unavailable")
            raise ModelAPIError("unavailable", "synthetic provider error")

        def reject(response: ModelResponse) -> bool:
            del response
            return True

        # Real native response rejection produces measured attempts. The outer
        # fallback preserves this failure without rewinding its pinned candidate.
        native = FallbackModel(
            FunctionModel(rejected, model_name="rejected"), FunctionModel(fail, model_name="unavailable"),
            fallback_on=[reject, ModelAPIError],
        )
        await native.request(messages, model_settings, model_request_parameters)
        raise AssertionError("the fallback chain must be exhausted")


class _CostLimitCapability(AbstractCapability[AgentContext[None]]):
    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert ctx.usage_limits is not None
        ctx.usage_limits.cost_limit = Decimal("0.008")


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "earlier_attempt"), [
    (None, False), ("tokens", False), ("cost", False), (None, True),
])
async def test_partial_response_and_exhausted_attempts_are_counted_once_and_checked_after_unwind(
    tmp_path: Path, limit: str | None, earlier_attempt: bool,
) -> None:
    requests: list[str] = []

    def features() -> RuntimeFeatures:
        group = RuntimeFeatures("partial-fallback")
        group.agent(
            "default", allow_tools=(), allow_skills=(), allow_subagents=(),
            usage_limits=AgentUsageLimits(input_tokens=8) if limit == "tokens" else None,
        )
        group.model("default", FallbackBinding(
            _Binding(_PartialThenFailure(requests, earlier_attempt=earlier_attempt)), fallback_on=FallbackExceptionGroup,
        ))
        group.capability(TraceCapability(), scope="runtime")
        if limit == "cost":
            group.capability(_CostLimitCapability(), id="partial-cost-limit", scope="runtime")
        return group

    async with Persistence.open('partial-fallback', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_159, Runtime.attach(persistence=persistence_159, features=features()) as runtime:
        execution = await runtime.agents.get().start("respond")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ("INTERNAL_ERROR" if limit is None else "EXECUTION_USAGE_LIMIT_EXCEEDED")
        # The accepted partial response contributes 5/2; its one failed native
        # chain contributes 4/1. The pinned candidate is never replayed.
        expected_input_tokens = 11 if earlier_attempt else 9
        assert result.usage.input_tokens == expected_input_tokens
        assert result.usage.output_tokens == 3
        assert result.usage.model_requests == 1
        attempts = result.safe_error_details["failed_attempts"]
        assert [item["outcome"] for item in attempts] == ["rejected", "error"]
        assert [item["usage"]["input_tokens"] for item in attempts if item["usage"] is not None] == [4]
        assert [item["usage"]["cost"] for item in attempts if item["usage"] is not None] == ["0.004"]
        interactions = (await execution.model_interactions(include_content=True)).items
        assert len(interactions) == 1
        assert interactions[0].model_request_seq == result.safe_error_details["model_request_seq"] == 1
        assert interactions[0].status == "FAILED"
        assert interactions[0].error_code == result.error_code
        # Continuation segments belong to this same logical request, so its
        # projection includes the accepted partial response as well as attempts.
        assert interactions[0].usage.input_tokens == expected_input_tokens
        assert interactions[0].usage.output_tokens == 3
        assert interactions[0].usage.model_requests == 1
        execution_id, principal = execution.execution_id, runtime.default_principal

    async with Persistence.open('partial-fallback', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_188, Runtime.attach(persistence=persistence_188, features=features()) as runtime:
        restored = await runtime.executions.result(execution_id, principal=principal)
        assert restored.usage == result.usage
        assert restored.error_code == result.error_code
        assert restored.safe_error_details == result.safe_error_details
        assert restored.error_diagnostics == result.error_diagnostics
        restored_interactions = (await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True,
        )).items
        assert restored_interactions[0].usage == interactions[0].usage
    assert requests == (["early-rejected"] if earlier_attempt else []) + [
        "partial", "rejected", "unavailable",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("current_usage", ["attempts", "partial", "zero_partial", "none"])
async def test_native_resume_observation_excludes_seed_usage_and_preserves_unknown_attempt_usage(
    current_usage: str,
) -> None:
    """Exercise the actual SDK resume boundary, not a Runtime recovery API."""
    recorder = _LifecycleRecorder()
    observation = ModelObservationCapability(
        None, source_namespace="resume", tenant_id="default", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="agent", interaction_recorder=recorder,
    )
    requests: list[str] = []

    async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        requests.append("unavailable")
        raise ModelAPIError("unavailable", "synthetic provider error")
        yield ""  # pragma: no cover

    class PartialThenUnmeasuredFailure(_PartialThenFailure):
        @asynccontextmanager
        async def request_stream(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            if self.calls:
                requests.append("unavailable")
                raise ModelAPIError("partial", "unmeasured continuation error")
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context,
            ) as response:
                yield response

    if current_usage == "attempts":
        model = FallbackModel(
            _PartialThenFailure(requests, earlier_attempt=False, partial_input_tokens=3),
            fallback_on=FallbackExceptionGroup,
        )
    elif current_usage in {"partial", "zero_partial"}:
        model = FallbackModel(PartialThenUnmeasuredFailure(
            requests, earlier_attempt=False,
            partial_input_tokens=3 if current_usage == "partial" else 0,
            partial_output_tokens=2 if current_usage == "partial" else 0,
        ))
    else:
        model = FallbackModel(FunctionModel(stream_function=fail, model_name="unavailable"))
    seed = ModelResponse(
        [TextPart("historical ")], model_name="partial", provider_name="function",
        provider_response_id="historical", state="suspended",
        usage=RequestUsage(input_tokens=5, output_tokens=2, cost=Decimal("0.005")),
    )
    usage = RunUsage()

    async def drain(ctx: RunContext[None], events: AsyncIterator[object]) -> None:
        del ctx
        async for _ in events:
            pass

    with pytest.raises(AIError) as caught:
        await Agent(model, capabilities=[observation]).run(
            message_history=[ModelRequest([UserPromptPart("historical prompt")]), seed],
            usage=usage, event_stream_handler=drain,
        )
    assert list(recorder.values) == [1]
    assert recorder.values[1]["status"] == "FAILED"
    observed = recorder.values[1]["usage"]
    if current_usage == "attempts":
        # The seed's five tokens were billed before this request. Its new
        # partial adds three and exhausted attempts add eight, not sixteen.
        assert usage.input_tokens == observed.input_tokens == 11
        assert usage.output_tokens == observed.output_tokens == 4
        assert requests == ["partial", "rejected", "unavailable", "rejected", "unavailable"]
    elif current_usage in {"partial", "zero_partial"}:
        assert observed is not None
        assert usage.input_tokens == observed.input_tokens == (3 if current_usage == "partial" else 0)
        assert usage.output_tokens == observed.output_tokens == (2 if current_usage == "partial" else 0)
        assert all(item["usage"] is None for item in caught.value.safe_details["failed_attempts"])
        assert requests == ["partial", "unavailable", "unavailable"]
    else:
        assert usage.input_tokens == usage.output_tokens == 0
        assert all(item["usage"] is None for item in caught.value.safe_details["failed_attempts"])
        assert observed is None
        assert requests == ["unavailable"]
